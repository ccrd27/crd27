#include "rcd/crd.hpp"

#include "rcd/progress.hpp"
#include "rcd/rr_sets.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

namespace rcd {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

struct LinearContext {
    double constant = 0.0;
    std::vector<Weight> coverage_weights;
    std::size_t positive_root_weights = 0;
};

std::vector<std::uint8_t> seed_mask(
    std::size_t num_nodes,
    const std::vector<NodeId>& seeds) {
    std::vector<std::uint8_t> mask(num_nodes, 0);
    for (const NodeId seed : seeds) {
        if (static_cast<std::size_t>(seed) >= num_nodes) {
            throw std::out_of_range("CRD-Linear seed is outside graph");
        }
        mask[seed] = 1;
    }
    return mask;
}

LinearContext build_linear_context(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const std::vector<NodeId>& opponent_seeds,
    OptimizedPlayer optimized_player) {
    const auto opponent_mask = seed_mask(graph.num_nodes(), opponent_seeds);
    const ConditionalTerms terms = build_conditional_terms(
        conflict, innate, opponent_mask, params, optimized_player);
    const double orientation =
        optimized_player == OptimizedPlayer::Attacker ? 1.0 : -1.0;

    LinearContext context;
    context.constant = terms.constant;
    context.coverage_weights.assign(static_cast<std::size_t>(terms.ell.size()), 0.0);
    for (Eigen::Index i = 0; i < terms.ell.size(); ++i) {
        const double weight = orientation * terms.ell[i];
        if (!std::isfinite(weight)) {
            throw std::runtime_error("CRD-Linear produced a non-finite root weight");
        }
        if (weight > 0.0) {
            context.coverage_weights[static_cast<std::size_t>(i)] = weight;
            ++context.positive_root_weights;
        }
    }
    return context;
}

std::vector<NodeId> all_nodes(const DirectedGraph& graph) {
    std::vector<NodeId> candidates(graph.num_nodes());
    for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
        candidates[i] = static_cast<NodeId>(i);
    }
    return candidates;
}

std::vector<std::uint8_t> candidate_mask(
    std::size_t num_nodes,
    const std::vector<NodeId>& candidates) {
    std::vector<std::uint8_t> mask(num_nodes, 0);
    for (const NodeId candidate : candidates) {
        if (static_cast<std::size_t>(candidate) >= num_nodes) {
            throw std::out_of_range("CRD-Linear candidate is outside graph");
        }
        mask[candidate] = 1;
    }
    return mask;
}

bool attack_pool_contains(
    const std::vector<AttackPoolEntry>& pool,
    const std::vector<NodeId>& attack) {
    std::vector<NodeId> key = attack;
    std::sort(key.begin(), key.end());
    return std::any_of(pool.begin(), pool.end(), [&key](const AttackPoolEntry& entry) {
        auto existing = entry.attack_seeds;
        std::sort(existing.begin(), existing.end());
        return existing == key;
    });
}

struct AttackSelection {
    std::vector<NodeId> seeds;
    double gain = 0.0;
    std::size_t sample_count = 0;
};

AttackSelection select_attack(
    const DirectedGraph& graph,
    const LinearContext& context,
    const CRDOptions& options,
    const std::vector<std::uint8_t>& candidates,
    std::uint64_t seed,
    const ProgressCallback& progress) {
    const auto samples = generate_weighted_rr_samples_from_node_weights(
        options.propagation_model,
        graph,
        options.attack_rr_samples,
        seed,
        context.coverage_weights,
        progress);
    AttackSelection result;
    result.seeds = weighted_coverage_greedy(
        graph.num_nodes(), samples, options.attack_budget, candidates, progress);
    std::sort(result.seeds.begin(), result.seeds.end());
    result.gain = weighted_coverage_value(samples, result.seeds);
    result.sample_count = samples.size();
    return result;
}

struct DefensePool {
    double constant = 0.0;
    std::vector<WeightedRRSample> samples;
    std::vector<std::vector<RRIndex>> covered_by_node;
    std::vector<std::uint8_t> covered;
    double covered_weight_sum = 0.0;
};

DefensePool make_defense_pool(
    const DirectedGraph& graph,
    const LinearContext& context,
    const CRDOptions& options,
    std::uint64_t seed,
    const ProgressCallback& progress) {
    DefensePool pool;
    pool.constant = context.constant;
    pool.samples = generate_weighted_rr_samples_from_node_weights(
        options.propagation_model,
        graph,
        options.defense_rr_samples,
        seed,
        context.coverage_weights,
        progress);
    pool.covered_by_node.assign(graph.num_nodes(), {});
    pool.covered.assign(pool.samples.size(), 0);
    for (RRIndex sample_id = 0; sample_id < pool.samples.size(); ++sample_id) {
        for (const NodeId node : pool.samples[sample_id].rr_set) {
            pool.covered_by_node[node].push_back(sample_id);
        }
    }
    return pool;
}

double candidate_gain(const DefensePool& pool, NodeId node) {
    double gain = 0.0;
    for (const RRIndex sample_id : pool.covered_by_node[node]) {
        if (pool.covered[sample_id] == 0) {
            gain += pool.samples[sample_id].weight;
        }
    }
    return gain;
}

double residual_value(const DefensePool& pool, double gain = 0.0) {
    if (pool.samples.empty()) {
        throw std::invalid_argument("CRD-Linear defense pool has no RR samples");
    }
    return pool.constant -
        (pool.covered_weight_sum + gain) / static_cast<double>(pool.samples.size());
}

void apply_selection(DefensePool& pool, NodeId node) {
    for (const RRIndex sample_id : pool.covered_by_node[node]) {
        if (pool.covered[sample_id] == 0) {
            pool.covered[sample_id] = 1;
            pool.covered_weight_sum += pool.samples[sample_id].weight;
        }
    }
}

double worst_value(const std::vector<DefensePool>& pools) {
    double result = -std::numeric_limits<double>::infinity();
    for (const auto& pool : pools) {
        result = std::max(result, residual_value(pool));
    }
    return result;
}

struct QueueEntry {
    double upper_bound = 0.0;
    NodeId node = 0;
};

struct QueueEntryLess {
    bool operator()(const QueueEntry& left, const QueueEntry& right) const {
        if (left.upper_bound != right.upper_bound) {
            return left.upper_bound < right.upper_bound;
        }
        return left.node > right.node;
    }
};

double node_improvement_upper_bound(
    const std::vector<DefensePool>& pools,
    NodeId node) {
    double upper_bound = 0.0;
    for (const auto& pool : pools) {
        upper_bound = std::max(
            upper_bound,
            candidate_gain(pool, node) / static_cast<double>(pool.samples.size()));
    }
    return upper_bound;
}

struct DefenseSelection {
    std::vector<NodeId> seeds;
    double worst_residual = 0.0;
    std::size_t sample_count = 0;
};

DefenseSelection select_defense(
    const DirectedGraph& graph,
    const std::vector<LinearContext>& attack_contexts,
    const CRDOptions& options,
    const std::vector<std::uint8_t>& candidates,
    std::uint64_t seed,
    const ProgressCallback& progress) {
    std::vector<DefensePool> pools;
    pools.reserve(attack_contexts.size());
    for (std::size_t i = 0; i < attack_contexts.size(); ++i) {
        const auto pool_progress = [&progress, i, &attack_contexts](
                                       const std::string& stage, double percent) {
            if (progress) {
                progress(
                    "CRD-Linear defense pool " + std::to_string(i + 1) + "/" +
                        std::to_string(attack_contexts.size()) + ": " + stage,
                    percent);
            }
        };
        pools.push_back(make_defense_pool(
            graph,
            attack_contexts[i],
            options,
            seed ^ (0x9e3779b97f4a7c15ULL * (i + 1)),
            pool_progress));
    }

    const std::size_t progress_interval =
        std::max<std::size_t>(1, options.defense_budget / 10);
    std::vector<std::uint8_t> selected(graph.num_nodes(), 0);
    std::vector<NodeId> seeds;
    seeds.reserve(options.defense_budget);
    double current_worst = worst_value(pools);

    for (std::size_t round = 0; round < options.defense_budget; ++round) {
        std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryLess> queue;
        for (std::size_t node = 0; node < graph.num_nodes(); ++node) {
            const NodeId candidate = static_cast<NodeId>(node);
            if (candidates[candidate] != 0 && selected[candidate] == 0) {
                queue.push(QueueEntry{node_improvement_upper_bound(pools, candidate), candidate});
            }
        }

        NodeId best_node = 0;
        double best_improvement = 0.0;
        bool found = false;
        while (!queue.empty()) {
            const QueueEntry entry = queue.top();
            queue.pop();
            if (entry.upper_bound <= best_improvement + 1e-15) {
                break;
            }

            double candidate_worst = -std::numeric_limits<double>::infinity();
            for (const auto& pool : pools) {
                candidate_worst = std::max(
                    candidate_worst,
                    residual_value(pool, candidate_gain(pool, entry.node)));
            }
            const double improvement = current_worst - candidate_worst;
            if (!found || improvement > best_improvement + 1e-15 ||
                (std::abs(improvement - best_improvement) <= 1e-15 && entry.node < best_node)) {
                best_node = entry.node;
                best_improvement = improvement;
                found = true;
            }
        }

        if (!found || best_improvement <= 1e-15) {
            break;
        }
        selected[best_node] = 1;
        seeds.push_back(best_node);
        for (auto& pool : pools) {
            apply_selection(pool, best_node);
        }
        current_worst = worst_value(pools);
        if (progress &&
            (seeds.size() % progress_interval == 0 || seeds.size() == options.defense_budget)) {
            progress(
                "CRD-Linear defense: lazy greedy selection",
                100.0 * static_cast<double>(seeds.size()) /
                    static_cast<double>(std::max<std::size_t>(options.defense_budget, 1)));
        }
    }

    DefenseSelection result;
    std::sort(seeds.begin(), seeds.end());
    result.seeds = std::move(seeds);
    result.worst_residual = current_worst;
    for (const auto& pool : pools) {
        result.sample_count += pool.samples.size();
    }
    return result;
}

}  // namespace

CRDResult run_crd_linear(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const CRDOptions& options,
    std::ostream* progress_os) {
    if (options.variant != CRDVariant::LinearApprox) {
        throw std::invalid_argument("CRD-Linear requires the linear_approx variant");
    }
    if (options.crd_mode != CRDMode::Engineering) {
        throw std::invalid_argument("CRD-Linear supports engineering mode only");
    }
    if (graph.num_nodes() == 0 || options.attack_rr_samples == 0 ||
        options.defense_rr_samples == 0) {
        throw std::invalid_argument("CRD-Linear requires a nonempty graph and positive RR counts");
    }

    AlgorithmTimer algorithm_timer;
    const auto progress = [&](const std::string& stage, double percent) {
        if (progress_os == nullptr) {
            return;
        }
        algorithm_timer.pause_for_output();
        print_progress_line(*progress_os, stage, percent);
        algorithm_timer.resume_after_output();
    };

    CRDResult result;
    result.variant = CRDVariant::LinearApprox;
    const auto candidates = candidate_mask(graph.num_nodes(), all_nodes(graph));

    const auto empty_defense_context = build_linear_context(
        graph, conflict, innate, params, {}, OptimizedPlayer::Attacker);
    const auto initial_attack = select_attack(
        graph,
        empty_defense_context,
        options,
        candidates,
        options.seed ^ 0x452821e638d01377ULL,
        progress);
    result.attack_pool.push_back(AttackPoolEntry{initial_attack.seeds, std::numeric_limits<double>::quiet_NaN()});
    std::vector<LinearContext> defense_contexts;
    defense_contexts.push_back(build_linear_context(
        graph, conflict, innate, params, initial_attack.seeds, OptimizedPlayer::Defender));
    result.initial_attack_pool_policy = "empty_defense_linear_weighted_rr";
    result.initial_attack_pool_size = 1;
    result.attack_seeds = initial_attack.seeds;

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "CRD-Linear warm start\n"
                     << "  selection_mode: reference_seed_mask_linear_term\n"
                     << "  initial_attack_pool_size: 1\n"
                     << "  initial_attack_seeds: "
                     << format_seed_set_preview(initial_attack.seeds) << '\n'
                     << "  initial_attack_rr_samples: " << initial_attack.sample_count << '\n'
                     << "  initial_attack_positive_root_weights: "
                     << empty_defense_context.positive_root_weights << '\n';
        algorithm_timer.resume_after_output();
    }

    for (std::size_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
        const auto iteration_start = Clock::now();
        const std::size_t pool_size_before = result.attack_pool.size();
        const auto defense = select_defense(
            graph,
            defense_contexts,
            options,
            candidates,
            options.seed ^ (0xbe5466cf34e90c6cULL * (iteration + 1)),
            progress);
        const auto attack_context = build_linear_context(
            graph, conflict, innate, params, defense.seeds, OptimizedPlayer::Attacker);
        const auto attack = select_attack(
            graph,
            attack_context,
            options,
            candidates,
            options.seed ^ (0xc0ac29b7c97c50ddULL * (iteration + 1)),
            progress);
        const bool duplicate_attack = attack_pool_contains(result.attack_pool, attack.seeds);
        if (!duplicate_attack) {
            result.attack_pool.push_back(AttackPoolEntry{
                attack.seeds, std::numeric_limits<double>::quiet_NaN()});
            defense_contexts.push_back(build_linear_context(
                graph, conflict, innate, params, attack.seeds, OptimizedPlayer::Defender));
        }

        CRDIterationLog log;
        log.iteration = iteration;
        log.crd_mode = to_string(options.crd_mode);
        log.crd_variant = to_string(options.variant);
        log.theta_a = attack.sample_count;
        log.theta_b = defense.sample_count;
        log.theta_emp = 0;
        log.attack_pool_size_before_defense = pool_size_before;
        log.attack_pool_size_after_attack = result.attack_pool.size();
        log.attack_pool_size = result.attack_pool.size();
        log.attack_pool_update = duplicate_attack ? "duplicate_ignored" : "append_cumulative";
        log.propagation_model = to_string(options.propagation_model);
        log.defense_seeds = defense.seeds;
        log.attack_seeds = attack.seeds;
        log.defense_status = DefenseOracleStatus::SearchCompleted;
        log.defense_candidate_source = DefenseCandidateSource::Surrogate;
        log.defense_status_message = "linear_reference_weighted_rr";
        log.defense_candidate_count = graph.num_nodes();
        log.empirical_evaluator = "not_applicable_linear_approx";
        log.attack_certificate_status = "not_applicable_linear_approx";
        log.defense_certificate_status = "not_applicable_linear_approx";
        log.iteration_runtime_sec = elapsed_seconds(iteration_start, Clock::now());
        result.iterations.push_back(log);
        result.defense_seeds = defense.seeds;
        result.attack_seeds = attack.seeds;

        if (progress_os != nullptr) {
            algorithm_timer.pause_for_output();
            *progress_os << "CRD-Linear iteration " << iteration << '\n'
                         << "  attack_pool_size: " << result.attack_pool.size() << '\n'
                         << "  defense_seeds: " << format_seed_set_preview(defense.seeds) << '\n'
                         << "  attack_seeds: " << format_seed_set_preview(attack.seeds) << '\n'
                         << "  historical_worst_linear_residual: "
                         << format_decimal(defense.worst_residual) << '\n'
                         << "  attack_linear_gain: " << format_decimal(attack.gain) << '\n'
                         << "  defense_rr_samples: " << defense.sample_count << '\n'
                         << "  attack_rr_samples: " << attack.sample_count << '\n'
                         << "  duplicate_attack: " << (duplicate_attack ? "true" : "false") << '\n'
                         << "  iteration_runtime_sec: "
                         << format_decimal(log.iteration_runtime_sec) << '\n';
            algorithm_timer.resume_after_output();
        }
    }

    if (!result.iterations.empty()) {
        result.final_selection_policy = "last_iteration_linear_approx";
        result.final_selected_iteration = result.iterations.back().iteration;
    }
    result.algorithm_runtime_sec = algorithm_timer.elapsed_seconds();
    return result;
}

}  // namespace rcd
