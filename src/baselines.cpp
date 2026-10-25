#include "rcd/baselines.hpp"

#include "rcd/certificate.hpp"
#include "rcd/crd.hpp"
#include "rcd/progress.hpp"
#include "rcd/rr_sets.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>

namespace rcd {
namespace {

std::vector<NodeId> select_degree_defense(
    const DirectedGraph& graph,
    std::size_t budget) {
    std::vector<std::pair<std::size_t, NodeId>> ranked;
    ranked.reserve(graph.num_nodes());
    for (NodeId node = 0; node < graph.num_nodes(); ++node) {
        ranked.emplace_back(graph.out_adj[node].size(), node);
    }

    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first > rhs.first;
        }
        return lhs.second < rhs.second;
    });

    const auto take = std::min<std::size_t>(budget, ranked.size());
    std::vector<NodeId> seeds;
    seeds.reserve(take);
    for (std::size_t i = 0; i < take; ++i) {
        seeds.push_back(ranked[i].second);
    }
    return seeds;
}

std::vector<NodeId> all_candidate_nodes(const DirectedGraph& graph) {
    std::vector<NodeId> candidates;
    candidates.reserve(graph.num_nodes());
    for (NodeId node = 0; node < graph.num_nodes(); ++node) {
        candidates.push_back(node);
    }
    return candidates;
}

std::size_t saturating_binomial(std::size_t n, std::size_t k) {
    if (k > n) {
        return 0;
    }
    k = std::min(k, n - k);
    long double value = 1.0L;
    const long double maximum = static_cast<long double>(std::numeric_limits<std::size_t>::max());
    for (std::size_t i = 1; i <= k; ++i) {
        value *= static_cast<long double>(n - k + i);
        value /= static_cast<long double>(i);
        if (!std::isfinite(value) || value >= maximum) {
            return std::numeric_limits<std::size_t>::max();
        }
    }
    return static_cast<std::size_t>(std::llround(value));
}

std::size_t saturating_product(std::size_t lhs, std::size_t rhs) {
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    if (lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        return std::numeric_limits<std::size_t>::max();
    }
    return lhs * rhs;
}

void enumerate_seed_sets(
    std::size_t num_nodes,
    std::size_t budget,
    const std::function<void(const std::vector<NodeId>&)>& callback) {
    std::vector<NodeId> current;
    current.reserve(budget);
    const std::function<void(NodeId)> visit = [&](NodeId next_node) {
        if (current.size() == budget) {
            callback(current);
            return;
        }
        const auto remaining = budget - current.size();
        if (static_cast<std::size_t>(next_node) + remaining > num_nodes) {
            return;
        }
        for (NodeId node = next_node; node < num_nodes; ++node) {
            current.push_back(node);
            visit(node + 1);
            current.pop_back();
        }
    };
    if (budget == 0) {
        callback(current);
    } else if (budget <= num_nodes) {
        visit(0);
    }
}

std::vector<NodeId> sorted_seed_key(std::vector<NodeId> seeds) {
    std::sort(seeds.begin(), seeds.end());
    return seeds;
}

bool attack_pool_contains(
    const std::vector<std::vector<NodeId>>& attack_pool,
    const std::vector<NodeId>& attack) {
    const auto key = sorted_seed_key(attack);
    return std::any_of(attack_pool.begin(), attack_pool.end(), [&](const auto& existing) {
        return sorted_seed_key(existing) == key;
    });
}

struct FJCLinearRootWeights {
    std::vector<Weight> coverage_weights;
    double constant = 0.0;
    double raw_min = 0.0;
    double raw_max = 0.0;
    double raw_sum = 0.0;
    std::size_t positive_count = 0;
};

// Linearize the complete FJ conflict objective around the realization induced
// by the fixed opponent. The remaining optimized player's reach vector r has
// objective constant + ell^T r + r^T C r. FJC-Linear-Iter intentionally drops
// the quadratic term and uses ordinary weighted RR coverage to optimize the
// oriented linear gain: ell for attack and -ell for defense.
[[maybe_unused]] FJCLinearRootWeights build_fjc_linear_root_weights(
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<std::uint8_t>& fixed_opponent_reached,
    const CascadeParams& params,
    OptimizedPlayer optimized_player) {
    if (fixed_opponent_reached.size() != static_cast<std::size_t>(innate.size())) {
        throw std::invalid_argument(
            "FJC linear root weights require an opponent reach mask matching innate opinions");
    }

    const ConditionalTerms terms = build_conditional_terms(
        conflict, innate, fixed_opponent_reached, params, optimized_player);
    const double direction =
        optimized_player == OptimizedPlayer::Attacker ? 1.0 : -1.0;

    FJCLinearRootWeights result;
    result.coverage_weights.resize(static_cast<std::size_t>(terms.ell.size()), 0.0);
    result.constant = terms.constant;
    if (terms.ell.size() == 0) {
        return result;
    }

    result.raw_min = std::numeric_limits<double>::infinity();
    result.raw_max = -std::numeric_limits<double>::infinity();
    for (Eigen::Index index = 0; index < terms.ell.size(); ++index) {
        const double raw_weight = direction * terms.ell[index];
        if (!std::isfinite(raw_weight)) {
            throw std::runtime_error("FJC linear root weight is not finite");
        }
        result.raw_min = std::min(result.raw_min, raw_weight);
        result.raw_max = std::max(result.raw_max, raw_weight);
        result.raw_sum += raw_weight;
        if (raw_weight > 0.0) {
            result.coverage_weights[static_cast<std::size_t>(index)] = raw_weight;
            ++result.positive_count;
        }
    }
    return result;
}

std::vector<std::uint8_t> make_fjc_seed_context_mask(
    std::size_t num_nodes,
    const std::vector<NodeId>& seeds) {
    std::vector<std::uint8_t> reached(num_nodes, 0);
    for (const NodeId seed : seeds) {
        if (static_cast<std::size_t>(seed) >= num_nodes) {
            throw std::out_of_range("FJC linear context seed is outside graph");
        }
        reached[seed] = 1;
    }
    return reached;
}

std::vector<std::uint8_t> make_fjc_candidate_mask(
    std::size_t num_nodes,
    const std::vector<NodeId>& candidates) {
    std::vector<std::uint8_t> candidate_mask(num_nodes, 0);
    for (const NodeId candidate : candidates) {
        if (static_cast<std::size_t>(candidate) >= num_nodes) {
            throw std::out_of_range("FJC linear candidate is outside graph");
        }
        candidate_mask[candidate] = 1;
    }
    return candidate_mask;
}

struct FJCLinearRRAttackSelection {
    std::vector<NodeId> attack_seeds;
    double selection_value = 0.0;
    std::size_t selection_rr_samples = 0;
    double root_weight_min = 0.0;
    double root_weight_max = 0.0;
    double root_weight_sum = 0.0;
    std::size_t positive_root_weights = 0;
};

// This is the scalable Tu-style selection phase: ordinary weighted RR samples
// optimize only the attack-side linear FJ term. The opponent context deliberately
// contains its selected seeds, not a nested forward-MC simulation, so selection
// needs no per-sample conflict solve.
[[maybe_unused]] FJCLinearRRAttackSelection greedy_fjc_linear_rr_attack(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const std::vector<NodeId>& defense_seeds,
    const std::vector<NodeId>& candidates,
    const BaselineOptions& options,
    std::uint64_t seed,
    const ProgressCallback& progress_callback = {}) {
    if (options.fjc_linear_rr_samples == 0) {
        throw std::invalid_argument("FJC-Linear-Iter RR sample count must be positive");
    }
    if (graph.num_nodes() == 0) {
        throw std::invalid_argument("FJC-Linear-Iter requires a nonempty graph");
    }

    const auto fixed_defense_context =
        make_fjc_seed_context_mask(graph.num_nodes(), defense_seeds);
    const auto root_weights = build_fjc_linear_root_weights(
        conflict,
        innate,
        fixed_defense_context,
        params,
        OptimizedPlayer::Attacker);
    const auto candidate_mask = make_fjc_candidate_mask(graph.num_nodes(), candidates);
    const auto staged_progress = [&progress_callback](const std::string& stage, double percent) {
        if (progress_callback) {
            progress_callback("FJC-Linear-Iter attack: " + stage, percent);
        }
    };

    const auto samples = generate_weighted_rr_samples_from_node_weights(
        options.propagation_model,
        graph,
        options.fjc_linear_rr_samples,
        seed,
        root_weights.coverage_weights,
        staged_progress);
    auto attack_seeds = weighted_coverage_greedy(
        graph.num_nodes(),
        samples,
        options.attacker_budget,
        candidate_mask,
        staged_progress);
    std::sort(attack_seeds.begin(), attack_seeds.end());

    FJCLinearRRAttackSelection selection;
    selection.attack_seeds = std::move(attack_seeds);
    selection.selection_value = weighted_coverage_value(samples, selection.attack_seeds);
    selection.selection_rr_samples = samples.size();
    selection.root_weight_min = root_weights.raw_min;
    selection.root_weight_max = root_weights.raw_max;
    selection.root_weight_sum = root_weights.raw_sum;
    selection.positive_root_weights = root_weights.positive_count;
    return selection;
}

struct FJCLinearDefensePool {
    double constant = 0.0;
    std::vector<WeightedRRSample> samples;
    std::vector<std::vector<RRIndex>> covered_by_node;
    std::vector<std::uint8_t> covered;
    double covered_weight_sum = 0.0;
};

FJCLinearDefensePool build_fjc_linear_defense_pool(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const std::vector<NodeId>& attack_seeds,
    const BaselineOptions& options,
    std::uint64_t seed,
    const ProgressCallback& progress_callback) {
    const auto fixed_attack_context =
        make_fjc_seed_context_mask(graph.num_nodes(), attack_seeds);
    const auto root_weights = build_fjc_linear_root_weights(
        conflict,
        innate,
        fixed_attack_context,
        params,
        OptimizedPlayer::Defender);
    auto samples = generate_weighted_rr_samples_from_node_weights(
        options.propagation_model,
        graph,
        options.fjc_linear_rr_samples,
        seed,
        root_weights.coverage_weights,
        progress_callback);

    FJCLinearDefensePool pool;
    pool.constant = root_weights.constant;
    pool.covered_by_node.assign(graph.num_nodes(), {});
    pool.covered.assign(samples.size(), 0);
    for (RRIndex sample_id = 0; sample_id < samples.size(); ++sample_id) {
        for (const NodeId node : samples[sample_id].rr_set) {
            if (static_cast<std::size_t>(node) >= graph.num_nodes()) {
                throw std::out_of_range("FJC linear RR set contains node outside graph");
            }
            pool.covered_by_node[node].push_back(sample_id);
        }
    }
    pool.samples = std::move(samples);
    return pool;
}

double fjc_linear_defense_pool_candidate_gain(
    const FJCLinearDefensePool& pool,
    NodeId candidate) {
    if (static_cast<std::size_t>(candidate) >= pool.covered_by_node.size()) {
        throw std::out_of_range("FJC linear defense candidate is outside graph");
    }
    double gain = 0.0;
    for (const RRIndex sample_id : pool.covered_by_node[candidate]) {
        if (sample_id >= pool.samples.size() || sample_id >= pool.covered.size()) {
            throw std::out_of_range("FJC linear defense inverted index contains invalid sample id");
        }
        if (pool.covered[sample_id] == 0) {
            gain += pool.samples[sample_id].weight;
        }
    }
    return gain;
}

double fjc_linear_defense_pool_residual_value(
    const FJCLinearDefensePool& pool,
    double additional_covered_weight = 0.0) {
    if (pool.samples.empty()) {
        throw std::invalid_argument("FJC linear defense pool must contain RR samples");
    }
    return pool.constant -
        (pool.covered_weight_sum + additional_covered_weight) /
            static_cast<double>(pool.samples.size());
}

void apply_fjc_linear_defense_pool_selection(
    FJCLinearDefensePool& pool,
    NodeId selected_node) {
    if (static_cast<std::size_t>(selected_node) >= pool.covered_by_node.size()) {
        throw std::out_of_range("FJC linear defense selection is outside graph");
    }
    for (const RRIndex sample_id : pool.covered_by_node[selected_node]) {
        if (sample_id >= pool.samples.size() || sample_id >= pool.covered.size()) {
            throw std::out_of_range("FJC linear defense inverted index contains invalid sample id");
        }
        if (pool.covered[sample_id] == 0) {
            pool.covered[sample_id] = 1;
            pool.covered_weight_sum += pool.samples[sample_id].weight;
        }
    }
}

struct FJCLinearRRDefenseSelection {
    std::vector<NodeId> defense_seeds;
    double selection_value = 0.0;
    std::size_t selection_rr_samples = 0;
    std::size_t attack_pool_size = 0;
};

// The minimax objective couples historical attacks, so a single-pool lazy queue
// is not valid here. We still maintain each pool's inverted index and covered
// state incrementally; candidate scans therefore never traverse an RR set that
// is not incident to that candidate and never invoke forward Monte Carlo.
[[maybe_unused]] FJCLinearRRDefenseSelection greedy_fjc_linear_rr_defense(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const std::vector<std::vector<NodeId>>& attack_pool,
    const std::vector<NodeId>& candidates,
    const BaselineOptions& options,
    std::uint64_t seed,
    const ProgressCallback& progress_callback = {}) {
    if (attack_pool.empty()) {
        throw std::invalid_argument("FJC-Linear-Iter defense requires a nonempty attack pool");
    }
    if (options.fjc_linear_rr_samples == 0) {
        throw std::invalid_argument("FJC-Linear-Iter RR sample count must be positive");
    }

    std::vector<FJCLinearDefensePool> pools;
    pools.reserve(attack_pool.size());
    for (std::size_t attack_index = 0; attack_index < attack_pool.size(); ++attack_index) {
        const auto staged_progress = [&progress_callback, attack_index, &attack_pool](
                                         const std::string& stage,
                                         double percent) {
            if (progress_callback) {
                progress_callback(
                    "FJC-Linear-Iter defense pool " +
                        std::to_string(attack_index + 1) + "/" +
                        std::to_string(attack_pool.size()) + ": " + stage,
                    percent);
            }
        };
        pools.push_back(build_fjc_linear_defense_pool(
            graph,
            conflict,
            innate,
            params,
            attack_pool[attack_index],
            options,
            seed ^ (0x9e3779b97f4a7c15ULL * (attack_index + 1)),
            staged_progress));
    }

    const auto candidate_mask = make_fjc_candidate_mask(graph.num_nodes(), candidates);
    std::vector<std::uint8_t> selected(graph.num_nodes(), 0);
    std::vector<NodeId> defense_seeds;
    defense_seeds.reserve(std::min(options.defender_budget, candidates.size()));
    const std::size_t progress_interval =
        std::max<std::size_t>(1, options.defender_budget / 10);

    auto current_worst_value = [&pools]() {
        double worst = -std::numeric_limits<double>::infinity();
        for (const auto& pool : pools) {
            worst = std::max(worst, fjc_linear_defense_pool_residual_value(pool));
        }
        return worst;
    };
    double worst_value = current_worst_value();

    for (std::size_t round = 0; round < options.defender_budget; ++round) {
        NodeId best_node = 0;
        double best_worst_value = std::numeric_limits<double>::infinity();
        bool found_candidate = false;

        for (std::size_t node_index = 0; node_index < graph.num_nodes(); ++node_index) {
            const NodeId candidate = static_cast<NodeId>(node_index);
            if (candidate_mask[candidate] == 0 || selected[candidate] != 0) {
                continue;
            }

            double candidate_worst_value = -std::numeric_limits<double>::infinity();
            for (const auto& pool : pools) {
                const double gain = fjc_linear_defense_pool_candidate_gain(pool, candidate);
                candidate_worst_value = std::max(
                    candidate_worst_value,
                    fjc_linear_defense_pool_residual_value(pool, gain));
            }
            if (!found_candidate || candidate_worst_value < best_worst_value ||
                (candidate_worst_value == best_worst_value && candidate < best_node)) {
                best_node = candidate;
                best_worst_value = candidate_worst_value;
                found_candidate = true;
            }
        }

        if (!found_candidate || best_worst_value >= worst_value - 1e-15) {
            break;
        }
        selected[best_node] = 1;
        defense_seeds.push_back(best_node);
        for (auto& pool : pools) {
            apply_fjc_linear_defense_pool_selection(pool, best_node);
        }
        worst_value = current_worst_value();
        if (progress_callback &&
            (defense_seeds.size() % progress_interval == 0 ||
             defense_seeds.size() == options.defender_budget)) {
            progress_callback(
                "FJC-Linear-Iter defense: Greedy minimax selection",
                100.0 * static_cast<double>(defense_seeds.size()) /
                    static_cast<double>(std::max<std::size_t>(options.defender_budget, 1)));
        }
    }

    std::sort(defense_seeds.begin(), defense_seeds.end());
    FJCLinearRRDefenseSelection selection;
    selection.defense_seeds = std::move(defense_seeds);
    selection.selection_value = worst_value;
    selection.attack_pool_size = attack_pool.size();
    for (const auto& pool : pools) {
        selection.selection_rr_samples += pool.samples.size();
    }
    return selection;
}

struct FJCPairRRSample {
    NodeId first_root = 0;
    NodeId second_root = 0;
    std::vector<NodeId> first_rr_set;
    std::vector<NodeId> second_rr_set;
    double single_weight = 0.0;
    double pair_weight = 0.0;
};

struct FJCPairRRPool {
    std::vector<FJCPairRRSample> samples;
    std::vector<std::vector<std::size_t>> first_covered_by_node;
    std::vector<std::vector<std::size_t>> second_covered_by_node;
};

void normalize_fjc_rr_set(std::vector<NodeId>& rr_set) {
    std::sort(rr_set.begin(), rr_set.end());
    rr_set.erase(std::unique(rr_set.begin(), rr_set.end()), rr_set.end());
}

void add_fjc_pair_rr_membership(
    std::vector<std::vector<std::size_t>>& covered_by_node,
    const std::vector<NodeId>& rr_set,
    std::size_t sample_index) {
    for (const auto node : rr_set) {
        if (static_cast<std::size_t>(node) >= covered_by_node.size()) {
            throw std::out_of_range("Pair-RR set contains node outside graph");
        }
        covered_by_node[node].push_back(sample_index);
    }
}

std::size_t validate_fjc_pair_rr_pool(
    const FJCPairRRPool& pool,
    const Eigen::VectorXd& innate) {
    if (pool.samples.empty()) {
        throw std::invalid_argument("FJC pair-RR pool must be nonempty");
    }
    if (pool.first_covered_by_node.size() != pool.second_covered_by_node.size()) {
        throw std::invalid_argument("FJC pair-RR pool side indices have different node counts");
    }
    const auto num_nodes = pool.first_covered_by_node.size();
    if (static_cast<std::size_t>(innate.size()) != num_nodes) {
        throw std::invalid_argument("FJC pair-RR pool node count does not match opinion vector");
    }
    return num_nodes;
}

std::vector<std::uint8_t> fjc_pair_rr_side_coverage(
    const std::vector<std::vector<std::size_t>>& covered_by_node,
    const std::vector<NodeId>& seeds,
    std::size_t sample_count) {
    std::vector<std::uint8_t> covered(sample_count, 0);
    for (const auto seed : seeds) {
        if (static_cast<std::size_t>(seed) >= covered_by_node.size()) {
            throw std::out_of_range("Seed node is outside FJC pair-RR pool node universe");
        }
        for (const auto sample_index : covered_by_node[seed]) {
            if (sample_index >= sample_count) {
                throw std::out_of_range("FJC pair-RR inverted index contains invalid sample id");
            }
            covered[sample_index] = 1;
        }
    }
    return covered;
}

double fjc_rho_value(
    const CascadeParams& params,
    NodeId node,
    std::size_t expected_size) {
    if (params.rho_u_by_node.size() == 0) {
        return params.default_rho_u;
    }
    if (static_cast<std::size_t>(params.rho_u_by_node.size()) != expected_size) {
        throw std::invalid_argument("rho_by_node size does not match opinion vector size");
    }
    return params.rho_u_by_node[static_cast<Eigen::Index>(node)];
}

double fjc_pair_rr_perturbed_opinion_at(
    const Eigen::VectorXd& innate,
    NodeId node,
    bool attack_reached,
    bool defense_reached,
    const CascadeParams& params) {
    if (static_cast<Eigen::Index>(node) >= innate.size()) {
        throw std::out_of_range("FJC pair-RR root node is outside opinion vector");
    }
    const double s = innate[static_cast<Eigen::Index>(node)];
    const double r_a = attack_reached ? 1.0 : 0.0;
    const double r_b = defense_reached ? 1.0 : 0.0;
    const double rho = fjc_rho_value(params, node, static_cast<std::size_t>(innate.size()));
    return clip_unit_interval(
        s +
        r_a * (1.0 - rho * r_b) * params.alpha_A * (s - 0.5) +
        r_b * params.alpha_B * (0.5 - s));
}

[[maybe_unused]] FJCPairRRPool build_fjc_pair_rr_pool(
    PropagationModel model,
    const DirectedGraph& graph,
    const DenseConflict& conflict,
    std::size_t sample_count,
    std::uint64_t seed) {
    if (graph.num_nodes() == 0) {
        throw std::invalid_argument("FJC pair-RR pool requires a nonempty graph");
    }
    if (sample_count == 0) {
        throw std::invalid_argument("FJC pair-RR sample count must be positive");
    }
    if (conflict.num_nodes() != graph.num_nodes()) {
        throw std::invalid_argument("FJC pair-RR conflict matrix size does not match graph");
    }

    FJCPairRRPool pool;
    pool.samples.reserve(sample_count);
    pool.first_covered_by_node.assign(graph.num_nodes(), {});
    pool.second_covered_by_node.assign(graph.num_nodes(), {});

    const double n_squared =
        static_cast<double>(graph.num_nodes()) * static_cast<double>(graph.num_nodes());
    const auto& c = conflict.conflict_matrix();
    std::mt19937_64 root_rng(seed ^ 0x6a09e667f3bcc909ULL);
    std::uniform_int_distribution<std::size_t> root_dist(0, graph.num_nodes() - 1);
    LazyLiveEdgeOracle live_oracle(graph.num_edges(), seed ^ 0xbb67ae8584caa73bULL);
    LazyTriggeringEdgeOracle triggering_oracle(graph.num_nodes(), seed ^ 0x3c6ef372fe94f82bULL);
    TraversalWorkspace first_rr_workspace(graph.num_nodes());
    TraversalWorkspace second_rr_workspace(graph.num_nodes());

    for (std::size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        FJCPairRRSample sample;
        sample.first_root = static_cast<NodeId>(root_dist(root_rng));
        sample.second_root = static_cast<NodeId>(root_dist(root_rng));
        sample.first_rr_set = reverse_rr_set(
            model,
            graph,
            sample.first_root,
            live_oracle,
            triggering_oracle,
            first_rr_workspace);
        sample.second_rr_set = reverse_rr_set(
            model,
            graph,
            sample.second_root,
            live_oracle,
            triggering_oracle,
            second_rr_workspace);
        normalize_fjc_rr_set(sample.first_rr_set);
        normalize_fjc_rr_set(sample.second_rr_set);

        sample.single_weight = 0.0;
        sample.pair_weight =
            n_squared *
            c(static_cast<Eigen::Index>(sample.first_root),
              static_cast<Eigen::Index>(sample.second_root));

        add_fjc_pair_rr_membership(
            pool.first_covered_by_node,
            sample.first_rr_set,
            sample_index);
        add_fjc_pair_rr_membership(
            pool.second_covered_by_node,
            sample.second_rr_set,
            sample_index);
        pool.samples.push_back(std::move(sample));

        live_oracle.reset();
        triggering_oracle.reset();
    }

    return pool;
}

[[maybe_unused]] double evaluate_fjc_pair_rr_value(
    const FJCPairRRPool& pool,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds,
    const Eigen::VectorXd& innate,
    const CascadeParams& params) {
    const auto num_nodes = validate_fjc_pair_rr_pool(pool, innate);
    const auto first_attack_covered = fjc_pair_rr_side_coverage(
        pool.first_covered_by_node,
        attack_seeds,
        pool.samples.size());
    const auto second_attack_covered = fjc_pair_rr_side_coverage(
        pool.second_covered_by_node,
        attack_seeds,
        pool.samples.size());
    const auto first_defense_covered = fjc_pair_rr_side_coverage(
        pool.first_covered_by_node,
        defense_seeds,
        pool.samples.size());
    const auto second_defense_covered = fjc_pair_rr_side_coverage(
        pool.second_covered_by_node,
        defense_seeds,
        pool.samples.size());

    double total = 0.0;
    for (std::size_t sample_index = 0; sample_index < pool.samples.size(); ++sample_index) {
        const auto& sample = pool.samples[sample_index];
        if (static_cast<std::size_t>(sample.first_root) >= num_nodes ||
            static_cast<std::size_t>(sample.second_root) >= num_nodes) {
            throw std::out_of_range("FJC pair-RR sample root is outside node universe");
        }
        const double first_opinion = fjc_pair_rr_perturbed_opinion_at(
            innate,
            sample.first_root,
            first_attack_covered[sample_index] != 0,
            first_defense_covered[sample_index] != 0,
            params);
        const double second_opinion = fjc_pair_rr_perturbed_opinion_at(
            innate,
            sample.second_root,
            second_attack_covered[sample_index] != 0,
            second_defense_covered[sample_index] != 0,
            params);
        total += sample.single_weight * first_opinion +
                 sample.pair_weight * first_opinion * second_opinion;
    }

    return total / static_cast<double>(pool.samples.size());
}

double fjc_pair_rr_sample_value(
    const FJCPairRRSample& sample,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    bool first_attack_covered,
    bool second_attack_covered,
    bool first_defense_covered,
    bool second_defense_covered) {
    const double first_opinion = fjc_pair_rr_perturbed_opinion_at(
        innate,
        sample.first_root,
        first_attack_covered,
        first_defense_covered,
        params);
    const double second_opinion = fjc_pair_rr_perturbed_opinion_at(
        innate,
        sample.second_root,
        second_attack_covered,
        second_defense_covered,
        params);
    return sample.single_weight * first_opinion +
           sample.pair_weight * first_opinion * second_opinion;
}

void collect_fjc_candidate_side_updates(
    const std::vector<std::vector<std::size_t>>& covered_by_node,
    NodeId candidate,
    const std::vector<std::uint8_t>& current_side_covered,
    std::vector<std::uint8_t>& side_added,
    std::vector<std::size_t>& affected_samples,
    std::vector<std::uint8_t>& affected_mask) {
    if (static_cast<std::size_t>(candidate) >= covered_by_node.size()) {
        throw std::out_of_range("Candidate node is outside FJC pair-RR pool node universe");
    }
    for (const auto sample_index : covered_by_node[candidate]) {
        if (sample_index >= current_side_covered.size() ||
            sample_index >= side_added.size() ||
            sample_index >= affected_mask.size()) {
            throw std::out_of_range("FJC pair-RR inverted index contains invalid sample id");
        }
        if (current_side_covered[sample_index] != 0) {
            continue;
        }
        side_added[sample_index] = 1;
        if (affected_mask[sample_index] == 0) {
            affected_mask[sample_index] = 1;
            affected_samples.push_back(sample_index);
        }
    }
}

void clear_fjc_candidate_updates(
    const std::vector<std::size_t>& affected_samples,
    std::vector<std::uint8_t>& first_side_added,
    std::vector<std::uint8_t>& second_side_added,
    std::vector<std::uint8_t>& affected_mask) {
    for (const auto sample_index : affected_samples) {
        first_side_added[sample_index] = 0;
        second_side_added[sample_index] = 0;
        affected_mask[sample_index] = 0;
    }
}

double initialize_fjc_pair_rr_values(
    const FJCPairRRPool& pool,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const std::vector<std::uint8_t>& first_attack_covered,
    const std::vector<std::uint8_t>& second_attack_covered,
    const std::vector<std::uint8_t>& first_defense_covered,
    const std::vector<std::uint8_t>& second_defense_covered,
    std::vector<double>& sample_values) {
    sample_values.assign(pool.samples.size(), 0.0);
    double total = 0.0;
    for (std::size_t sample_index = 0; sample_index < pool.samples.size(); ++sample_index) {
        const double value = fjc_pair_rr_sample_value(
            pool.samples[sample_index],
            innate,
            params,
            first_attack_covered[sample_index] != 0,
            second_attack_covered[sample_index] != 0,
            first_defense_covered[sample_index] != 0,
            second_defense_covered[sample_index] != 0);
        sample_values[sample_index] = value;
        total += value;
    }
    return total;
}

struct FJCPairRRAttackSelection {
    std::vector<NodeId> attack_seeds;
    double selection_value = 0.0;
    std::size_t selection_rr_samples = 0;
};

[[maybe_unused]] FJCPairRRAttackSelection greedy_fjc_pair_rr_attack(
    const DirectedGraph& graph,
    const DenseConflict& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const std::vector<NodeId>& defense_seeds,
    const std::vector<NodeId>& candidates,
    const BaselineOptions& options,
    std::uint64_t seed_offset) {
    const auto pool = build_fjc_pair_rr_pool(
        options.propagation_model,
        graph,
        conflict,
        options.fjc_pair_rr_samples,
        options.seed ^ seed_offset);

    FJCPairRRAttackSelection selection;
    selection.selection_rr_samples = pool.samples.size();
    selection.attack_seeds.reserve(std::min<std::size_t>(options.attacker_budget, candidates.size()));

    const auto first_defense_covered = fjc_pair_rr_side_coverage(
        pool.first_covered_by_node,
        defense_seeds,
        pool.samples.size());
    const auto second_defense_covered = fjc_pair_rr_side_coverage(
        pool.second_covered_by_node,
        defense_seeds,
        pool.samples.size());
    std::vector<std::uint8_t> first_attack_covered(pool.samples.size(), 0);
    std::vector<std::uint8_t> second_attack_covered(pool.samples.size(), 0);
    std::vector<double> sample_values;
    double total_value = initialize_fjc_pair_rr_values(
        pool,
        innate,
        params,
        first_attack_covered,
        second_attack_covered,
        first_defense_covered,
        second_defense_covered,
        sample_values);
    selection.selection_value = total_value / static_cast<double>(pool.samples.size());

    std::vector<std::uint8_t> selected(graph.num_nodes(), 0);
    std::vector<std::uint8_t> first_added(pool.samples.size(), 0);
    std::vector<std::uint8_t> second_added(pool.samples.size(), 0);
    std::vector<std::uint8_t> affected_mask(pool.samples.size(), 0);
    std::vector<std::size_t> affected_samples;

    for (std::size_t slot = 0; slot < options.attacker_budget; ++slot) {
        NodeId best_node = graph.num_nodes();
        double best_value = -std::numeric_limits<double>::infinity();
        bool found_candidate = false;

        for (const auto candidate : candidates) {
            if (static_cast<std::size_t>(candidate) >= graph.num_nodes()) {
                throw std::out_of_range("Candidate node is outside graph");
            }
            if (selected[candidate] != 0) {
                continue;
            }

            affected_samples.clear();
            collect_fjc_candidate_side_updates(
                pool.first_covered_by_node,
                candidate,
                first_attack_covered,
                first_added,
                affected_samples,
                affected_mask);
            collect_fjc_candidate_side_updates(
                pool.second_covered_by_node,
                candidate,
                second_attack_covered,
                second_added,
                affected_samples,
                affected_mask);

            double gain = 0.0;
            for (const auto sample_index : affected_samples) {
                const double new_value = fjc_pair_rr_sample_value(
                    pool.samples[sample_index],
                    innate,
                    params,
                    first_attack_covered[sample_index] != 0 || first_added[sample_index] != 0,
                    second_attack_covered[sample_index] != 0 || second_added[sample_index] != 0,
                    first_defense_covered[sample_index] != 0,
                    second_defense_covered[sample_index] != 0);
                gain += new_value - sample_values[sample_index];
            }
            const double value =
                (total_value + gain) / static_cast<double>(pool.samples.size());
            clear_fjc_candidate_updates(
                affected_samples,
                first_added,
                second_added,
                affected_mask);
            if (!found_candidate ||
                value > best_value ||
                (value == best_value && candidate < best_node)) {
                found_candidate = true;
                best_node = candidate;
                best_value = value;
            }
        }

        if (!found_candidate) {
            break;
        }

        affected_samples.clear();
        collect_fjc_candidate_side_updates(
            pool.first_covered_by_node,
            best_node,
            first_attack_covered,
            first_added,
            affected_samples,
            affected_mask);
        collect_fjc_candidate_side_updates(
            pool.second_covered_by_node,
            best_node,
            second_attack_covered,
            second_added,
            affected_samples,
            affected_mask);
        for (const auto sample_index : affected_samples) {
            const double new_value = fjc_pair_rr_sample_value(
                pool.samples[sample_index],
                innate,
                params,
                first_attack_covered[sample_index] != 0 || first_added[sample_index] != 0,
                second_attack_covered[sample_index] != 0 || second_added[sample_index] != 0,
                first_defense_covered[sample_index] != 0,
                second_defense_covered[sample_index] != 0);
            total_value += new_value - sample_values[sample_index];
            sample_values[sample_index] = new_value;
            if (first_added[sample_index] != 0) {
                first_attack_covered[sample_index] = 1;
            }
            if (second_added[sample_index] != 0) {
                second_attack_covered[sample_index] = 1;
            }
        }
        clear_fjc_candidate_updates(
            affected_samples,
            first_added,
            second_added,
            affected_mask);

        selected[best_node] = 1;
        selection.attack_seeds.push_back(best_node);
        selection.selection_value = total_value / static_cast<double>(pool.samples.size());
    }

    return selection;
}

[[maybe_unused]] double worst_fjc_pair_rr_value_over_attack_pool(
    const FJCPairRRPool& pool,
    const std::vector<std::vector<NodeId>>& attack_pool,
    const std::vector<NodeId>& defense_seeds,
    const Eigen::VectorXd& innate,
    const CascadeParams& params) {
    if (attack_pool.empty()) {
        return 0.0;
    }

    double worst = -std::numeric_limits<double>::infinity();
    for (const auto& attack : attack_pool) {
        worst = std::max(
            worst,
            evaluate_fjc_pair_rr_value(
                pool,
                attack,
                defense_seeds,
                innate,
                params));
    }
    return worst;
}

struct FJCPairRRDefenseSelection {
    std::vector<NodeId> defense_seeds;
    double selection_value = 0.0;
    std::size_t selection_rr_samples = 0;
};

struct FJCPairRRAttackCoverageState {
    std::vector<std::uint8_t> first_attack_covered;
    std::vector<std::uint8_t> second_attack_covered;
    std::vector<double> sample_values;
    double total_value = 0.0;
};

[[maybe_unused]] FJCPairRRDefenseSelection greedy_fjc_pair_rr_defense(
    const DirectedGraph& graph,
    const DenseConflict& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const std::vector<std::vector<NodeId>>& attack_pool,
    const std::vector<NodeId>& candidates,
    const BaselineOptions& options,
    std::uint64_t seed_offset) {
    const auto pool = build_fjc_pair_rr_pool(
        options.propagation_model,
        graph,
        conflict,
        options.fjc_pair_rr_samples,
        options.seed ^ seed_offset);

    FJCPairRRDefenseSelection selection;
    selection.selection_rr_samples = pool.samples.size();
    selection.defense_seeds.reserve(
        std::min<std::size_t>(options.defender_budget, candidates.size()));

    std::vector<std::uint8_t> first_defense_covered(pool.samples.size(), 0);
    std::vector<std::uint8_t> second_defense_covered(pool.samples.size(), 0);
    std::vector<FJCPairRRAttackCoverageState> attack_states;
    attack_states.reserve(attack_pool.size());
    double current_worst_total = 0.0;
    for (const auto& attack : attack_pool) {
        FJCPairRRAttackCoverageState state;
        state.first_attack_covered = fjc_pair_rr_side_coverage(
            pool.first_covered_by_node,
            attack,
            pool.samples.size());
        state.second_attack_covered = fjc_pair_rr_side_coverage(
            pool.second_covered_by_node,
            attack,
            pool.samples.size());
        state.total_value = initialize_fjc_pair_rr_values(
            pool,
            innate,
            params,
            state.first_attack_covered,
            state.second_attack_covered,
            first_defense_covered,
            second_defense_covered,
            state.sample_values);
        current_worst_total = std::max(current_worst_total, state.total_value);
        attack_states.push_back(std::move(state));
    }
    selection.selection_value =
        current_worst_total / static_cast<double>(pool.samples.size());

    std::vector<std::uint8_t> selected(graph.num_nodes(), 0);
    std::vector<std::uint8_t> first_added(pool.samples.size(), 0);
    std::vector<std::uint8_t> second_added(pool.samples.size(), 0);
    std::vector<std::uint8_t> affected_mask(pool.samples.size(), 0);
    std::vector<std::size_t> affected_samples;

    for (std::size_t slot = 0; slot < options.defender_budget; ++slot) {
        NodeId best_node = graph.num_nodes();
        double best_value = std::numeric_limits<double>::infinity();
        bool found_candidate = false;

        for (const auto candidate : candidates) {
            if (static_cast<std::size_t>(candidate) >= graph.num_nodes()) {
                throw std::out_of_range("Candidate node is outside graph");
            }
            if (selected[candidate] != 0) {
                continue;
            }

            affected_samples.clear();
            collect_fjc_candidate_side_updates(
                pool.first_covered_by_node,
                candidate,
                first_defense_covered,
                first_added,
                affected_samples,
                affected_mask);
            collect_fjc_candidate_side_updates(
                pool.second_covered_by_node,
                candidate,
                second_defense_covered,
                second_added,
                affected_samples,
                affected_mask);

            double worst_total = 0.0;
            for (const auto& state : attack_states) {
                double delta = 0.0;
                for (const auto sample_index : affected_samples) {
                    const double new_value = fjc_pair_rr_sample_value(
                        pool.samples[sample_index],
                        innate,
                        params,
                        state.first_attack_covered[sample_index] != 0,
                        state.second_attack_covered[sample_index] != 0,
                        first_defense_covered[sample_index] != 0 || first_added[sample_index] != 0,
                        second_defense_covered[sample_index] != 0 || second_added[sample_index] != 0);
                    delta += new_value - state.sample_values[sample_index];
                }
                worst_total = std::max(worst_total, state.total_value + delta);
            }
            const double value = worst_total / static_cast<double>(pool.samples.size());
            clear_fjc_candidate_updates(
                affected_samples,
                first_added,
                second_added,
                affected_mask);
            if (!found_candidate ||
                value < best_value ||
                (value == best_value && candidate < best_node)) {
                found_candidate = true;
                best_node = candidate;
                best_value = value;
            }
        }

        if (!found_candidate) {
            break;
        }

        affected_samples.clear();
        collect_fjc_candidate_side_updates(
            pool.first_covered_by_node,
            best_node,
            first_defense_covered,
            first_added,
            affected_samples,
            affected_mask);
        collect_fjc_candidate_side_updates(
            pool.second_covered_by_node,
            best_node,
            second_defense_covered,
            second_added,
            affected_samples,
            affected_mask);
        current_worst_total = 0.0;
        for (auto& state : attack_states) {
            for (const auto sample_index : affected_samples) {
                const double new_value = fjc_pair_rr_sample_value(
                    pool.samples[sample_index],
                    innate,
                    params,
                    state.first_attack_covered[sample_index] != 0,
                    state.second_attack_covered[sample_index] != 0,
                    first_defense_covered[sample_index] != 0 || first_added[sample_index] != 0,
                    second_defense_covered[sample_index] != 0 || second_added[sample_index] != 0);
                state.total_value += new_value - state.sample_values[sample_index];
                state.sample_values[sample_index] = new_value;
            }
            current_worst_total = std::max(current_worst_total, state.total_value);
        }
        for (const auto sample_index : affected_samples) {
            if (first_added[sample_index] != 0) {
                first_defense_covered[sample_index] = 1;
            }
            if (second_added[sample_index] != 0) {
                second_defense_covered[sample_index] = 1;
            }
        }
        clear_fjc_candidate_updates(
            affected_samples,
            first_added,
            second_added,
            affected_mask);

        selected[best_node] = 1;
        selection.defense_seeds.push_back(best_node);
        selection.selection_value =
            current_worst_total / static_cast<double>(pool.samples.size());
    }

    return selection;
}

struct HistRRSamplePool {
    std::vector<std::vector<NodeId>> rr_sets;
    std::vector<std::vector<std::size_t>> covered_by_node;
    std::size_t early_stopped_samples = 0;
};

struct HistDefenseSelection {
    std::vector<NodeId> defense_seeds;
    double selection_value = 0.0;
    double validation_value = std::numeric_limits<double>::quiet_NaN();
    double validation_upper_bound = std::numeric_limits<double>::quiet_NaN();
    std::size_t sentinel_size = 0;
    std::size_t selection_rr_samples = 0;
    std::size_t validation_rr_samples = 0;
    std::size_t validation_early_stopped_samples = 0;
    std::string validation_status = "not_available";
    std::size_t adaptive_rounds_used = 1;
    double target_quality = std::numeric_limits<double>::quiet_NaN();
    std::size_t required_rr_samples = 0;
    std::size_t generated_rr_samples = 0;
    double verification_bound = std::numeric_limits<double>::quiet_NaN();
    bool verification_accepted = false;
    bool rr_sample_cap_triggered = false;
};

struct HistAttackSelection {
    std::vector<NodeId> attack_seeds;
    double selection_value = 0.0;
    double verification_lower_bound = std::numeric_limits<double>::quiet_NaN();
    std::size_t selection_rr_samples = 0;
    std::size_t validation_rr_samples = 0;
    std::size_t validation_early_stopped_samples = 0;
    std::size_t adaptive_rounds_used = 1;
    double target_quality = std::numeric_limits<double>::quiet_NaN();
    std::size_t required_rr_samples = 0;
    std::size_t generated_rr_samples = 0;
    bool verification_accepted = false;
    bool rr_sample_cap_triggered = false;
    std::string validation_status = "not_available";
};

void append_hist_rr_samples(
    HistRRSamplePool& pool,
    PropagationModel model,
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const ProgressCallback& progress_callback = {},
    const std::string& progress_stage = "HIST RR sampling") {
    if (sample_count == 0) {
        throw std::invalid_argument("HIST RR sample count must be positive");
    }

    if (pool.covered_by_node.empty()) {
        pool.covered_by_node.assign(graph.num_nodes(), {});
    }
    if (pool.covered_by_node.size() != graph.num_nodes()) {
        throw std::invalid_argument("HIST RR pool node universe does not match graph");
    }
    if (graph.num_nodes() == 0) {
        return;
    }

    std::mt19937_64 root_rng(seed);
    std::uniform_int_distribution<std::size_t> root_dist(0, graph.num_nodes() - 1);
    LazyLiveEdgeOracle live_oracle(graph.num_edges(), seed ^ 0x9e3779b97f4a7c15ULL);
    LazyTriggeringEdgeOracle triggering_oracle(graph.num_nodes(), seed ^ 0xbf58476d1ce4e5b9ULL);
    TraversalWorkspace workspace(graph.num_nodes());

    pool.rr_sets.reserve(pool.rr_sets.size() + sample_count);
    std::size_t reported_progress_bucket = 0;
    for (std::size_t local_id = 0; local_id < sample_count; ++local_id) {
        live_oracle.reset();
        triggering_oracle.reset();
        const NodeId root = static_cast<NodeId>(root_dist(root_rng));
        auto rr_set = reverse_rr_set(
            model,
            graph,
            root,
            live_oracle,
            triggering_oracle,
            workspace);
        std::sort(rr_set.begin(), rr_set.end());
        rr_set.erase(std::unique(rr_set.begin(), rr_set.end()), rr_set.end());

        const std::size_t sample_id = pool.rr_sets.size();
        for (const auto node : rr_set) {
            if (static_cast<std::size_t>(node) >= graph.num_nodes()) {
                throw std::out_of_range("RR set contains node outside graph");
            }
            pool.covered_by_node[node].push_back(sample_id);
        }
        pool.rr_sets.push_back(std::move(rr_set));

        const auto completed = local_id + 1;
        const auto progress_bucket = completed * 10 / sample_count;
        if (progress_callback &&
            (progress_bucket > reported_progress_bucket || completed == sample_count)) {
            progress_callback(
                progress_stage,
                100.0 * static_cast<double>(completed) /
                    static_cast<double>(sample_count));
            reported_progress_bucket = progress_bucket;
        }
    }
}

HistRRSamplePool generate_hist_rr_pool(
    PropagationModel model,
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const ProgressCallback& progress_callback = {},
    const std::string& progress_stage = "HIST RR sampling") {
    HistRRSamplePool pool;
    append_hist_rr_samples(
        pool, model, graph, sample_count, seed, progress_callback, progress_stage);

    return pool;
}

HistRRSamplePool generate_hist_validation_rr_pool_early_stop(
    PropagationModel model,
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const std::vector<NodeId>& defense_seeds) {
    if (sample_count == 0) {
        throw std::invalid_argument("HIST-Iter validation RR sample count must be positive");
    }

    HistRRSamplePool pool;
    pool.covered_by_node.assign(graph.num_nodes(), {});
    if (graph.num_nodes() == 0) {
        return pool;
    }

    std::vector<std::uint8_t> stop_mask(graph.num_nodes(), 0);
    for (const auto node : defense_seeds) {
        if (static_cast<std::size_t>(node) >= graph.num_nodes()) {
            throw std::out_of_range("Defense seed is outside graph");
        }
        stop_mask[node] = 1;
    }

    std::mt19937_64 root_rng(seed);
    std::uniform_int_distribution<std::size_t> root_dist(0, graph.num_nodes() - 1);
    LazyLiveEdgeOracle live_oracle(graph.num_edges(), seed ^ 0x9e3779b97f4a7c15ULL);
    LazyTriggeringEdgeOracle triggering_oracle(graph.num_nodes(), seed ^ 0xbf58476d1ce4e5b9ULL);
    TraversalWorkspace workspace(graph.num_nodes());
    pool.rr_sets.reserve(sample_count);

    for (std::size_t sample_id = 0; sample_id < sample_count; ++sample_id) {
        live_oracle.reset();
        triggering_oracle.reset();
        const NodeId root = static_cast<NodeId>(root_dist(root_rng));
        auto sample = reverse_rr_set_until_hit(
            model,
            graph,
            root,
            stop_mask,
            live_oracle,
            triggering_oracle,
            workspace);
        auto& rr_set = sample.rr_prefix;
        std::sort(rr_set.begin(), rr_set.end());
        rr_set.erase(std::unique(rr_set.begin(), rr_set.end()), rr_set.end());
        for (const auto node : rr_set) {
            pool.covered_by_node[node].push_back(sample_id);
        }
        if (sample.stopped_early) {
            ++pool.early_stopped_samples;
        }
        pool.rr_sets.push_back(std::move(rr_set));
    }

    return pool;
}

std::vector<std::uint8_t> rr_sets_covered_by_seeds(
    const HistRRSamplePool& pool,
    const std::vector<NodeId>& seeds) {
    std::vector<std::uint8_t> covered(pool.rr_sets.size(), 0);
    for (const auto seed : seeds) {
        if (static_cast<std::size_t>(seed) >= pool.covered_by_node.size()) {
            throw std::out_of_range("Seed node is outside RR pool node universe");
        }
        for (const auto rr_id : pool.covered_by_node[seed]) {
            covered[rr_id] = 1;
        }
    }
    return covered;
}

std::size_t choose_hist_sentinel_size(std::size_t total_budget, double epsilon) {
    if (total_budget <= 1) {
        return total_budget;
    }

    const auto quarter = std::max<std::size_t>(1, total_budget / 4);
    const double bounded_epsilon = std::min(std::max(epsilon, 1e-6), 0.999999);
    const double base = 1.0 - 1.0 / static_cast<double>(total_budget);
    std::size_t min_sub_size = 1;
    if (base > 0.0 && base < 1.0) {
        min_sub_size = static_cast<std::size_t>(
            std::ceil(std::log(1.0 - bounded_epsilon) / std::log(base)));
        min_sub_size = std::max<std::size_t>(1, min_sub_size);
    }
    return std::min(total_budget, std::max(quarter, min_sub_size));
}

double bounded_failure_probability(double failure_probability) {
    return std::min(std::max(failure_probability, 1e-12), 0.999999);
}

std::pair<std::vector<NodeId>, double> greedy_rr_residual_attack(
    const DirectedGraph& graph,
    const HistRRSamplePool& pool,
    const std::vector<NodeId>& defense_seeds,
    const std::vector<NodeId>& candidates,
    std::size_t budget) {
    const auto blocked = rr_sets_covered_by_seeds(pool, defense_seeds);
    std::vector<std::uint8_t> covered(pool.rr_sets.size(), 0);
    std::vector<std::uint8_t> selected(graph.num_nodes(), 0);
    std::vector<NodeId> attack;
    attack.reserve(std::min<std::size_t>(budget, candidates.size()));

    std::size_t covered_count = 0;
    for (std::size_t slot = 0; slot < budget; ++slot) {
        NodeId best_node = graph.num_nodes();
        std::size_t best_gain = 0;
        bool found = false;

        for (const auto candidate : candidates) {
            if (static_cast<std::size_t>(candidate) >= graph.num_nodes()) {
                throw std::out_of_range("Candidate node is outside graph");
            }
            if (selected[candidate]) {
                continue;
            }

            std::size_t gain = 0;
            for (const auto rr_id : pool.covered_by_node[candidate]) {
                if (!blocked[rr_id] && !covered[rr_id]) {
                    ++gain;
                }
            }

            if (!found || gain > best_gain || (gain == best_gain && candidate < best_node)) {
                best_node = candidate;
                best_gain = gain;
                found = true;
            }
        }

        if (!found || best_gain == 0) {
            break;
        }

        selected[best_node] = 1;
        attack.push_back(best_node);
        for (const auto rr_id : pool.covered_by_node[best_node]) {
            if (!blocked[rr_id] && !covered[rr_id]) {
                covered[rr_id] = 1;
                ++covered_count;
            }
        }
    }

    const double estimate = pool.rr_sets.empty()
        ? 0.0
        : static_cast<double>(graph.num_nodes()) *
              static_cast<double>(covered_count) /
              static_cast<double>(pool.rr_sets.size());
    return {attack, estimate};
}

double estimate_rr_residual_attack_value(
    const DirectedGraph& graph,
    const HistRRSamplePool& pool,
    const std::vector<NodeId>& defense_seeds,
    const std::vector<NodeId>& attack_seeds) {
    if (pool.rr_sets.empty()) {
        return 0.0;
    }

    const auto blocked = rr_sets_covered_by_seeds(pool, defense_seeds);
    std::vector<std::uint8_t> covered(pool.rr_sets.size(), 0);
    std::size_t covered_count = 0;
    for (const auto seed : attack_seeds) {
        if (static_cast<std::size_t>(seed) >= pool.covered_by_node.size()) {
            throw std::out_of_range("Attack seed is outside RR pool node universe");
        }
        for (const auto rr_id : pool.covered_by_node[seed]) {
            if (!blocked[rr_id] && !covered[rr_id]) {
                covered[rr_id] = 1;
                ++covered_count;
            }
        }
    }

    return static_cast<double>(graph.num_nodes()) *
           static_cast<double>(covered_count) /
           static_cast<double>(pool.rr_sets.size());
}

std::pair<std::vector<NodeId>, double> greedy_rr_coverage(
    const DirectedGraph& graph,
    const HistRRSamplePool& pool,
    std::size_t budget,
    const ProgressCallback& progress_callback = {},
    const std::string& progress_stage = "HIST coverage greedy") {
    std::vector<NodeId> seeds;
    seeds.reserve(std::min<std::size_t>(budget, graph.num_nodes()));
    std::vector<std::uint8_t> selected(graph.num_nodes(), 0);
    std::vector<std::uint8_t> covered(pool.rr_sets.size(), 0);
    std::size_t covered_count = 0;
    std::size_t reported_progress_bucket = 0;

    struct GainEntry {
        std::size_t gain;
        NodeId node;
    };
    struct GainEntryLess {
        bool operator()(const GainEntry& lhs, const GainEntry& rhs) const {
            if (lhs.gain != rhs.gain) {
                return lhs.gain < rhs.gain;
            }
            return lhs.node > rhs.node;
        }
    };

    // The inverted RR index makes each cached gain an upper bound: coverage
    // only grows, so stale candidates are refreshed only when they reach the
    // queue top instead of scanning every node at every seed-selection slot.
    std::priority_queue<GainEntry, std::vector<GainEntry>, GainEntryLess> gain_queue;
    for (NodeId node = 0; node < graph.num_nodes(); ++node) {
        gain_queue.push(GainEntry{pool.covered_by_node[node].size(), node});
    }

    for (std::size_t slot = 0; slot < budget; ++slot) {
        bool selected_node = false;
        while (!gain_queue.empty()) {
            const auto entry = gain_queue.top();
            gain_queue.pop();
            if (selected[entry.node] != 0) {
                continue;
            }

            std::size_t current_gain = 0;
            for (const auto rr_id : pool.covered_by_node[entry.node]) {
                if (covered[rr_id] == 0) {
                    ++current_gain;
                }
            }
            if (current_gain < entry.gain) {
                gain_queue.push(GainEntry{current_gain, entry.node});
                continue;
            }

            selected[entry.node] = 1;
            seeds.push_back(entry.node);
            for (const auto rr_id : pool.covered_by_node[entry.node]) {
                if (covered[rr_id] == 0) {
                    covered[rr_id] = 1;
                    ++covered_count;
                }
            }
            selected_node = true;
            break;
        }

        if (!selected_node) {
            break;
        }

        const auto completed = slot + 1;
        const auto progress_bucket = budget == 0 ? 0 : completed * 10 / budget;
        if (progress_callback && budget != 0 &&
            (progress_bucket > reported_progress_bucket || completed == budget)) {
            progress_callback(
                progress_stage,
                100.0 * static_cast<double>(completed) /
                    static_cast<double>(budget));
            reported_progress_bucket = progress_bucket;
        }
    }

    const double estimate = pool.rr_sets.empty()
        ? 0.0
        : static_cast<double>(graph.num_nodes()) *
          static_cast<double>(covered_count) /
          static_cast<double>(pool.rr_sets.size());
    return {seeds, estimate};
}

std::size_t hist_subsim_base_rr_samples(
    const BaselineOptions& options,
    double phase_delta) {
    const double delta = bounded_failure_probability(phase_delta);
    const auto theoretical_base = static_cast<std::size_t>(
        std::ceil(3.0 * std::log(1.0 / delta)));
    return std::max<std::size_t>(
        1,
        std::max(options.hist_rr_samples_floor, theoretical_base));
}

std::size_t hist_subsim_max_rr_samples(
    const DirectedGraph& graph,
    std::size_t budget,
    const BaselineOptions& options,
    double phase_epsilon,
    double phase_delta) {
    const double delta = bounded_failure_probability(phase_delta);
    const double epsilon = std::max(phase_epsilon, 1e-6);
    const double approximation = 1.0 - 1.0 / std::exp(1.0);
    const double alpha = std::sqrt(std::log(6.0 / delta));
    const double beta = std::sqrt(approximation * log_uniform_concentration_term(
        graph.num_nodes(), std::min(budget, graph.num_nodes()), delta));
    const double denominator = static_cast<double>(std::max<std::size_t>(1, budget)) * epsilon * epsilon;
    const double bound = 2.0 * static_cast<double>(graph.num_nodes()) *
        (alpha + beta) * (alpha + beta) / denominator + 1.0;
    if (!std::isfinite(bound) || bound >= static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        return std::numeric_limits<std::size_t>::max();
    }
    return std::max(
        hist_subsim_base_rr_samples(options, phase_delta),
        static_cast<std::size_t>(std::ceil(bound)));
}

std::size_t hist_subsim_round_target(
    std::size_t base,
    std::size_t maximum,
    std::size_t round) {
    std::size_t target = base;
    for (std::size_t i = 0; i < round && target < maximum; ++i) {
        target = target > maximum / 2 ? maximum : target * 2;
    }
    return std::min(target, maximum);
}

std::size_t hist_subsim_theoretical_rounds(std::size_t base, std::size_t maximum) {
    if (base == 0 || maximum <= base) {
        return 1;
    }
    std::size_t rounds = 1;
    std::size_t target = base;
    while (target < maximum) {
        target = target > maximum / 2 ? maximum : target * 2;
        ++rounds;
    }
    return rounds;
}

std::size_t hist_subsim_apply_cap(
    std::size_t requested,
    const BaselineOptions& options,
    bool& cap_triggered) {
    if (options.max_rr_samples != 0 && requested > options.max_rr_samples) {
        cap_triggered = true;
        return options.max_rr_samples;
    }
    return requested;
}

double hist_subsim_quality(double epsilon) {
    return std::max(0.0, 1.0 - 1.0 / std::exp(1.0) - epsilon);
}

double estimate_rr_coverage_value(
    const DirectedGraph& graph,
    const HistRRSamplePool& pool,
    const std::vector<NodeId>& seeds) {
    if (pool.rr_sets.empty()) {
        return 0.0;
    }

    const auto covered = rr_sets_covered_by_seeds(pool, seeds);
    const auto covered_count = static_cast<std::size_t>(
        std::count(covered.begin(), covered.end(), static_cast<std::uint8_t>(1)));
    return static_cast<double>(graph.num_nodes()) *
           static_cast<double>(covered_count) /
           static_cast<double>(pool.rr_sets.size());
}

struct HistCoverageSelection {
    std::vector<NodeId> defense_seeds;
    double selection_value = 0.0;
    double validation_value = std::numeric_limits<double>::quiet_NaN();
    double verification_lower_bound = std::numeric_limits<double>::quiet_NaN();
    std::size_t selection_rr_samples = 0;
    std::size_t validation_rr_samples = 0;
    std::size_t adaptive_rounds_used = 0;
    double target_quality = std::numeric_limits<double>::quiet_NaN();
    std::size_t required_rr_samples = 0;
    std::size_t generated_rr_samples = 0;
    bool verification_accepted = false;
    bool rr_sample_cap_triggered = false;
    std::string validation_status = "not_available";
};

HistCoverageSelection select_hist_coverage_subsim_adaptive(
    const DirectedGraph& graph,
    const BaselineOptions& options,
    const ProgressCallback& progress_callback) {
    HistRRSamplePool r1;
    HistCoverageSelection last_selection;
    const double phase_epsilon = options.epsilon;
    const double phase_delta = options.validation_failure_probability;
    const auto base = hist_subsim_base_rr_samples(options, phase_delta);
    const auto theoretical_max = hist_subsim_max_rr_samples(
        graph, options.defender_budget, options, phase_epsilon, phase_delta);
    const auto theoretical_rounds = hist_subsim_theoretical_rounds(base, theoretical_max);
    bool cap_triggered = false;

    for (std::size_t round = 0; round < options.max_adaptive_rounds; ++round) {
        const auto requested = hist_subsim_round_target(base, theoretical_max, round);
        const auto target = hist_subsim_apply_cap(requested, options, cap_triggered);
        if (progress_callback) {
            progress_callback(
                "HIST adaptive round " + std::to_string(round + 1) +
                    ": R1 target=" + std::to_string(target),
                0.0);
        }
        if (target > r1.rr_sets.size()) {
            append_hist_rr_samples(
                r1,
                options.propagation_model,
                graph,
                target - r1.rr_sets.size(),
                options.seed ^ (0x6a09e667f3bcc909ULL * (round + 1)),
                progress_callback,
                "HIST adaptive round " + std::to_string(round + 1) + ": R1 sampling");
        }

        auto [defense, selection_value] = greedy_rr_coverage(
            graph,
            r1,
            options.defender_budget,
            progress_callback,
            "HIST adaptive round " + std::to_string(round + 1) + ": coverage greedy");
        const auto r2_count = hist_subsim_apply_cap(r1.rr_sets.size(), options, cap_triggered);
        if (progress_callback) {
            progress_callback(
                "HIST adaptive round " + std::to_string(round + 1) +
                    ": independent R2 target=" + std::to_string(r2_count),
                0.0);
        }
        const auto r2 = generate_hist_rr_pool(
            options.propagation_model,
            graph,
            r2_count,
            options.seed ^ (0xbb67ae8584caa73bULL * (round + 1)),
            progress_callback,
            "HIST adaptive round " + std::to_string(round + 1) + ": R2 validation sampling");
        const double validation_value = estimate_rr_coverage_value(graph, r2, defense);
        const double lcb = std::max(0.0, lower_confidence_bound(
            validation_value,
            static_cast<double>(graph.num_nodes()),
            r2.rr_sets.size(),
            bounded_failure_probability(phase_delta) /
                static_cast<double>(theoretical_rounds)));

        last_selection.defense_seeds = std::move(defense);
        last_selection.selection_value = selection_value;
        last_selection.validation_value = validation_value;
        last_selection.verification_lower_bound = lcb;
        last_selection.selection_rr_samples = r1.rr_sets.size();
        last_selection.validation_rr_samples = r2.rr_sets.size();
        last_selection.adaptive_rounds_used = round + 1;
        last_selection.target_quality = hist_subsim_quality(phase_epsilon);
        last_selection.required_rr_samples = requested;
        last_selection.generated_rr_samples = r1.rr_sets.size();
        last_selection.verification_accepted =
            lcb >= last_selection.target_quality * selection_value;
        last_selection.rr_sample_cap_triggered = cap_triggered;
        last_selection.validation_status = cap_triggered
            ? "sample_cap_triggered"
            : (last_selection.verification_accepted
                ? "subsim_adaptive_accepted"
                : "subsim_adaptive_continue");
        if (progress_callback) {
            progress_callback(
                "HIST adaptive round " + std::to_string(round + 1) +
                    (last_selection.verification_accepted
                        ? ": validation accepted"
                        : ": validation requires expansion"),
                100.0);
        }
        if (last_selection.verification_accepted || cap_triggered ||
            r1.rr_sets.size() >= theoretical_max) {
            return last_selection;
        }
    }

    if (!last_selection.rr_sample_cap_triggered) {
        last_selection.validation_status = "subsim_adaptive_max_rounds";
    }
    return last_selection;
}

HistAttackSelection select_hist_attack_subsim_adaptive(
    const DirectedGraph& graph,
    const std::vector<NodeId>& defense_seeds,
    const std::vector<NodeId>& candidates,
    const BaselineOptions& options,
    std::uint64_t seed_offset) {
    HistRRSamplePool r1;
    HistAttackSelection last_selection;
    const double phase_epsilon = options.epsilon / 2.0;
    const double phase_delta = options.validation_failure_probability / 2.0;
    const auto base = hist_subsim_base_rr_samples(options, phase_delta);
    const auto theoretical_max = hist_subsim_max_rr_samples(
        graph, options.attacker_budget, options, phase_epsilon, phase_delta);
    const auto theoretical_rounds = hist_subsim_theoretical_rounds(base, theoretical_max);
    bool cap_triggered = false;

    for (std::size_t round = 0; round < options.max_adaptive_rounds; ++round) {
        const auto requested = hist_subsim_round_target(base, theoretical_max, round);
        const auto target = hist_subsim_apply_cap(requested, options, cap_triggered);
        if (target > r1.rr_sets.size()) {
            append_hist_rr_samples(
                r1,
                options.propagation_model,
                graph,
                target - r1.rr_sets.size(),
                options.seed ^ seed_offset ^ (0x9e3779b97f4a7c15ULL * (round + 1)));
        }

        auto [attack, selection_value] = greedy_rr_residual_attack(
            graph, r1, defense_seeds, candidates, options.attacker_budget);
        const auto r2_count = hist_subsim_apply_cap(
            r1.rr_sets.size(), options, cap_triggered);
        const auto r2 = generate_hist_validation_rr_pool_early_stop(
            options.propagation_model,
            graph,
            r2_count,
            options.seed ^ seed_offset ^ (0xbf58476d1ce4e5b9ULL * (round + 1)),
            defense_seeds);
        const double validation_value = estimate_rr_residual_attack_value(
            graph, r2, defense_seeds, attack);
        const double lcb = std::max(0.0, lower_confidence_bound(
            validation_value,
            static_cast<double>(graph.num_nodes()),
            r2.rr_sets.size(),
            bounded_failure_probability(phase_delta) /
                static_cast<double>(theoretical_rounds)));

        last_selection.attack_seeds = std::move(attack);
        last_selection.selection_value = selection_value;
        last_selection.selection_rr_samples = r1.rr_sets.size();
        last_selection.validation_rr_samples = r2.rr_sets.size();
        last_selection.validation_early_stopped_samples = r2.early_stopped_samples;
        last_selection.adaptive_rounds_used = round + 1;
        last_selection.target_quality = hist_subsim_quality(phase_epsilon);
        last_selection.required_rr_samples = requested;
        last_selection.generated_rr_samples = r1.rr_sets.size();
        last_selection.verification_lower_bound = lcb;
        last_selection.verification_accepted =
            lcb >= hist_subsim_quality(phase_epsilon) * selection_value;
        last_selection.rr_sample_cap_triggered = cap_triggered;
        last_selection.validation_status = cap_triggered
            ? "sample_cap_triggered"
            : (last_selection.verification_accepted
                ? "subsim_adaptive_accepted"
                : "subsim_adaptive_continue");
        if (last_selection.verification_accepted || cap_triggered ||
            r1.rr_sets.size() >= theoretical_max) {
            return last_selection;
        }
    }

    if (!last_selection.rr_sample_cap_triggered) {
        last_selection.validation_status = "subsim_adaptive_max_rounds";
    }
    return last_selection;
}

HistAttackSelection select_hist_attack(
    const DirectedGraph& graph,
    const std::vector<NodeId>& defense_seeds,
    const std::vector<NodeId>& candidates,
    const BaselineOptions& options,
    std::uint64_t seed_offset) {
    return select_hist_attack_subsim_adaptive(
        graph, defense_seeds, candidates, options, seed_offset);
}

std::pair<std::vector<NodeId>, double> greedy_rr_history_defense(
    const DirectedGraph& graph,
    const HistRRSamplePool& pool,
    const std::vector<std::vector<NodeId>>& attack_pool,
    const std::vector<NodeId>& initial_defense,
    const std::vector<NodeId>& candidates,
    std::size_t budget) {
    std::vector<std::vector<std::uint8_t>> residual_by_attack(
        attack_pool.size(),
        std::vector<std::uint8_t>(pool.rr_sets.size(), 0));
    std::vector<std::size_t> residual_counts(attack_pool.size(), 0);

    for (std::size_t attack_id = 0; attack_id < attack_pool.size(); ++attack_id) {
        for (const auto seed : attack_pool[attack_id]) {
            if (static_cast<std::size_t>(seed) >= graph.num_nodes()) {
                throw std::out_of_range("Attack seed is outside graph");
            }
            for (const auto rr_id : pool.covered_by_node[seed]) {
                if (!residual_by_attack[attack_id][rr_id]) {
                    residual_by_attack[attack_id][rr_id] = 1;
                    ++residual_counts[attack_id];
                }
            }
        }
    }

    std::vector<std::uint8_t> selected(graph.num_nodes(), 0);
    std::vector<NodeId> defense = initial_defense;
    defense.reserve(std::min<std::size_t>(budget, candidates.size() + initial_defense.size()));
    for (const auto node : initial_defense) {
        if (static_cast<std::size_t>(node) >= graph.num_nodes()) {
            throw std::out_of_range("Initial defense seed is outside graph");
        }
        if (selected[node]) {
            continue;
        }
        selected[node] = 1;
        for (std::size_t attack_id = 0; attack_id < attack_pool.size(); ++attack_id) {
            for (const auto rr_id : pool.covered_by_node[node]) {
                if (residual_by_attack[attack_id][rr_id]) {
                    residual_by_attack[attack_id][rr_id] = 0;
                    --residual_counts[attack_id];
                }
            }
        }
    }

    while (defense.size() < budget) {
        NodeId best_node = graph.num_nodes();
        std::size_t best_worst = std::numeric_limits<std::size_t>::max();
        bool found = false;

        for (const auto candidate : candidates) {
            if (static_cast<std::size_t>(candidate) >= graph.num_nodes()) {
                throw std::out_of_range("Candidate node is outside graph");
            }
            if (selected[candidate]) {
                continue;
            }

            std::size_t worst_after_candidate = 0;
            for (std::size_t attack_id = 0; attack_id < attack_pool.size(); ++attack_id) {
                std::size_t newly_blocked = 0;
                for (const auto rr_id : pool.covered_by_node[candidate]) {
                    if (residual_by_attack[attack_id][rr_id]) {
                        ++newly_blocked;
                    }
                }
                const auto residual_after =
                    residual_counts[attack_id] > newly_blocked
                        ? residual_counts[attack_id] - newly_blocked
                        : 0;
                worst_after_candidate = std::max(worst_after_candidate, residual_after);
            }

            if (!found ||
                worst_after_candidate < best_worst ||
                (worst_after_candidate == best_worst && candidate < best_node)) {
                best_node = candidate;
                best_worst = worst_after_candidate;
                found = true;
            }
        }

        if (!found) {
            break;
        }

        selected[best_node] = 1;
        defense.push_back(best_node);
        for (std::size_t attack_id = 0; attack_id < attack_pool.size(); ++attack_id) {
            for (const auto rr_id : pool.covered_by_node[best_node]) {
                if (residual_by_attack[attack_id][rr_id]) {
                    residual_by_attack[attack_id][rr_id] = 0;
                    --residual_counts[attack_id];
                }
            }
        }
    }

    const auto worst_count = residual_counts.empty()
        ? 0
        : *std::max_element(residual_counts.begin(), residual_counts.end());
    const double estimate = pool.rr_sets.empty()
        ? 0.0
        : static_cast<double>(graph.num_nodes()) *
              static_cast<double>(worst_count) /
              static_cast<double>(pool.rr_sets.size());
    return {defense, estimate};
}

std::pair<std::vector<NodeId>, double> greedy_rr_history_defense(
    const DirectedGraph& graph,
    const HistRRSamplePool& pool,
    const std::vector<std::vector<NodeId>>& attack_pool,
    const std::vector<NodeId>& candidates,
    std::size_t budget) {
    return greedy_rr_history_defense(graph, pool, attack_pool, {}, candidates, budget);
}

HistDefenseSelection validate_hist_defense(
    const DirectedGraph& graph,
    const HistRRSamplePool& validation_pool,
    const std::vector<std::vector<NodeId>>& attack_pool,
    const std::vector<NodeId>& defense,
    double failure_probability) {
    HistDefenseSelection result;
    result.defense_seeds = defense;
    result.validation_rr_samples = validation_pool.rr_sets.size();
    result.validation_early_stopped_samples = validation_pool.early_stopped_samples;
    if (validation_pool.rr_sets.empty() || attack_pool.empty()) {
        result.validation_value = 0.0;
        result.validation_upper_bound = 0.0;
        result.validation_status = "validated_empty";
        return result;
    }

    const auto blocked = rr_sets_covered_by_seeds(validation_pool, defense);
    std::size_t worst_count = 0;
    for (const auto& attack : attack_pool) {
        std::vector<std::uint8_t> residual(validation_pool.rr_sets.size(), 0);
        std::size_t count = 0;
        for (const auto seed : attack) {
            if (static_cast<std::size_t>(seed) >= graph.num_nodes()) {
                throw std::out_of_range("Attack seed is outside graph");
            }
            for (const auto rr_id : validation_pool.covered_by_node[seed]) {
                if (!blocked[rr_id] && !residual[rr_id]) {
                    residual[rr_id] = 1;
                    ++count;
                }
            }
        }
        worst_count = std::max(worst_count, count);
    }

    const double sample_count = static_cast<double>(validation_pool.rr_sets.size());
    const double mean = static_cast<double>(worst_count) / sample_count;
    const double delta = std::min(std::max(failure_probability, 1e-12), 0.999999);
    const double union_terms = std::max<double>(2.0, 2.0 * static_cast<double>(attack_pool.size()));
    const double radius = std::sqrt(std::log(union_terms / delta) / (2.0 * sample_count));
    const double upper_fraction = std::min(1.0, mean + radius);

    result.validation_value = static_cast<double>(graph.num_nodes()) * mean;
    result.validation_upper_bound = static_cast<double>(graph.num_nodes()) * upper_fraction;
    result.validation_status = "validated";
    return result;
}

HistDefenseSelection select_hist_defense_subsim_adaptive(
    const DirectedGraph& graph,
    const std::vector<std::vector<NodeId>>& attack_pool,
    const std::vector<NodeId>& candidates,
    const BaselineOptions& options,
    std::uint64_t seed_offset) {
    HistRRSamplePool sentinel_r1;
    const double phase_epsilon = options.epsilon / 2.0;
    const double phase_delta = options.validation_failure_probability / 2.0;
    const auto base = hist_subsim_base_rr_samples(options, phase_delta);
    const auto theoretical_max = hist_subsim_max_rr_samples(
        graph, options.defender_budget, options, phase_epsilon, phase_delta);
    const auto theoretical_rounds = hist_subsim_theoretical_rounds(base, theoretical_max);
    const auto sentinel_budget = choose_hist_sentinel_size(options.defender_budget, phase_epsilon);
    const double quality = hist_subsim_quality(options.epsilon);
    bool cap_triggered = false;
    std::vector<NodeId> sentinel;
    HistDefenseSelection last_selection;

    // SUBSIM phase 1: grow R1 geometrically until the sentinel prefix passes R2 validation.
    for (std::size_t round = 0; round < options.max_adaptive_rounds; ++round) {
        const auto requested = hist_subsim_round_target(base, theoretical_max, round);
        const auto target = hist_subsim_apply_cap(requested, options, cap_triggered);
        if (target > sentinel_r1.rr_sets.size()) {
            append_hist_rr_samples(
                sentinel_r1,
                options.propagation_model,
                graph,
                target - sentinel_r1.rr_sets.size(),
                options.seed ^ seed_offset ^ (0x94d049bb133111ebULL * (round + 1)));
        }
        auto sentinel_selection = greedy_rr_history_defense(
            graph, sentinel_r1, attack_pool, candidates, sentinel_budget);
        sentinel = std::move(sentinel_selection.first);

        const auto r2_count = hist_subsim_apply_cap(
            sentinel_r1.rr_sets.size(), options, cap_triggered);
        const auto r2 = generate_hist_rr_pool(
            options.propagation_model,
            graph,
            r2_count,
            options.seed ^ seed_offset ^ (0x2545f4914f6cdd1dULL * (round + 1)));
        const auto sentinel_validation = validate_hist_defense(
            graph, r2, attack_pool, sentinel,
            phase_delta / static_cast<double>(theoretical_rounds));
        const auto empty_validation = validate_hist_defense(
            graph, r2, attack_pool, {},
            phase_delta / static_cast<double>(theoretical_rounds));
        const double reduction = empty_validation.validation_upper_bound > 0.0
            ? std::max(0.0, 1.0 - sentinel_validation.validation_upper_bound /
                empty_validation.validation_upper_bound)
            : 1.0;
        const double target_reduction = std::max(
            0.0,
            1.0 - std::pow(1.0 - 1.0 / static_cast<double>(
                std::max<std::size_t>(1, options.defender_budget)), sentinel.size()) - phase_epsilon);
        if (reduction >= target_reduction || cap_triggered ||
            sentinel_r1.rr_sets.size() >= theoretical_max ||
            round + 1 == options.max_adaptive_rounds) {
            break;
        }
    }

    // SUBSIM phase 2: restart R1 for completion while retaining the validated sentinel set.
    HistRRSamplePool completion_r1;
    for (std::size_t round = 0; round < options.max_adaptive_rounds; ++round) {
        const auto requested = hist_subsim_round_target(base, theoretical_max, round);
        const auto target = hist_subsim_apply_cap(requested, options, cap_triggered);
        if (target > completion_r1.rr_sets.size()) {
            append_hist_rr_samples(
                completion_r1,
                options.propagation_model,
                graph,
                target - completion_r1.rr_sets.size(),
                options.seed ^ seed_offset ^ (0xd6e8feb86659fd93ULL * (round + 1)));
        }
        auto [defense, selection_value] = greedy_rr_history_defense(
            graph,
            completion_r1,
            attack_pool,
            sentinel,
            candidates,
            options.defender_budget);
        const auto r2_count = hist_subsim_apply_cap(
            completion_r1.rr_sets.size(), options, cap_triggered);
        const auto r2 = generate_hist_validation_rr_pool_early_stop(
            options.propagation_model,
            graph,
            r2_count,
            options.seed ^ seed_offset ^ (0xa4093822299f31d0ULL * (round + 1)),
            defense);
        const auto validation = validate_hist_defense(
            graph, r2, attack_pool, defense,
            phase_delta / static_cast<double>(theoretical_rounds));

        last_selection.defense_seeds = std::move(defense);
        last_selection.selection_value = selection_value;
        last_selection.sentinel_size = sentinel.size();
        last_selection.selection_rr_samples =
            sentinel_r1.rr_sets.size() + completion_r1.rr_sets.size();
        last_selection.validation_rr_samples = r2.rr_sets.size();
        last_selection.validation_early_stopped_samples = r2.early_stopped_samples;
        last_selection.validation_value = validation.validation_value;
        last_selection.validation_upper_bound = validation.validation_upper_bound;
        last_selection.adaptive_rounds_used = round + 1;
        last_selection.target_quality = quality;
        last_selection.required_rr_samples = requested;
        last_selection.generated_rr_samples = last_selection.selection_rr_samples;
        last_selection.verification_bound = validation.validation_upper_bound;
        last_selection.verification_accepted = validation.validation_upper_bound <=
            selection_value / std::max(quality, 1e-12);
        last_selection.rr_sample_cap_triggered = cap_triggered;
        last_selection.validation_status = cap_triggered
            ? "sample_cap_triggered"
            : (last_selection.verification_accepted
                ? "subsim_adaptive_accepted"
                : "subsim_adaptive_continue");
        if (last_selection.verification_accepted || cap_triggered ||
            completion_r1.rr_sets.size() >= theoretical_max) {
            return last_selection;
        }
    }

    if (!last_selection.rr_sample_cap_triggered) {
        last_selection.validation_status = "subsim_adaptive_max_rounds";
    }
    return last_selection;
}

HistDefenseSelection select_hist_defense(
    const DirectedGraph& graph,
    const std::vector<std::vector<NodeId>>& attack_pool,
    const std::vector<NodeId>& candidates,
    const BaselineOptions& options,
    std::uint64_t seed_offset) {
    return select_hist_defense_subsim_adaptive(
        graph, attack_pool, candidates, options, seed_offset);
}

BaselineResult run_hist_baseline(
    const DirectedGraph& graph,
    const BaselineOptions& options,
    std::ostream* progress_os) {
    AlgorithmTimer algorithm_timer;
    BaselineResult result;
    result.algorithm = BaselineAlgorithm::HIST;
    result.selection_objective = "rr_influence_coverage";
    result.uses_attack_pool = false;
    result.uses_fj_objective_in_selection = false;
    result.uses_certificate = false;
    result.certificate_status = "not_applicable";
    result.hist_rr_samples_floor = options.hist_rr_samples_floor;
    result.sampling_mode = BaselineSamplingMode::SubsimAdaptive;
    result.max_adaptive_rounds = options.max_adaptive_rounds;

    const ProgressCallback progress_callback =
        progress_os == nullptr
            ? ProgressCallback{}
            : ProgressCallback{[&](const std::string& stage, double completed_percent) {
                algorithm_timer.pause_for_output();
                print_progress_line(*progress_os, stage, completed_percent);
                algorithm_timer.resume_after_output();
            }};

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "HIST baseline\n";
        *progress_os << "  candidate_domain: all_nodes (" << graph.num_nodes() << ")\n";
        *progress_os << "  sampling_mode: subsim-adaptive\n";
        *progress_os << "  hist_rr_samples_floor: " << options.hist_rr_samples_floor << '\n';
        *progress_os << "  r1_r2_schedule: epsilon_delta_formula\n";
        *progress_os << "  selection_failure_probability: "
                     << format_decimal(options.validation_failure_probability) << '\n';
        algorithm_timer.resume_after_output();
    }

    const auto selection = select_hist_coverage_subsim_adaptive(
        graph, options, progress_callback);
    result.defense_seeds = selection.defense_seeds;

    BaselineIterationLog log;
    log.iteration = 1;
    log.defense_seeds = result.defense_seeds;
    log.selection_objective_value = selection.selection_value;
    log.iteration_runtime_sec = algorithm_timer.elapsed_seconds();
    log.selection_rr_samples = selection.selection_rr_samples;
    log.validation_rr_samples = selection.validation_rr_samples;
    log.validation_objective_value = selection.validation_value;
    log.validation_status = selection.validation_status;
    log.sampling_mode = "subsim-adaptive";
    log.adaptive_rounds_used = selection.adaptive_rounds_used;
    log.target_quality = selection.target_quality;
    log.required_rr_samples = selection.required_rr_samples;
    log.generated_rr_samples = selection.generated_rr_samples;
    log.verification_samples = selection.validation_rr_samples;
    log.verification_bound = selection.verification_lower_bound;
    log.verification_accepted = selection.verification_accepted;
    log.rr_sample_cap_triggered = selection.rr_sample_cap_triggered;
    result.iterations.push_back(log);
    result.algorithm_runtime_sec = algorithm_timer.elapsed_seconds();

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "  adaptive_rounds_used: "
                     << selection.adaptive_rounds_used << '\n';
        *progress_os << "  required_rr_samples: "
                     << selection.required_rr_samples << '\n';
        *progress_os << "  generated_rr_samples: "
                     << selection.generated_rr_samples << '\n';
        *progress_os << "  validation_rr_samples: "
                     << selection.validation_rr_samples << '\n';
        *progress_os << "  validation_value: "
                     << format_decimal(selection.validation_value) << '\n';
        *progress_os << "  verification_lower_bound: "
                     << format_decimal(selection.verification_lower_bound) << '\n';
        *progress_os << "  validation_status: "
                     << selection.validation_status << '\n';
        *progress_os << "  selected_defense_seeds: "
                     << format_seed_set_preview(result.defense_seeds) << '\n';
        *progress_os << "  estimated_coverage: "
                     << format_decimal(selection.selection_value) << '\n';
        print_progress_line(*progress_os, "HIST baseline", 90.0);
        algorithm_timer.resume_after_output();
    }
    return result;
}

BaselineResult run_random_baseline(
    const DirectedGraph& graph,
    const BaselineOptions& options,
    std::ostream* progress_os) {
    AlgorithmTimer algorithm_timer;
    std::vector<NodeId> candidates = all_candidate_nodes(graph);
    std::mt19937_64 generator(options.seed ^ 0xbb67ae8584caa73bULL);
    std::shuffle(candidates.begin(), candidates.end(), generator);
    const auto take = std::min<std::size_t>(options.defender_budget, candidates.size());

    BaselineResult result;
    result.algorithm = BaselineAlgorithm::Random;
    result.selection_objective = "uniform_random_selection";
    result.uses_attack_pool = false;
    result.uses_fj_objective_in_selection = false;
    result.uses_certificate = false;
    result.certificate_status = "not_applicable";
    result.sampling_mode = BaselineSamplingMode::NotApplicable;
    result.defense_seeds.assign(candidates.begin(), candidates.begin() + take);

    BaselineIterationLog log;
    log.iteration = 1;
    log.defense_seeds = result.defense_seeds;
    log.sampling_mode = "not_applicable";
    log.validation_status = "not_applicable";
    log.iteration_runtime_sec = algorithm_timer.elapsed_seconds();
    result.iterations.push_back(log);
    result.algorithm_runtime_sec = algorithm_timer.elapsed_seconds();

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "Random baseline\n";
        *progress_os << "  candidate_domain: all_nodes (" << graph.num_nodes() << ")\n";
        *progress_os << "  selected_defense_seeds: "
                     << format_seed_set_preview(result.defense_seeds) << '\n';
        print_progress_line(*progress_os, "Random baseline", 90.0);
        algorithm_timer.resume_after_output();
    }
    return result;
}

BaselineResult run_exact_baseline(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const BaselineOptions& options,
    std::ostream* progress_os) {
    AlgorithmTimer algorithm_timer;
    BaselineResult result;
    result.algorithm = BaselineAlgorithm::Exact;
    result.selection_objective = "exact_enumeration_mc_minimax";
    result.uses_attack_pool = false;
    result.uses_fj_objective_in_selection = true;
    result.uses_certificate = false;
    result.certificate_status = "not_applicable";
    result.sampling_mode = BaselineSamplingMode::NotApplicable;
    result.selection_mc_simulations = options.mc_simulations;

    const auto defense_set_count = saturating_binomial(
        graph.num_nodes(), options.defender_budget);
    const auto attack_set_count = saturating_binomial(
        graph.num_nodes(), options.attacker_budget);
    const auto total_seed_pairs = saturating_product(defense_set_count, attack_set_count);

    double best_worst_value = std::numeric_limits<double>::infinity();
    std::vector<NodeId> best_defense;
    std::vector<NodeId> best_attack;
    std::uint64_t pair_index = 0;
    std::size_t evaluated_pairs = 0;
    std::size_t next_progress_percent = 10;

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "Exact baseline\n";
        *progress_os << "  search: exhaustive defense/attack seed enumeration\n";
        *progress_os << "  defense_seed_sets: " << defense_set_count << '\n';
        *progress_os << "  attack_seed_sets_per_defense: " << attack_set_count << '\n';
        *progress_os << "  total_seed_pairs: " << total_seed_pairs << '\n';
        *progress_os << "  mc_simulations_per_seed_pair: "
                     << options.mc_simulations << '\n';
        print_progress_line(*progress_os, "Exact enumeration: seed-pair evaluation", 0.0);
        algorithm_timer.resume_after_output();
    }

    enumerate_seed_sets(
        graph.num_nodes(),
        options.defender_budget,
        [&](const std::vector<NodeId>& defense) {
            double worst_value = -std::numeric_limits<double>::infinity();
            std::vector<NodeId> worst_attack;
            enumerate_seed_sets(
                graph.num_nodes(),
                options.attacker_budget,
                [&](const std::vector<NodeId>& attack) {
                    const auto evaluation = estimate_expected_conflict(
                        options.propagation_model,
                        graph,
                        conflict,
                        innate,
                        attack,
                        defense,
                        params,
                        MonteCarloOptions{
                            options.mc_simulations,
                            options.seed ^ (0x9e3779b97f4a7c15ULL * (++pair_index))
                        });
                    ++evaluated_pairs;
                    while (progress_os != nullptr &&
                           next_progress_percent <= 100 &&
                           static_cast<long double>(evaluated_pairs) * 100.0L >=
                               static_cast<long double>(total_seed_pairs) *
                                   static_cast<long double>(next_progress_percent)) {
                        algorithm_timer.pause_for_output();
                        *progress_os << "  evaluated_seed_pairs: " << evaluated_pairs
                                     << '/' << total_seed_pairs << '\n';
                        print_progress_line(
                            *progress_os,
                            "Exact enumeration: seed-pair evaluation",
                            static_cast<double>(next_progress_percent));
                        algorithm_timer.resume_after_output();
                        next_progress_percent += 10;
                    }
                    if (evaluation.mean > worst_value ||
                        (evaluation.mean == worst_value && attack < worst_attack)) {
                        worst_value = evaluation.mean;
                        worst_attack = attack;
                    }
                });

            if (worst_value < best_worst_value ||
                (worst_value == best_worst_value && defense < best_defense)) {
                best_worst_value = worst_value;
                best_defense = defense;
                best_attack = worst_attack;
            }
        });

    result.defense_seeds = std::move(best_defense);
    result.attack_seeds = std::move(best_attack);
    BaselineIterationLog log;
    log.iteration = 1;
    log.defense_seeds = result.defense_seeds;
    log.attack_seeds = result.attack_seeds;
    log.selection_objective_value = best_worst_value;
    log.selection_rr_samples = 0;
    log.validation_status = "exact_seed_enumeration_mc";
    log.sampling_mode = "not_applicable";
    log.iteration_runtime_sec = algorithm_timer.elapsed_seconds();
    result.iterations.push_back(log);
    result.algorithm_runtime_sec = algorithm_timer.elapsed_seconds();

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "  selected_defense_seeds: "
                     << format_seed_set_preview(result.defense_seeds) << '\n';
        *progress_os << "  selected_attack_seeds: "
                     << format_seed_set_preview(result.attack_seeds) << '\n';
        *progress_os << "  worst_case_mc_objective: "
                     << format_decimal(best_worst_value) << '\n';
        algorithm_timer.resume_after_output();
    }
    return result;
}

BaselineResult run_degree_baseline(
    const DirectedGraph& graph,
    const BaselineOptions& options,
    std::ostream* progress_os) {
    AlgorithmTimer algorithm_timer;

    BaselineResult result;
    result.algorithm = BaselineAlgorithm::Degree;
    result.selection_objective = "directed_out_degree";
    result.uses_attack_pool = false;
    result.uses_fj_objective_in_selection = false;
    result.uses_certificate = false;
    result.certificate_status = "not_applicable";
    result.fjc_pair_rr_samples = options.fjc_pair_rr_samples;
    result.sampling_mode = BaselineSamplingMode::Fixed;
    result.max_adaptive_rounds = options.max_adaptive_rounds;
    result.defense_seeds = select_degree_defense(graph, options.defender_budget);

    BaselineIterationLog log;
    log.iteration = 1;
    log.attack_pool_size = 0;
    log.defense_seeds = result.defense_seeds;
    log.selection_objective_value = 0.0;

    log.iteration_runtime_sec = algorithm_timer.elapsed_seconds();
    result.algorithm_runtime_sec = log.iteration_runtime_sec;
    result.iterations.push_back(log);

    if (progress_os != nullptr) {
        *progress_os << "Degree baseline\n";
        *progress_os << "  selected_defense_seeds: "
                     << format_seed_set_preview(result.defense_seeds) << '\n';
        *progress_os << "  algorithm_runtime_sec: "
                     << format_decimal(result.algorithm_runtime_sec) << '\n';
        print_progress_line(*progress_os, "Degree baseline", 90.0);
    }
    return result;
}

BaselineResult run_fjc_iter_baseline(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const BaselineOptions& options,
    std::ostream* progress_os) {
    AlgorithmTimer algorithm_timer;
    const auto* dense_conflict = dynamic_cast<const DenseConflict*>(&conflict);
    if (dense_conflict == nullptr) {
        throw std::invalid_argument(
            "FJC-Iter requires --conflict-backend dense because its selection phase uses pair-RR samples");
    }
    const std::string selection_backend = "dense_pair_rr";

    BaselineResult result;
    result.algorithm = BaselineAlgorithm::FJCIter;
    result.selection_objective = "pair_rr_fj_cascade_conflict";
    result.uses_attack_pool = true;
    result.uses_fj_objective_in_selection = true;
    result.uses_certificate = false;
    result.certificate_status = "not_applicable";
    result.selection_backend = selection_backend;
    result.selection_context = "pair_rr_realization";
    result.fjc_pair_rr_samples = options.fjc_pair_rr_samples;
    result.sampling_mode = BaselineSamplingMode::Fixed;
    result.max_adaptive_rounds = options.max_adaptive_rounds;

    const auto candidates = all_candidate_nodes(graph);

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "FJC-Iter baseline\n";
        *progress_os << "  candidate_domain: all_nodes (" << candidates.size() << ")\n";
        *progress_os << "  pair_rr_samples: " << options.fjc_pair_rr_samples << '\n';
        *progress_os << "  selection_backend: " << selection_backend << '\n';
        algorithm_timer.resume_after_output();
    }

    std::vector<std::vector<NodeId>> attack_pool;
    std::vector<NodeId> initial_attack;
    double initial_value = 0.0;
    std::size_t initial_attack_rr_samples = 0;
    const auto initial_selection = greedy_fjc_pair_rr_attack(
        graph,
        *dense_conflict,
        innate,
        params,
        {},
        candidates,
        options,
        0x452821e638d01377ULL);
    initial_attack = initial_selection.attack_seeds;
    initial_value = initial_selection.selection_value;
    initial_attack_rr_samples = initial_selection.selection_rr_samples;
    attack_pool.push_back(initial_attack);
    result.attack_seeds = initial_attack;

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "  initial_attack_seeds: "
                     << format_seed_set_preview(initial_attack) << '\n';
        *progress_os << "  initial_attack_conflict: " << format_decimal(initial_value) << '\n';
        print_progress_line(*progress_os, "FJC-Iter initialization", 35.0);
        *progress_os << "  initial_attack_rr_samples: "
                     << initial_attack_rr_samples << '\n';
        algorithm_timer.resume_after_output();
    }

    for (std::size_t iteration = 1; iteration <= options.iterations; ++iteration) {
        const auto iteration_started = std::chrono::steady_clock::now();

        std::vector<NodeId> defense;
        std::vector<NodeId> attack;
        double defense_value = 0.0;
        double attack_value = 0.0;
        std::size_t defense_rr_samples = 0;
        std::size_t attack_rr_samples = 0;
        const auto defense_selection = greedy_fjc_pair_rr_defense(
            graph,
            *dense_conflict,
            innate,
            params,
            attack_pool,
            candidates,
            options,
            0xbe5466cf34e90c6cULL ^ iteration);
        defense = defense_selection.defense_seeds;
        defense_value = defense_selection.selection_value;
        defense_rr_samples = defense_selection.selection_rr_samples;

        const auto attack_selection = greedy_fjc_pair_rr_attack(
            graph,
            *dense_conflict,
            innate,
            params,
            defense,
            candidates,
            options,
            0xc0ac29b7c97c50ddULL ^ iteration);
        attack = attack_selection.attack_seeds;
        attack_value = attack_selection.selection_value;
        attack_rr_samples = attack_selection.selection_rr_samples;

        const bool duplicate_attack = attack_pool_contains(attack_pool, attack);
        if (!duplicate_attack) {
            attack_pool.push_back(attack);
        }

        const auto iteration_finished = std::chrono::steady_clock::now();
        BaselineIterationLog log;
        log.iteration = iteration;
        log.attack_pool_size = attack_pool.size();
        log.defense_seeds = defense;
        log.attack_seeds = attack;
        log.selection_objective_value = defense_value;
        log.iteration_runtime_sec =
            std::chrono::duration<double>(iteration_finished - iteration_started).count();
        log.duplicate_attack = duplicate_attack;
        log.selection_rr_samples = defense_rr_samples;
        log.generated_rr_samples = defense_rr_samples;
        log.attack_generated_rr_samples = attack_rr_samples;
        log.validation_status = selection_backend;
        log.attack_validation_status = selection_backend;
        result.iterations.push_back(log);
        result.defense_seeds = defense;
        result.attack_seeds = attack;

        if (progress_os != nullptr) {
            algorithm_timer.pause_for_output();
            *progress_os << "Iteration " << iteration << '\n';
            *progress_os << "  attack_pool_size: " << attack_pool.size() << '\n';
            *progress_os << "  defense_seeds: " << format_seed_set_preview(defense) << '\n';
            *progress_os << "  attack_seeds: " << format_seed_set_preview(attack) << '\n';
            *progress_os << "  historical_worst_conflict: "
                         << format_decimal(defense_value) << '\n';
            *progress_os << "  new_attack_conflict: " << format_decimal(attack_value) << '\n';
            *progress_os << "  selection_backend: " << selection_backend << '\n';
            *progress_os << "  selection_rr_samples: " << defense_rr_samples << '\n';
            *progress_os << "  attack_rr_samples: " << attack_rr_samples << '\n';
            *progress_os << "  duplicate_attack: " << (duplicate_attack ? "true" : "false") << '\n';
            *progress_os << "  iteration_runtime_sec: "
                         << format_decimal(log.iteration_runtime_sec) << '\n';
            print_progress_line(
                *progress_os,
                "FJC-Iter iteration " + std::to_string(iteration),
                35.0 + 55.0 * static_cast<double>(iteration) /
                    static_cast<double>(options.iterations));
            algorithm_timer.resume_after_output();
        }
    }

    result.algorithm_runtime_sec = algorithm_timer.elapsed_seconds();
    return result;
}

BaselineResult run_fjc_linear_iter_baseline(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const BaselineOptions& options,
    std::ostream* progress_os) {
    AlgorithmTimer algorithm_timer;
    const std::string selection_backend =
        std::string("linear_rr_") + conflict.backend_name() + "_seed_context";

    BaselineResult result;
    result.algorithm = BaselineAlgorithm::FJCLinearIter;
    result.selection_objective = "linear_rr_fj_cascade_conflict_surrogate";
    result.uses_attack_pool = true;
    result.uses_fj_objective_in_selection = true;
    result.uses_certificate = false;
    result.certificate_status = "not_applicable";
    result.selection_backend = selection_backend;
    result.selection_context = "opponent_seed_mask";
    result.fjc_linear_rr_samples = options.fjc_linear_rr_samples;
    result.sampling_mode = BaselineSamplingMode::Fixed;
    result.max_adaptive_rounds = options.max_adaptive_rounds;

    const auto candidates = all_candidate_nodes(graph);
    const auto report_progress = [&](const std::string& stage, double percent) {
        if (progress_os == nullptr) {
            return;
        }
        algorithm_timer.pause_for_output();
        print_progress_line(*progress_os, stage, percent);
        algorithm_timer.resume_after_output();
    };

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "FJC-Linear-Iter baseline\n";
        *progress_os << "  candidate_domain: all_nodes (" << candidates.size() << ")\n";
        *progress_os << "  linear_rr_samples: " << options.fjc_linear_rr_samples << '\n';
        *progress_os << "  selection_backend: " << selection_backend << '\n';
        *progress_os << "  selection_context: opponent_seed_mask\n";
        algorithm_timer.resume_after_output();
    }

    std::vector<std::vector<NodeId>> attack_pool;
    const auto initial_selection = greedy_fjc_linear_rr_attack(
        graph,
        conflict,
        innate,
        params,
        {},
        candidates,
        options,
        0x452821e638d01377ULL,
        report_progress);
    attack_pool.push_back(initial_selection.attack_seeds);
    result.attack_seeds = initial_selection.attack_seeds;

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "  initial_attack_seeds: "
                     << format_seed_set_preview(initial_selection.attack_seeds) << '\n';
        *progress_os << "  initial_attack_linear_surrogate_gain: "
                     << format_decimal(initial_selection.selection_value) << '\n';
        *progress_os << "  initial_attack_rr_samples: "
                     << initial_selection.selection_rr_samples << '\n';
        *progress_os << "  initial_attack_positive_root_weights: "
                     << initial_selection.positive_root_weights << '\n';
        *progress_os << "  initial_attack_root_weight_min: "
                     << format_decimal(initial_selection.root_weight_min) << '\n';
        *progress_os << "  initial_attack_root_weight_max: "
                     << format_decimal(initial_selection.root_weight_max) << '\n';
        algorithm_timer.resume_after_output();
    }

    for (std::size_t iteration = 1; iteration <= options.iterations; ++iteration) {
        AlgorithmTimer iteration_timer;
        const auto iteration_progress = [&](const std::string& stage, double percent) {
            if (progress_os == nullptr) {
                return;
            }
            algorithm_timer.pause_for_output();
            iteration_timer.pause_for_output();
            print_progress_line(*progress_os, stage, percent);
            iteration_timer.resume_after_output();
            algorithm_timer.resume_after_output();
        };

        const auto defense_selection = greedy_fjc_linear_rr_defense(
            graph,
            conflict,
            innate,
            params,
            attack_pool,
            candidates,
            options,
            0xbe5466cf34e90c6cULL ^ iteration,
            iteration_progress);
        const auto attack_selection = greedy_fjc_linear_rr_attack(
            graph,
            conflict,
            innate,
            params,
            defense_selection.defense_seeds,
            candidates,
            options,
            0xc0ac29b7c97c50ddULL ^ iteration,
            iteration_progress);

        const bool duplicate_attack =
            attack_pool_contains(attack_pool, attack_selection.attack_seeds);
        if (!duplicate_attack) {
            attack_pool.push_back(attack_selection.attack_seeds);
        }

        BaselineIterationLog log;
        log.iteration = iteration;
        log.attack_pool_size = attack_pool.size();
        log.defense_seeds = defense_selection.defense_seeds;
        log.attack_seeds = attack_selection.attack_seeds;
        log.selection_objective_value = defense_selection.selection_value;
        log.iteration_runtime_sec = iteration_timer.elapsed_seconds();
        log.duplicate_attack = duplicate_attack;
        log.selection_rr_samples = defense_selection.selection_rr_samples;
        log.generated_rr_samples = defense_selection.selection_rr_samples;
        log.attack_generated_rr_samples = attack_selection.selection_rr_samples;
        log.validation_status = selection_backend;
        log.attack_validation_status = selection_backend;
        log.sampling_mode = "weighted-rr-linear";
        result.iterations.push_back(log);
        result.defense_seeds = defense_selection.defense_seeds;
        result.attack_seeds = attack_selection.attack_seeds;

        if (progress_os != nullptr) {
            algorithm_timer.pause_for_output();
            *progress_os << "Iteration " << iteration << '\n';
            *progress_os << "  attack_pool_size: " << attack_pool.size() << '\n';
            *progress_os << "  defense_seeds: "
                         << format_seed_set_preview(defense_selection.defense_seeds) << '\n';
            *progress_os << "  attack_seeds: "
                         << format_seed_set_preview(attack_selection.attack_seeds) << '\n';
            *progress_os << "  historical_worst_linear_residual: "
                         << format_decimal(defense_selection.selection_value) << '\n';
            *progress_os << "  new_attack_linear_surrogate_gain: "
                         << format_decimal(attack_selection.selection_value) << '\n';
            *progress_os << "  selection_backend: " << selection_backend << '\n';
            *progress_os << "  defense_selection_rr_samples: "
                         << defense_selection.selection_rr_samples << '\n';
            *progress_os << "  attack_selection_rr_samples: "
                         << attack_selection.selection_rr_samples << '\n';
            *progress_os << "  attack_positive_root_weights: "
                         << attack_selection.positive_root_weights << '\n';
            *progress_os << "  attack_root_weight_min: "
                         << format_decimal(attack_selection.root_weight_min) << '\n';
            *progress_os << "  attack_root_weight_max: "
                         << format_decimal(attack_selection.root_weight_max) << '\n';
            *progress_os << "  duplicate_attack: "
                         << (duplicate_attack ? "true" : "false") << '\n';
            *progress_os << "  iteration_runtime_sec: "
                         << format_decimal(log.iteration_runtime_sec) << '\n';
            algorithm_timer.resume_after_output();
        }
    }

    result.algorithm_runtime_sec = algorithm_timer.elapsed_seconds();
    return result;
}

BaselineResult run_hist_iter_baseline(
    const DirectedGraph& graph,
    const BaselineOptions& options,
    std::ostream* progress_os) {
    AlgorithmTimer algorithm_timer;

    BaselineResult result;
    result.algorithm = BaselineAlgorithm::HISTIter;
    result.selection_objective = "rr_residual_reach";
    result.uses_attack_pool = true;
    result.uses_fj_objective_in_selection = false;
    result.uses_certificate = false;
    result.certificate_status = "not_applicable";
    result.hist_rr_samples_floor = options.hist_rr_samples_floor;
    result.sampling_mode = BaselineSamplingMode::SubsimAdaptive;
    result.max_adaptive_rounds = options.max_adaptive_rounds;

    const auto candidates = all_candidate_nodes(graph);

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "HIST-Iter baseline\n";
        *progress_os << "  candidate_domain: all_nodes (" << candidates.size() << ")\n";
        *progress_os << "  sampling_mode: subsim-adaptive\n";
        *progress_os << "  hist_rr_samples_floor: " << options.hist_rr_samples_floor << '\n';
        *progress_os << "  r1_r2_schedule: epsilon_delta_formula\n";
        *progress_os << "  selection_failure_probability: "
                     << format_decimal(options.validation_failure_probability) << '\n';
        algorithm_timer.resume_after_output();
    }

    std::vector<std::vector<NodeId>> attack_pool;
    const auto initial_attack_selection = select_hist_attack(
        graph,
        {},
        candidates,
        options,
        0x243f6a8885a308d3ULL);
    const auto& initial_attack = initial_attack_selection.attack_seeds;
    const double initial_value = initial_attack_selection.selection_value;
    attack_pool.push_back(initial_attack);
    result.attack_seeds = initial_attack;

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "  initial_attack_seeds: "
                     << format_seed_set_preview(initial_attack) << '\n';
        *progress_os << "  initial_attack_rr_residual: " << format_decimal(initial_value) << '\n';
        *progress_os << "  initial_attack_status: "
                     << initial_attack_selection.validation_status << '\n';
        print_progress_line(*progress_os, "HIST-Iter initialization", 35.0);
        algorithm_timer.resume_after_output();
    }

    for (std::size_t iteration = 1; iteration <= options.iterations; ++iteration) {
        const auto iteration_started = std::chrono::steady_clock::now();

        const auto defense_selection = select_hist_defense(
            graph,
            attack_pool,
            candidates,
            options,
            0x13198a2e03707344ULL ^ iteration);
        const auto& defense = defense_selection.defense_seeds;
        const auto attack_selection = select_hist_attack(
            graph,
            defense,
            candidates,
            options,
            0xa4093822299f31d0ULL ^ iteration);
        const auto& attack = attack_selection.attack_seeds;
        const double attack_value = attack_selection.selection_value;

        const bool duplicate_attack = attack_pool_contains(attack_pool, attack);
        if (!duplicate_attack) {
            attack_pool.push_back(attack);
        }

        const auto iteration_finished = std::chrono::steady_clock::now();
        BaselineIterationLog log;
        log.iteration = iteration;
        log.attack_pool_size = attack_pool.size();
        log.defense_seeds = defense;
        log.attack_seeds = attack;
        log.selection_objective_value = defense_selection.selection_value;
        log.iteration_runtime_sec =
            std::chrono::duration<double>(iteration_finished - iteration_started).count();
        log.duplicate_attack = duplicate_attack;
        log.sentinel_size = defense_selection.sentinel_size;
        log.selection_rr_samples = defense_selection.selection_rr_samples;
        log.validation_rr_samples = defense_selection.validation_rr_samples;
        log.validation_early_stopped_samples =
            defense_selection.validation_early_stopped_samples;
        log.validation_objective_value = defense_selection.validation_value;
        log.validation_upper_bound = defense_selection.validation_upper_bound;
        log.validation_status = defense_selection.validation_status;
        log.sampling_mode = "subsim-adaptive";
        log.adaptive_rounds_used = defense_selection.adaptive_rounds_used;
        log.target_quality = defense_selection.target_quality;
        log.required_rr_samples = defense_selection.required_rr_samples;
        log.generated_rr_samples = defense_selection.generated_rr_samples;
        log.verification_samples = defense_selection.validation_rr_samples;
        log.verification_bound = defense_selection.verification_bound;
        log.verification_accepted = defense_selection.verification_accepted;
        log.rr_sample_cap_triggered = defense_selection.rr_sample_cap_triggered;
        log.attack_adaptive_rounds_used = attack_selection.adaptive_rounds_used;
        log.attack_target_quality = attack_selection.target_quality;
        log.attack_required_rr_samples = attack_selection.required_rr_samples;
        log.attack_generated_rr_samples = attack_selection.generated_rr_samples;
        log.attack_verification_samples = attack_selection.validation_rr_samples;
        log.attack_validation_early_stopped_samples =
            attack_selection.validation_early_stopped_samples;
        log.attack_verification_lower_bound = attack_selection.verification_lower_bound;
        log.attack_verification_accepted = attack_selection.verification_accepted;
        log.attack_rr_sample_cap_triggered = attack_selection.rr_sample_cap_triggered;
        log.attack_validation_status = attack_selection.validation_status;
        result.iterations.push_back(log);
        result.defense_seeds = defense;
        result.attack_seeds = attack;

        if (progress_os != nullptr) {
            algorithm_timer.pause_for_output();
            *progress_os << "Iteration " << iteration << '\n';
            *progress_os << "  attack_pool_size: " << attack_pool.size() << '\n';
            *progress_os << "  defense_seeds: " << format_seed_set_preview(defense) << '\n';
            *progress_os << "  attack_seeds: " << format_seed_set_preview(attack) << '\n';
            *progress_os << "  sentinel_size: " << defense_selection.sentinel_size << '\n';
            *progress_os << "  selection_rr_samples: "
                         << defense_selection.selection_rr_samples << '\n';
            *progress_os << "  validation_rr_samples: "
                         << defense_selection.validation_rr_samples << '\n';
            *progress_os << "  validation_early_stopped_samples: "
                         << defense_selection.validation_early_stopped_samples << '\n';
            *progress_os << "  historical_worst_rr_residual: "
                         << format_decimal(defense_selection.selection_value) << '\n';
            *progress_os << "  validation_worst_rr_residual: "
                         << format_decimal(defense_selection.validation_value) << '\n';
            *progress_os << "  validation_upper_bound: "
                         << format_decimal(defense_selection.validation_upper_bound) << '\n';
            *progress_os << "  validation_status: "
                         << defense_selection.validation_status << '\n';
            *progress_os << "  defense_adaptive_rounds_used: "
                         << defense_selection.adaptive_rounds_used << '\n';
            *progress_os << "  defense_target_quality: "
                         << format_decimal(defense_selection.target_quality) << '\n';
            *progress_os << "  defense_verification_accepted: "
                         << (defense_selection.verification_accepted ? "true" : "false") << '\n';
            *progress_os << "  new_attack_rr_residual: " << format_decimal(attack_value) << '\n';
            *progress_os << "  attack_adaptive_rounds_used: "
                         << attack_selection.adaptive_rounds_used << '\n';
            *progress_os << "  attack_verification_lower_bound: "
                         << format_decimal(attack_selection.verification_lower_bound) << '\n';
            *progress_os << "  attack_validation_early_stopped_samples: "
                         << attack_selection.validation_early_stopped_samples << '\n';
            *progress_os << "  attack_validation_status: "
                         << attack_selection.validation_status << '\n';
            *progress_os << "  duplicate_attack: " << (duplicate_attack ? "true" : "false") << '\n';
            *progress_os << "  iteration_runtime_sec: "
                         << format_decimal(log.iteration_runtime_sec) << '\n';
            print_progress_line(
                *progress_os,
                "HIST-Iter iteration " + std::to_string(iteration),
                35.0 + 55.0 * static_cast<double>(iteration) /
                    static_cast<double>(options.iterations));
            algorithm_timer.resume_after_output();
        }
    }

    result.algorithm_runtime_sec = algorithm_timer.elapsed_seconds();
    return result;
}

}  // namespace

BaselineAlgorithm parse_baseline_algorithm(const std::string& value) {
    if (value == "fjc-iter") {
        return BaselineAlgorithm::FJCIter;
    }
    if (value == "fjc-linear-iter") {
        return BaselineAlgorithm::FJCLinearIter;
    }
    if (value == "hist-iter") {
        return BaselineAlgorithm::HISTIter;
    }
    if (value == "hist") {
        return BaselineAlgorithm::HIST;
    }
    if (value == "degree") {
        return BaselineAlgorithm::Degree;
    }
    if (value == "random") {
        return BaselineAlgorithm::Random;
    }
    if (value == "exact") {
        return BaselineAlgorithm::Exact;
    }
    throw std::invalid_argument(
        "Baseline algorithm must be one of fjc-iter, fjc-linear-iter, hist-iter, hist, degree, random, or exact");
}

std::string to_string(BaselineAlgorithm algorithm) {
    switch (algorithm) {
    case BaselineAlgorithm::FJCIter:
        return "fjc-iter";
    case BaselineAlgorithm::FJCLinearIter:
        return "fjc-linear-iter";
    case BaselineAlgorithm::HISTIter:
        return "hist-iter";
    case BaselineAlgorithm::HIST:
        return "hist";
    case BaselineAlgorithm::Degree:
        return "degree";
    case BaselineAlgorithm::Random:
        return "random";
    case BaselineAlgorithm::Exact:
        return "exact";
    }
    return "unknown";
}

std::string to_string(BaselineSamplingMode mode) {
    switch (mode) {
    case BaselineSamplingMode::Fixed:
        return "fixed";
    case BaselineSamplingMode::SubsimAdaptive:
        return "subsim-adaptive";
    case BaselineSamplingMode::ClassicFullRR:
        return "classic-full-rr";
    case BaselineSamplingMode::NotApplicable:
        return "not-applicable";
    }
    return "unknown";
}

BaselineResult run_baseline(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const BaselineOptions& options,
    std::ostream* progress_os) {
    if (options.defender_budget > graph.num_nodes()) {
        throw std::invalid_argument("Baseline defender budget exceeds number of nodes");
    }
    if (options.attacker_budget > graph.num_nodes()) {
        throw std::invalid_argument("Baseline attacker budget exceeds number of nodes");
    }
    if (options.algorithm == BaselineAlgorithm::FJCIter && options.fjc_pair_rr_samples == 0) {
        throw std::invalid_argument("Baseline requires at least one selection sample");
    }
    if (options.algorithm == BaselineAlgorithm::FJCLinearIter &&
        options.fjc_linear_rr_samples == 0) {
        throw std::invalid_argument("FJC-Linear-Iter requires at least one RR sample");
    }
    if (!std::isfinite(options.validation_failure_probability) ||
        options.validation_failure_probability <= 0.0 ||
        options.validation_failure_probability >= 1.0) {
        throw std::invalid_argument("Baseline failure probability must be in (0, 1)");
    }
    if ((options.algorithm == BaselineAlgorithm::HISTIter ||
         options.algorithm == BaselineAlgorithm::HIST) &&
        (options.epsilon <= 0.0 || options.epsilon >= 1.0)) {
        throw std::invalid_argument("HIST sampling requires epsilon in (0, 1)");
    }
    if (options.algorithm == BaselineAlgorithm::FJCIter &&
        options.max_rr_samples != 0 && options.max_rr_samples < options.fjc_pair_rr_samples) {
        throw std::invalid_argument(
            "Baseline max RR samples must be 0 or at least the base selection samples");
    }
    if (options.algorithm == BaselineAlgorithm::FJCLinearIter &&
        options.max_rr_samples != 0 && options.max_rr_samples < options.fjc_linear_rr_samples) {
        throw std::invalid_argument(
            "FJC-Linear-Iter max RR samples must be 0 or at least the linear RR sample count");
    }

    switch (options.algorithm) {
    case BaselineAlgorithm::HIST:
        return run_hist_baseline(graph, options, progress_os);
    case BaselineAlgorithm::Degree:
        return run_degree_baseline(graph, options, progress_os);
    case BaselineAlgorithm::FJCIter:
        return run_fjc_iter_baseline(graph, conflict, innate, params, options, progress_os);
    case BaselineAlgorithm::FJCLinearIter:
        return run_fjc_linear_iter_baseline(graph, conflict, innate, params, options, progress_os);
    case BaselineAlgorithm::HISTIter:
        return run_hist_iter_baseline(graph, options, progress_os);
    case BaselineAlgorithm::Random:
        return run_random_baseline(graph, options, progress_os);
    case BaselineAlgorithm::Exact:
        return run_exact_baseline(graph, conflict, innate, params, options, progress_os);
    }

    throw std::runtime_error("Unknown baseline algorithm");
}

}  // namespace rcd
