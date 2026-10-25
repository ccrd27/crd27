#include "rcd/defense_oracle.hpp"

#include "rcd/certificate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace rcd {
namespace {

void validate_node(std::size_t num_nodes, NodeId node) {
    if (static_cast<std::size_t>(node) >= num_nodes) {
        throw std::out_of_range("Seed node is out of range");
    }
}

void validate_problem_sizes(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate) {
    if (conflict.num_nodes() != graph.num_nodes() ||
        static_cast<std::size_t>(innate.size()) != graph.num_nodes()) {
        throw std::invalid_argument("Graph, conflict backend, and opinion vector sizes must match");
    }
}

void validate_certificate_options(const DefenseOracleOptions& options) {
    if (options.rr_samples == 0) {
        throw std::invalid_argument("Defense Oracle requires at least one RR sample");
    }
    if (options.max_rr_samples != 0 && options.max_rr_samples < options.rr_samples) {
        throw std::invalid_argument(
            "Defense Oracle max RR samples must be 0 or at least the base RR samples");
    }
    if (!std::isfinite(options.epsilon_rr) || options.epsilon_rr <= 0.0) {
        throw std::invalid_argument("Defense Oracle epsilon_rr must be finite and positive");
    }
    if (!std::isnan(options.oracle_epsilon) &&
        (!std::isfinite(options.oracle_epsilon) || options.oracle_epsilon <= 0.0)) {
        throw std::invalid_argument("Defense Oracle oracle_epsilon must be finite and positive");
    }
}

void validate_empirical_search_options(const DefenseOracleOptions& options) {
    if (options.empirical_samples == 0) {
        throw std::invalid_argument("Empirical defense search requires at least one simulation");
    }
}

void validate_mixed_distribution_options(const DefenseOracleOptions& options) {
    if (options.mixed_rounds == 0) {
        throw std::invalid_argument("Mixed distribution requires at least one round");
    }
    if (!std::isfinite(options.mixed_learning_rate) || options.mixed_learning_rate < 0.0) {
        throw std::invalid_argument("Mixed learning rate must be finite and nonnegative");
    }
    if (!std::isfinite(options.failure_probability) ||
        options.failure_probability <= 0.0 ||
        options.failure_probability >= 1.0) {
        throw std::invalid_argument("Defense failure probability must be in (0, 1)");
    }
}

std::vector<double> normalize_attack_distribution(
    std::size_t attack_pool_size,
    const std::vector<double>& attack_distribution) {
    if (attack_distribution.size() != attack_pool_size) {
        throw std::invalid_argument("Attack distribution size must match attack pool size");
    }

    double sum = 0.0;
    for (const double probability : attack_distribution) {
        if (!std::isfinite(probability)) {
            throw std::invalid_argument("Attack distribution probabilities must be finite");
        }
        if (probability < 0.0) {
            throw std::invalid_argument("Attack distribution probabilities must be nonnegative");
        }
        sum += probability;
    }
    if (sum <= 0.0) {
        throw std::invalid_argument("Attack distribution must have positive total mass");
    }

    std::vector<double> normalized = attack_distribution;
    for (double& probability : normalized) {
        probability /= sum;
    }
    return normalized;
}

std::vector<double> make_uniform_attack_distribution(std::size_t attack_pool_size) {
    if (attack_pool_size == 0) {
        return {};
    }
    return std::vector<double>(
        attack_pool_size,
        1.0 / static_cast<double>(attack_pool_size));
}

std::vector<double> normalized_scores_unit_interval(const std::vector<double>& scores) {
    if (scores.empty()) {
        return {};
    }

    const auto [min_it, max_it] = std::minmax_element(scores.begin(), scores.end());
    const double min_score = *min_it;
    const double max_score = *max_it;
    if (max_score <= min_score) {
        return std::vector<double>(scores.size(), 0.0);
    }

    std::vector<double> normalized;
    normalized.reserve(scores.size());
    for (const double score : scores) {
        normalized.push_back((score - min_score) / (max_score - min_score));
    }
    return normalized;
}

std::vector<double> update_attack_distribution(
    const std::vector<double>& current_distribution,
    const std::vector<double>& normalized_scores,
    double learning_rate) {
    if (current_distribution.size() != normalized_scores.size()) {
        throw std::invalid_argument("Distribution and score vector sizes must match");
    }

    std::vector<double> updated;
    updated.reserve(current_distribution.size());
    for (std::size_t i = 0; i < current_distribution.size(); ++i) {
        updated.push_back(current_distribution[i] * std::exp(learning_rate * normalized_scores[i]));
    }
    return normalize_attack_distribution(updated.size(), updated);
}

std::vector<std::uint8_t> make_reached_mask(
    std::size_t num_nodes,
    const std::vector<NodeId>& reached) {
    std::vector<std::uint8_t> mask(num_nodes, 0);
    for (const auto node : reached) {
        validate_node(num_nodes, node);
        mask[node] = 1;
    }
    return mask;
}

std::vector<std::uint8_t> make_defense_candidate_mask(
    std::size_t num_nodes,
    const std::vector<NodeId>& fixed_attack) {
    std::vector<std::uint8_t> candidate_mask(num_nodes, 1);
    for (const auto seed : fixed_attack) {
        validate_node(num_nodes, seed);
        candidate_mask[seed] = 0;
    }
    return candidate_mask;
}

struct MixedLowerSampleBatch {
    std::vector<std::size_t> sampled_attack_indices;
    std::vector<WeightedRRSample> samples;
    double constant_sum = 0.0;
};

struct MixedLowerSolveResult {
    std::vector<NodeId> greedy_defense;
    double estimated_constant = 0.0;
    double estimated_lower_surrogate_value = 0.0;
    double lower_certificate = std::numeric_limits<double>::quiet_NaN();
    double unclamped_lower_certificate = std::numeric_limits<double>::quiet_NaN();
    double raw_lower_certificate = std::numeric_limits<double>::quiet_NaN();
    double conservative_constant = std::numeric_limits<double>::quiet_NaN();
    double conservative_lower_surrogate_value = std::numeric_limits<double>::quiet_NaN();
};

struct FailureAllocation {
    double oracle_delta = std::numeric_limits<double>::quiet_NaN();
    double distribution_delta = std::numeric_limits<double>::quiet_NaN();
    double value_delta = std::numeric_limits<double>::quiet_NaN();
    double constant_delta = std::numeric_limits<double>::quiet_NaN();
    double surrogate_delta = std::numeric_limits<double>::quiet_NaN();
};

struct PackedSeedMask {
    std::size_t num_nodes = 0;
    std::vector<std::uint64_t> words;
};

PackedSeedMask make_seed_mask(
    std::size_t num_nodes,
    const std::vector<NodeId>& seeds) {
    PackedSeedMask mask;
    mask.num_nodes = num_nodes;
    mask.words.assign((num_nodes + 63U) / 64U, 0);
    for (const auto seed : seeds) {
        validate_node(num_nodes, seed);
        const auto index = static_cast<std::size_t>(seed);
        mask.words[index / 64U] |= std::uint64_t{1} << (index % 64U);
    }
    return mask;
}

bool rr_set_intersects_mask(
    const std::vector<NodeId>& rr_set,
    const PackedSeedMask& seed_mask) {
    for (const auto node : rr_set) {
        validate_node(seed_mask.num_nodes, node);
        const auto index = static_cast<std::size_t>(node);
        if ((seed_mask.words[index / 64U] &
             (std::uint64_t{1} << (index % 64U))) != 0) {
            return true;
        }
    }
    return false;
}

double rho_value(const CascadeParams& params, NodeId node, std::size_t expected_size) {
    if (params.rho_u_by_node.size() == 0) {
        return params.default_rho_u;
    }
    if (static_cast<std::size_t>(params.rho_u_by_node.size()) != expected_size) {
        throw std::invalid_argument("rho_u_by_node size does not match opinion vector size");
    }
    return params.rho_u_by_node[static_cast<Eigen::Index>(node)];
}

double pair_rr_perturbed_opinion_at(
    const Eigen::VectorXd& innate,
    NodeId node,
    bool attack_reached,
    bool defense_reached,
    const CascadeParams& params) {
    if (static_cast<Eigen::Index>(node) >= innate.size()) {
        throw std::out_of_range("Node id is out of range");
    }
    const double s = innate[static_cast<Eigen::Index>(node)];
    const double r_a = attack_reached ? 1.0 : 0.0;
    const double r_b = defense_reached ? 1.0 : 0.0;
    const double rho = rho_value(params, node, static_cast<std::size_t>(innate.size()));
    return clip_unit_interval(
        s +
        r_a * (1.0 - rho * r_b) * params.alpha_A * (s - 0.5) +
        r_b * params.alpha_B * (0.5 - s));
}

struct PairRREmpiricalSample {
    NodeId first_root = 0;
    NodeId second_root = 0;
    std::vector<NodeId> first_rr_set;
    std::vector<NodeId> second_rr_set;
    std::vector<std::uint8_t> first_attack_reached;
    std::vector<std::uint8_t> second_attack_reached;
    double coefficient = 0.0;
};

struct PairRREmpiricalPoolSamples {
    std::vector<PairRREmpiricalSample> samples;
    std::size_t attack_count = 0;
};

template <typename SampleConsumer>
void for_each_pair_rr_empirical_sample(
    PropagationModel propagation_model,
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const std::vector<AttackPoolEntry>& attack_pool,
    std::size_t sample_count,
    std::uint64_t seed,
    const std::string& progress_stage,
    const ProgressCallback& progress_callback,
    bool cache_coefficients,
    SampleConsumer&& sample_consumer) {
    if (graph.num_nodes() == 0) {
        throw std::invalid_argument("Pair-RR empirical evaluator requires a nonempty graph");
    }
    if (attack_pool.empty()) {
        throw std::invalid_argument("Pair-RR empirical evaluator requires a nonempty attack pool");
    }
    if (sample_count == 0) {
        throw std::invalid_argument("Pair-RR empirical evaluator requires at least one sample");
    }

    std::vector<PackedSeedMask> attack_masks;
    attack_masks.reserve(attack_pool.size());
    for (const auto& attack_entry : attack_pool) {
        const auto attack = normalized_seed_set(graph.num_nodes(), attack_entry.attack_seeds);
        attack_masks.push_back(make_seed_mask(graph.num_nodes(), attack));
    }

    if (progress_callback) {
        progress_callback(
            progress_stage + " started (total_samples=" +
                std::to_string(sample_count) + ")",
            0.0);
    }
    const double n_squared =
        static_cast<double>(graph.num_nodes()) * static_cast<double>(graph.num_nodes());
    std::unordered_map<std::uint64_t, double> coefficient_cache;
    if (cache_coefficients) {
        coefficient_cache.reserve(sample_count);
    }
    const auto* dense_conflict = dynamic_cast<const DenseConflict*>(&conflict);
    Eigen::VectorXd basis = dense_conflict == nullptr
        ? Eigen::VectorXd::Zero(static_cast<Eigen::Index>(graph.num_nodes()))
        : Eigen::VectorXd{};

    auto pair_coefficient = [&](NodeId first, NodeId second) {
        const NodeId low = std::min(first, second);
        const NodeId high = std::max(first, second);
        if (dense_conflict != nullptr) {
            return dense_conflict->conflict_matrix()(low, high);
        }

        const std::uint64_t key =
            (static_cast<std::uint64_t>(low) << 32U) | static_cast<std::uint64_t>(high);
        if (cache_coefficients) {
            const auto cached = coefficient_cache.find(key);
            if (cached != coefficient_cache.end()) {
                return cached->second;
            }
        }

        basis[static_cast<Eigen::Index>(high)] = 1.0;
        const double coefficient = conflict.dot_row(low, basis);
        basis[static_cast<Eigen::Index>(high)] = 0.0;
        if (cache_coefficients) {
            coefficient_cache.emplace(key, coefficient);
        }
        return coefficient;
    };

    std::mt19937_64 root_rng(seed ^ 0x6a09e667f3bcc909ULL);
    std::uniform_int_distribution<std::size_t> root_dist(0, graph.num_nodes() - 1);
    LazyLiveEdgeOracle live_oracle(graph.num_edges(), seed ^ 0xbb67ae8584caa73bULL);
    LazyTriggeringEdgeOracle triggering_oracle(graph.num_nodes(), seed ^ 0x3c6ef372fe94f82bULL);
    TraversalWorkspace first_rr_workspace(graph.num_nodes());
    TraversalWorkspace second_rr_workspace(graph.num_nodes());
    const std::size_t progress_interval = std::max<std::size_t>(1, sample_count / 10);

    for (std::size_t i = 0; i < sample_count; ++i) {
        PairRREmpiricalSample sample;
        sample.first_root = static_cast<NodeId>(root_dist(root_rng));
        sample.second_root = static_cast<NodeId>(root_dist(root_rng));
        sample.first_rr_set = reverse_rr_set(
            propagation_model,
            graph,
            sample.first_root,
            live_oracle,
            triggering_oracle,
            first_rr_workspace);
        sample.second_rr_set = reverse_rr_set(
            propagation_model,
            graph,
            sample.second_root,
            live_oracle,
            triggering_oracle,
            second_rr_workspace);
        sample.first_attack_reached.resize(attack_pool.size(), 0);
        sample.second_attack_reached.resize(attack_pool.size(), 0);
        for (std::size_t attack_index = 0; attack_index < attack_pool.size(); ++attack_index) {
            sample.first_attack_reached[attack_index] =
                rr_set_intersects_mask(sample.first_rr_set, attack_masks[attack_index]) ? 1 : 0;
            sample.second_attack_reached[attack_index] =
                rr_set_intersects_mask(sample.second_rr_set, attack_masks[attack_index]) ? 1 : 0;
        }
        sample.coefficient = n_squared * pair_coefficient(
            sample.first_root, sample.second_root);
        sample_consumer(sample);
        live_oracle.reset();
        triggering_oracle.reset();
        if (progress_callback) {
            if ((i + 1) % progress_interval == 0 || i + 1 == sample_count) {
                progress_callback(
                    progress_stage,
                    100.0 * static_cast<double>(i + 1) /
                        static_cast<double>(sample_count));
            }
        }
    }
}

PairRREmpiricalPoolSamples build_pair_rr_empirical_pool_samples(
    PropagationModel propagation_model,
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const std::vector<AttackPoolEntry>& attack_pool,
    std::size_t sample_count,
    std::uint64_t seed,
    const std::string& progress_stage,
    const ProgressCallback& progress_callback = {}) {
    PairRREmpiricalPoolSamples pool_samples;
    pool_samples.attack_count = attack_pool.size();
    pool_samples.samples.reserve(sample_count);
    for_each_pair_rr_empirical_sample(
        propagation_model,
        graph,
        conflict,
        attack_pool,
        sample_count,
        seed,
        progress_stage,
        progress_callback,
        true,
        [&pool_samples](PairRREmpiricalSample& sample) {
            pool_samples.samples.push_back(std::move(sample));
        });
    return pool_samples;
}

MonteCarloResult evaluate_pair_rr_attack_samples(
    const PairRREmpiricalPoolSamples& pool_samples,
    std::size_t attack_index,
    const Eigen::VectorXd& innate,
    const PackedSeedMask& defense_mask,
    const CascadeParams& params) {
    if (pool_samples.samples.empty()) {
        throw std::invalid_argument("Pair-RR empirical pool samples must be nonempty");
    }
    if (attack_index >= pool_samples.attack_count) {
        throw std::out_of_range("Pair-RR empirical attack index is out of range");
    }

    double mean = 0.0;
    double m2 = 0.0;
    for (std::size_t i = 0; i < pool_samples.samples.size(); ++i) {
        const auto& sample = pool_samples.samples[i];
        if (sample.first_attack_reached.size() != pool_samples.attack_count ||
            sample.second_attack_reached.size() != pool_samples.attack_count) {
            throw std::invalid_argument("Pair-RR empirical sample attack reachability size mismatch");
        }
        const bool first_defense_reached =
            rr_set_intersects_mask(sample.first_rr_set, defense_mask);
        const bool second_defense_reached =
            rr_set_intersects_mask(sample.second_rr_set, defense_mask);
        const double first_opinion = pair_rr_perturbed_opinion_at(
            innate,
            sample.first_root,
            sample.first_attack_reached[attack_index] != 0,
            first_defense_reached,
            params);
        const double second_opinion = pair_rr_perturbed_opinion_at(
            innate,
            sample.second_root,
            sample.second_attack_reached[attack_index] != 0,
            second_defense_reached,
            params);
        const double value = sample.coefficient * first_opinion * second_opinion;
        const double count = static_cast<double>(i + 1);
        const double delta = value - mean;
        mean += delta / count;
        const double delta2 = value - mean;
        m2 += delta * delta2;
    }

    MonteCarloResult result;
    result.mean = mean;
    result.simulations = pool_samples.samples.size();
    if (pool_samples.samples.size() > 1) {
        result.variance = m2 / static_cast<double>(pool_samples.samples.size() - 1);
        result.standard_error =
            std::sqrt(result.variance / static_cast<double>(pool_samples.samples.size()));
    }
    return result;
}

EmpiricalPoolEvaluationResult evaluate_pair_rr_empirical_pool_samples(
    const PairRREmpiricalPoolSamples& pool_samples,
    std::size_t num_nodes,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params) {
    if (pool_samples.samples.empty() || pool_samples.attack_count == 0) {
        throw std::invalid_argument("Pair-RR empirical pool samples must be nonempty");
    }
    const auto defense = normalized_seed_set(num_nodes, defense_seeds);
    const auto defense_mask = make_seed_mask(num_nodes, defense);

    EmpiricalPoolEvaluationResult result;
    result.pool_value = -std::numeric_limits<double>::infinity();
    result.attack_results.reserve(pool_samples.attack_count);
    for (std::size_t attack_index = 0; attack_index < pool_samples.attack_count; ++attack_index) {
        const auto attack_result = evaluate_pair_rr_attack_samples(
            pool_samples,
            attack_index,
            innate,
            defense_mask,
            params);
        result.attack_results.push_back(attack_result);
        if (attack_result.mean > result.pool_value) {
            result.pool_value = attack_result.mean;
            result.worst_attack_index = attack_index;
            result.worst_attack_result = attack_result;
        }
    }
    return result;
}

struct RunningMoments {
    std::size_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double value) {
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = value - mean;
        m2 += delta * delta2;
    }

    MonteCarloResult finish() const {
        MonteCarloResult result;
        result.mean = mean;
        result.simulations = count;
        if (count > 1) {
            result.variance = m2 / static_cast<double>(count - 1);
            result.standard_error =
                std::sqrt(result.variance / static_cast<double>(count));
            result.ci95_half_width = 1.96 * result.standard_error;
        }
        return result;
    }
};

EmpiricalDefenseSearchResult run_empirical_greedy_defense_search_pair_rr_from_samples(
    std::size_t num_nodes,
    const Eigen::VectorXd& innate,
    const PairRREmpiricalPoolSamples& pool_samples,
    const CascadeParams& params,
    const DefenseOracleOptions& options) {
    EmpiricalDefenseSearchResult result;
    std::vector<std::uint8_t> selected_mask(num_nodes, 0);
    result.defense_seeds.reserve(std::min(options.budget, num_nodes));
    result.round_best_values.reserve(std::min(options.budget, num_nodes));

    for (std::size_t round = 0; round < options.budget; ++round) {
        bool found_candidate = false;
        NodeId best_node = 0;
        double best_value = std::numeric_limits<double>::infinity();

        for (NodeId node = 0; static_cast<std::size_t>(node) < num_nodes; ++node) {
            if (selected_mask[node] != 0) {
                continue;
            }

            std::vector<NodeId> candidate = result.defense_seeds;
            candidate.push_back(node);
            const auto evaluation = evaluate_pair_rr_empirical_pool_samples(
                pool_samples, num_nodes, innate, candidate, params);
            if (!found_candidate ||
                evaluation.pool_value < best_value ||
                (evaluation.pool_value == best_value && node < best_node)) {
                found_candidate = true;
                best_node = node;
                best_value = evaluation.pool_value;
            }
        }

        if (!found_candidate) {
            break;
        }

        selected_mask[best_node] = 1;
        result.defense_seeds.push_back(best_node);
        result.round_best_values.push_back(best_value);
        if (options.progress_callback) {
            const std::size_t selected_count = result.defense_seeds.size();
            const std::size_t progress_interval =
                std::max<std::size_t>(1, options.budget / 10);
            if (selected_count % progress_interval == 0 ||
                selected_count == options.budget) {
                options.progress_callback(
                    "Defense Oracle: empirical pair-RR greedy selection",
                    100.0 * static_cast<double>(selected_count) /
                        static_cast<double>(std::max<std::size_t>(options.budget, 1)));
            }
        }
    }

    std::sort(result.defense_seeds.begin(), result.defense_seeds.end());
    result.defense_seeds.erase(
        std::unique(result.defense_seeds.begin(), result.defense_seeds.end()),
        result.defense_seeds.end());
    const auto final_evaluation = evaluate_pair_rr_empirical_pool_samples(
        pool_samples, num_nodes, innate, result.defense_seeds, params);
    result.empirical_pool_value = final_evaluation.pool_value;
    return result;
}

FailureAllocation fixed_failure_allocation(double failure_probability) {
    FailureAllocation allocation;
    allocation.oracle_delta = failure_probability;
    allocation.distribution_delta = failure_probability;
    allocation.value_delta = failure_probability;
    allocation.constant_delta = allocation.value_delta / 2.0;
    allocation.surrogate_delta = allocation.value_delta / 2.0;
    return allocation;
}

double defense_oracle_epsilon(const DefenseOracleOptions& options) {
    if (!std::isnan(options.oracle_epsilon)) {
        return options.oracle_epsilon;
    }
    return options.epsilon_rr;
}

std::vector<std::vector<NodeId>> normalize_attack_pool(
    std::size_t num_nodes,
    const std::vector<AttackPoolEntry>& attack_pool) {
    std::vector<std::vector<NodeId>> normalized_pool;
    normalized_pool.reserve(attack_pool.size());
    for (const auto& entry : attack_pool) {
        normalized_pool.push_back(normalized_seed_set(num_nodes, entry.attack_seeds));
    }
    return normalized_pool;
}

MixedLowerSampleBatch generate_mixed_lower_rr_samples_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<std::vector<NodeId>>& normalized_pool,
    const std::vector<double>& attack_distribution,
    const CascadeParams& params,
    PropagationModel propagation_model,
    std::size_t sample_count,
    std::uint64_t seed,
    const ProgressCallback& progress_callback = {}) {
    if (normalized_pool.empty()) {
        throw std::invalid_argument("Mixed lower sample generation requires a nonempty attack pool");
    }
    if (sample_count == 0) {
        throw std::invalid_argument("Mixed lower sample generation requires at least one sample");
    }

    std::mt19937_64 attack_rng(seed ^ 0xa4093822299f31d0ULL);
    std::discrete_distribution<std::size_t> attack_dist(
        attack_distribution.begin(), attack_distribution.end());
    std::mt19937_64 root_rng(seed ^ 0x082efa98ec4e6c89ULL);
    std::uniform_int_distribution<std::size_t> root_dist(0, graph.num_nodes() - 1);
    LazyLiveEdgeOracle live_oracle(graph.num_edges(), seed ^ 0x452821e638d01377ULL);
    LazyTriggeringEdgeOracle triggering_oracle(graph.num_nodes(), seed ^ 0x8c3d37c819544da2ULL);
    TraversalWorkspace attack_workspace(graph.num_nodes());
    TraversalWorkspace rr_workspace(graph.num_nodes());
    const std::size_t progress_interval = std::max<std::size_t>(1, sample_count / 10);

    MixedLowerSampleBatch batch;
    batch.sampled_attack_indices.reserve(sample_count);
    batch.samples.reserve(sample_count);
    if (progress_callback) {
        progress_callback(
            "Defense Oracle: lower-certificate RR sampling started (total_samples=" +
                std::to_string(sample_count) + ")",
            0.0);
    }

    for (std::size_t i = 0; i < sample_count; ++i) {
        const std::size_t attack_index = attack_dist(attack_rng);
        const auto& attack = normalized_pool[attack_index];

        const std::vector<NodeId> attack_reached = forward_reachable(
            propagation_model,
            graph,
            attack,
            live_oracle,
            triggering_oracle,
            attack_workspace);
        const std::vector<std::uint8_t> attack_reached_mask =
            make_reached_mask(graph.num_nodes(), attack_reached);

        const auto terms = build_conditional_terms(
            conflict, innate, attack_reached_mask, params, OptimizedPlayer::Defender);
        const Eigen::VectorXd lower_weights = lower_surrogate_weights(terms);
        batch.constant_sum += terms.constant;

        const NodeId root = static_cast<NodeId>(root_dist(root_rng));
        std::vector<NodeId> rr_set = reverse_rr_set(
            propagation_model,
            graph,
            root,
            live_oracle,
            triggering_oracle,
            rr_workspace);
        std::sort(rr_set.begin(), rr_set.end());
        const Weight weight = static_cast<Weight>(graph.num_nodes()) * lower_weights[root];
        batch.sampled_attack_indices.push_back(attack_index);
        batch.samples.push_back(WeightedRRSample{std::move(rr_set), root, weight});

        live_oracle.reset();
        triggering_oracle.reset();
        if (progress_callback &&
            ((i + 1) % progress_interval == 0 || i + 1 == sample_count)) {
            progress_callback(
                "Defense Oracle: RR sampling",
                100.0 * static_cast<double>(i + 1) /
                    static_cast<double>(std::max<std::size_t>(sample_count, 1)));
        }
    }

    return batch;
}

MixedLowerSolveResult solve_mixed_lower_from_samples(
    std::size_t num_nodes,
    const ConflictBackend& conflict,
    const MixedLowerSampleBatch& batch,
    const DefenseOracleOptions& options,
    const FailureAllocation& allocation) {
    if (batch.samples.empty()) {
        throw std::invalid_argument("Mixed lower solve requires at least one sample");
    }

    MixedLowerSolveResult result;
    result.greedy_defense = weighted_coverage_greedy(
        num_nodes, batch.samples, options.budget);
    std::sort(result.greedy_defense.begin(), result.greedy_defense.end());
    result.greedy_defense.erase(
        std::unique(result.greedy_defense.begin(), result.greedy_defense.end()),
        result.greedy_defense.end());

    result.estimated_constant =
        batch.constant_sum / static_cast<double>(batch.samples.size());
    result.estimated_lower_surrogate_value = weighted_coverage_value(
        batch.samples, result.greedy_defense);
    const double oracle_epsilon = defense_oracle_epsilon(options);
    const double factor = greedy_coverage_factor();
    result.raw_lower_certificate =
        result.estimated_constant -
        (result.estimated_lower_surrogate_value + oracle_epsilon) / factor;
    result.conservative_constant = lower_confidence_bound(
        result.estimated_constant,
        quadratic_constant_sample_bound(num_nodes),
        batch.samples.size(),
        allocation.constant_delta);
    result.conservative_lower_surrogate_value = upper_confidence_bound(
        result.estimated_lower_surrogate_value,
        defender_rr_sample_bound(conflict.num_nodes(), conflict_beta(conflict)),
        batch.samples.size(),
        allocation.surrogate_delta);
    result.unclamped_lower_certificate =
        result.conservative_constant -
        (result.conservative_lower_surrogate_value + oracle_epsilon) / factor;
    result.lower_certificate = std::max(0.0, result.unclamped_lower_certificate);
    return result;
}

MixedLowerCertificate make_mixed_lower_certificate_from_solve(
    const std::vector<double>& normalized_distribution,
    MixedLowerSampleBatch&& batch,
    MixedLowerSolveResult&& solve_result,
    const DefenseOracleOptions& options,
    const FailureAllocation& allocation) {
    MixedLowerCertificate certificate;
    certificate.distribution_type = "custom";
    certificate.attack_distribution = normalized_distribution;
    certificate.failure_probability = options.failure_probability;
    certificate.sampled_attack_indices = std::move(batch.sampled_attack_indices);
    certificate.samples = std::move(batch.samples);
    certificate.greedy_defense = std::move(solve_result.greedy_defense);
    certificate.estimated_constant = solve_result.estimated_constant;
    certificate.estimated_lower_surrogate_value =
        solve_result.estimated_lower_surrogate_value;
    certificate.lower_certificate = solve_result.lower_certificate;
    certificate.unclamped_lower_certificate = solve_result.unclamped_lower_certificate;
    certificate.raw_lower_certificate = solve_result.raw_lower_certificate;
    certificate.conservative_constant = solve_result.conservative_constant;
    certificate.conservative_lower_surrogate_value =
        solve_result.conservative_lower_surrogate_value;
    certificate.oracle_epsilon = defense_oracle_epsilon(options);
    certificate.certificate_formula = "additive_oracle_error";
    certificate.oracle_failure_probability = allocation.oracle_delta;
    certificate.distribution_failure_probability = allocation.distribution_delta;
    certificate.value_failure_probability = allocation.value_delta;
    certificate.constant_failure_probability = allocation.constant_delta;
    certificate.surrogate_failure_probability = allocation.surrogate_delta;
    return certificate;
}

MixedLowerCertificate compute_fixed_mixed_lower_certificate_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<std::vector<NodeId>>& normalized_pool,
    const std::vector<double>& normalized_distribution,
    const CascadeParams& params,
    const DefenseOracleOptions& options,
    double oracle_failure_probability,
    double distribution_failure_probability) {
    const auto allocation = fixed_failure_allocation(distribution_failure_probability);
    auto batch = generate_mixed_lower_rr_samples_ic(
        graph,
        conflict,
        innate,
        normalized_pool,
        normalized_distribution,
        params,
        options.propagation_model,
        options.rr_samples,
        options.seed,
        options.progress_callback);
    auto solve_result = solve_mixed_lower_from_samples(
        graph.num_nodes(), conflict, batch, options, allocation);
    auto certificate = make_mixed_lower_certificate_from_solve(
        normalized_distribution, std::move(batch), std::move(solve_result), options, allocation);
    certificate.oracle_failure_probability = oracle_failure_probability;
    certificate.distribution_failure_probability = distribution_failure_probability;
    certificate.required_rr_samples = options.rr_samples;
    certificate.rr_sample_cap = options.max_rr_samples;
    certificate.rr_sample_cap_triggered = false;
    certificate.generated_rr_samples = options.rr_samples;
    certificate.certificate_status = "fixed_sample_estimate";
    return certificate;
}

}  // namespace

std::string to_string(DefenseOracleStatus status) {
    switch (status) {
        case DefenseOracleStatus::SearchCompleted:
            return "SearchCompleted";
        case DefenseOracleStatus::InvalidInput:
            return "InvalidInput";
    }
    return "Unknown";
}

std::string to_string(DefenseCandidateSource source) {
    switch (source) {
        case DefenseCandidateSource::None:
            return "None";
        case DefenseCandidateSource::Empirical:
            return "Empirical";
        case DefenseCandidateSource::Surrogate:
            return "Surrogate";
    }
    return "Unknown";
}

std::string to_string(MixedDistributionMode mode) {
    switch (mode) {
        case MixedDistributionMode::Uniform:
            return "uniform";
        case MixedDistributionMode::Adaptive:
            return "adaptive";
    }
    return "unknown";
}

MixedDistributionMode parse_mixed_distribution_mode(const std::string& value) {
    if (value == "uniform") {
        return MixedDistributionMode::Uniform;
    }
    if (value == "adaptive") {
        return MixedDistributionMode::Adaptive;
    }
    throw std::invalid_argument("Mixed distribution must be either uniform or adaptive");
}

std::string to_string(DefenseOracleMode mode) {
    switch (mode) {
        case DefenseOracleMode::Engineering:
            return "engineering";
        case DefenseOracleMode::PaperExact:
            return "paper-exact";
    }
    return "unknown";
}

DefenseOracleMode parse_defense_oracle_mode(const std::string& value) {
    if (value == "engineering") {
        return DefenseOracleMode::Engineering;
    }
    if (value == "paper-exact") {
        return DefenseOracleMode::PaperExact;
    }
    throw std::invalid_argument("Defense Oracle mode must be either engineering or paper-exact");
}

std::vector<NodeId> normalized_seed_set(
    std::size_t num_nodes,
    const std::vector<NodeId>& seeds) {
    std::vector<NodeId> normalized = seeds;
    for (const auto seed : normalized) {
        validate_node(num_nodes, seed);
    }
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    return normalized;
}

std::vector<NodeId> residual_attack_set(
    std::size_t num_nodes,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds) {
    const std::vector<NodeId> attack = normalized_seed_set(num_nodes, attack_seeds);
    const std::vector<NodeId> defense = normalized_seed_set(num_nodes, defense_seeds);

    std::vector<NodeId> residual;
    residual.reserve(attack.size());
    std::set_difference(
        attack.begin(),
        attack.end(),
        defense.begin(),
        defense.end(),
        std::back_inserter(residual));
    return residual;
}

FixedAttackLowerCertificate compute_fixed_attack_lower_certificate_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& fixed_attack,
    const CascadeParams& params,
    const DefenseOracleOptions& options) {
    validate_problem_sizes(graph, conflict, innate);
    validate_certificate_options(options);

    const std::vector<NodeId> attack = normalized_seed_set(graph.num_nodes(), fixed_attack);
    const auto candidate_mask = make_defense_candidate_mask(graph.num_nodes(), attack);

    std::mt19937_64 root_rng(options.seed ^ 0x243f6a8885a308d3ULL);
    std::uniform_int_distribution<std::size_t> root_dist(0, graph.num_nodes() - 1);
    LazyLiveEdgeOracle live_oracle(graph.num_edges(), options.seed ^ 0x13198a2e03707344ULL);
    LazyTriggeringEdgeOracle triggering_oracle(graph.num_nodes(), options.seed ^ 0x243f6a8885a308d3ULL);
    TraversalWorkspace attack_workspace(graph.num_nodes());
    TraversalWorkspace rr_workspace(graph.num_nodes());

    std::vector<WeightedRRSample> samples;
    samples.reserve(options.rr_samples);
    double k_sum = 0.0;

    for (std::size_t i = 0; i < options.rr_samples; ++i) {
        const std::vector<NodeId> attack_reached = forward_reachable(
            options.propagation_model,
            graph,
            attack,
            live_oracle,
            triggering_oracle,
            attack_workspace);
        const std::vector<std::uint8_t> attack_reached_mask =
            make_reached_mask(graph.num_nodes(), attack_reached);

        const auto terms = build_conditional_terms(
            conflict, innate, attack_reached_mask, params, OptimizedPlayer::Defender);
        const Eigen::VectorXd lower_weights = lower_surrogate_weights(terms);
        k_sum += terms.constant;

        const NodeId root = static_cast<NodeId>(root_dist(root_rng));
        std::vector<NodeId> rr_set = reverse_rr_set(
            options.propagation_model,
            graph,
            root,
            live_oracle,
            triggering_oracle,
            rr_workspace);
        std::sort(rr_set.begin(), rr_set.end());
        const Weight weight = static_cast<Weight>(graph.num_nodes()) * lower_weights[root];
        samples.push_back(WeightedRRSample{std::move(rr_set), root, weight});

        live_oracle.reset();
        triggering_oracle.reset();
    }

    auto greedy_defense = weighted_coverage_greedy(
        graph.num_nodes(), samples, options.budget, candidate_mask,
        options.progress_callback);
    std::sort(greedy_defense.begin(), greedy_defense.end());
    greedy_defense.erase(std::unique(greedy_defense.begin(), greedy_defense.end()), greedy_defense.end());

    const double estimated_k = k_sum / static_cast<double>(options.rr_samples);
    const double estimated_surrogate = weighted_coverage_value(samples, greedy_defense);
    const double factor = greedy_coverage_factor();
    const double oracle_epsilon = defense_oracle_epsilon(options);

    FixedAttackLowerCertificate certificate;
    certificate.residual_attack = attack;
    certificate.greedy_defense = std::move(greedy_defense);
    certificate.estimated_k = estimated_k;
    certificate.estimated_lower_surrogate_value = estimated_surrogate;
    certificate.unclamped_lower_certificate =
        estimated_k - (estimated_surrogate + oracle_epsilon) / factor;
    certificate.lower_certificate = std::max(0.0, certificate.unclamped_lower_certificate);
    certificate.samples = std::move(samples);
    return certificate;
}

MixedLowerCertificate compute_mixed_lower_certificate_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const CascadeParams& params,
    const DefenseOracleOptions& options) {
    auto certificate = compute_mixed_lower_certificate_ic(
        graph,
        conflict,
        innate,
        attack_pool,
        make_uniform_attack_distribution(attack_pool.size()),
        params,
        options);
    certificate.distribution_type = "uniform";
    return certificate;
}

MixedLowerCertificate compute_mixed_lower_certificate_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<double>& attack_distribution,
    const CascadeParams& params,
    const DefenseOracleOptions& options) {
    validate_problem_sizes(graph, conflict, innate);
    validate_certificate_options(options);
    validate_mixed_distribution_options(options);
    if (attack_pool.empty()) {
        throw std::invalid_argument("Mixed lower certificate requires a nonempty attack pool");
    }
    if (graph.num_nodes() == 0) {
        throw std::invalid_argument("Mixed lower certificate requires a nonempty graph");
    }
    const std::vector<double> normalized_distribution =
        normalize_attack_distribution(attack_pool.size(), attack_distribution);

    const auto normalized_pool = normalize_attack_pool(graph.num_nodes(), attack_pool);
    return compute_fixed_mixed_lower_certificate_ic(
        graph,
        conflict,
        innate,
        normalized_pool,
        normalized_distribution,
        params,
        options,
        options.failure_probability,
        options.failure_probability);
}

std::vector<double> estimate_fixed_attack_scores_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const DefenseOracleOptions& options,
    std::size_t round) {
    std::vector<double> scores;
    scores.reserve(attack_pool.size());
    const std::vector<NodeId> defense = normalized_seed_set(graph.num_nodes(), defense_seeds);

    for (std::size_t attack_index = 0; attack_index < attack_pool.size(); ++attack_index) {
        const std::vector<NodeId> attack =
            normalized_seed_set(graph.num_nodes(), attack_pool[attack_index].attack_seeds);
        const MonteCarloOptions score_options{
            options.empirical_samples,
            options.seed +
                0x632be59bd9b4e019ULL * static_cast<std::uint64_t>(round + 1) +
                0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(attack_index + 1)
        };
        const auto score = estimate_expected_conflict(
            options.propagation_model,
            graph,
            conflict,
            innate,
            attack,
            defense,
            params,
            score_options);
        scores.push_back(score.mean);
    }
    return scores;
}

MixedLowerCertificate compute_adaptive_mixed_lower_certificate_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const CascadeParams& params,
    const DefenseOracleOptions& options) {
    validate_problem_sizes(graph, conflict, innate);
    validate_certificate_options(options);
    validate_empirical_search_options(options);
    validate_mixed_distribution_options(options);
    if (attack_pool.empty()) {
        throw std::invalid_argument("Adaptive mixed lower certificate requires a nonempty attack pool");
    }

    std::vector<double> distribution = make_uniform_attack_distribution(attack_pool.size());
    MixedLowerCertificate best_certificate;
    best_certificate.lower_certificate = -std::numeric_limits<double>::infinity();
    std::vector<double> round_lower_certificates;
    std::vector<std::vector<double>> round_distributions;
    std::vector<std::vector<NodeId>> round_greedy_defenses;
    round_lower_certificates.reserve(options.mixed_rounds);
    round_distributions.reserve(options.mixed_rounds);
    round_greedy_defenses.reserve(options.mixed_rounds);
    std::vector<double> final_distribution = distribution;

    for (std::size_t round = 0; round < options.mixed_rounds; ++round) {
        DefenseOracleOptions round_options = options;
        round_options.failure_probability =
            options.failure_probability /
            static_cast<double>(std::max<std::size_t>(1, options.mixed_rounds));
        round_options.seed =
            options.seed + 0x94d049bb133111ebULL * static_cast<std::uint64_t>(round + 1);
        auto round_certificate = compute_mixed_lower_certificate_ic(
            graph, conflict, innate, attack_pool, distribution, params, round_options);
        round_certificate.distribution_type = "adaptive";
        round_certificate.mixed_rounds = options.mixed_rounds;
        round_certificate.selected_round = round;
        round_certificate.failure_probability = options.failure_probability;
        round_certificate.oracle_failure_probability = options.failure_probability;
        round_certificate.distribution_failure_probability = round_options.failure_probability;

        round_distributions.push_back(distribution);
        round_lower_certificates.push_back(round_certificate.lower_certificate);
        round_greedy_defenses.push_back(round_certificate.greedy_defense);

        if (round == 0 || round_certificate.lower_certificate > best_certificate.lower_certificate) {
            best_certificate = round_certificate;
        }

        final_distribution = distribution;
        if (round + 1 < options.mixed_rounds) {
            const auto scores = estimate_fixed_attack_scores_ic(
                graph,
                conflict,
                innate,
                attack_pool,
                round_certificate.greedy_defense,
                params,
                options,
                round);
            distribution = update_attack_distribution(
                distribution,
                normalized_scores_unit_interval(scores),
                options.mixed_learning_rate);
            final_distribution = distribution;
        }
    }

    best_certificate.distribution_type = "adaptive";
    best_certificate.mixed_rounds = options.mixed_rounds;
    best_certificate.failure_probability = options.failure_probability;
    best_certificate.oracle_failure_probability = options.failure_probability;
    best_certificate.final_attack_distribution = final_distribution;
    best_certificate.round_lower_certificates = std::move(round_lower_certificates);
    best_certificate.round_attack_distributions = std::move(round_distributions);
    best_certificate.round_greedy_defenses = std::move(round_greedy_defenses);
    return best_certificate;
}

EmpiricalPoolEvaluationResult estimate_empirical_pool_value(
    PropagationModel propagation_model,
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options) {
    validate_problem_sizes(graph, conflict, innate);
    if (attack_pool.empty()) {
        throw std::invalid_argument("Empirical pool evaluation requires a nonempty attack pool");
    }
    if (options.simulations == 0) {
        throw std::invalid_argument("Empirical pool evaluation requires at least one simulation");
    }

    const std::vector<NodeId> defense = normalized_seed_set(graph.num_nodes(), defense_seeds);

    EmpiricalPoolEvaluationResult result;
    result.attack_results.reserve(attack_pool.size());
    result.pool_value = -std::numeric_limits<double>::infinity();

    for (std::size_t attack_index = 0; attack_index < attack_pool.size(); ++attack_index) {
        const std::vector<NodeId> attack =
            normalized_seed_set(graph.num_nodes(), attack_pool[attack_index].attack_seeds);
        const MonteCarloOptions attack_options{
            options.simulations,
            options.seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(attack_index + 1)
        };
        const auto attack_result = estimate_expected_conflict(
            propagation_model,
            graph,
            conflict,
            innate,
            attack,
            defense,
            params,
            attack_options);
        result.attack_results.push_back(attack_result);

        if (attack_index == 0 || attack_result.mean > result.pool_value) {
            result.pool_value = attack_result.mean;
            result.worst_attack_index = attack_index;
            result.worst_attack_result = attack_result;
        }
    }

    return result;
}

EmpiricalPoolEvaluationResult estimate_empirical_pool_value_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options) {
    return estimate_empirical_pool_value(
        PropagationModel::IC,
        graph,
        conflict,
        innate,
        attack_pool,
        defense_seeds,
        params,
        options);
}

PairRRStreamingEvaluationResult evaluate_empirical_pool_candidates_pair_rr_streaming(
    PropagationModel propagation_model,
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<std::vector<NodeId>>& defense_candidates,
    const CascadeParams& params,
    std::size_t sample_count,
    std::uint64_t seed,
    const std::string& progress_stage,
    const ProgressCallback& progress_callback) {
    validate_problem_sizes(graph, conflict, innate);
    if (attack_pool.empty()) {
        throw std::invalid_argument(
            "Streaming pair-RR evaluation requires a nonempty attack pool");
    }
    if (defense_candidates.empty()) {
        throw std::invalid_argument(
            "Streaming pair-RR evaluation requires at least one defense candidate");
    }
    if (sample_count == 0) {
        throw std::invalid_argument(
            "Streaming pair-RR evaluation requires at least one sample");
    }

    std::vector<PackedSeedMask> defense_masks;
    defense_masks.reserve(defense_candidates.size());
    for (const auto& candidate : defense_candidates) {
        defense_masks.push_back(make_seed_mask(
            graph.num_nodes(), normalized_seed_set(graph.num_nodes(), candidate)));
    }

    const std::size_t attack_count = attack_pool.size();
    std::vector<RunningMoments> moments(defense_candidates.size() * attack_count);
    std::vector<std::uint8_t> first_defense_reached(defense_candidates.size(), 0);
    std::vector<std::uint8_t> second_defense_reached(defense_candidates.size(), 0);

    for_each_pair_rr_empirical_sample(
        propagation_model,
        graph,
        conflict,
        attack_pool,
        sample_count,
        seed,
        progress_stage,
        progress_callback,
        false,
        [&](const PairRREmpiricalSample& sample) {
            if (sample.first_attack_reached.size() != attack_count ||
                sample.second_attack_reached.size() != attack_count) {
                throw std::invalid_argument(
                    "Streaming pair-RR sample attack reachability size mismatch");
            }

            for (std::size_t candidate_index = 0;
                 candidate_index < defense_masks.size();
                 ++candidate_index) {
                first_defense_reached[candidate_index] =
                    rr_set_intersects_mask(
                        sample.first_rr_set, defense_masks[candidate_index]) ? 1 : 0;
                second_defense_reached[candidate_index] =
                    rr_set_intersects_mask(
                        sample.second_rr_set, defense_masks[candidate_index]) ? 1 : 0;
            }

            for (std::size_t candidate_index = 0;
                 candidate_index < defense_candidates.size();
                 ++candidate_index) {
                for (std::size_t attack_index = 0;
                     attack_index < attack_count;
                     ++attack_index) {
                    const double first_opinion = pair_rr_perturbed_opinion_at(
                        innate,
                        sample.first_root,
                        sample.first_attack_reached[attack_index] != 0,
                        first_defense_reached[candidate_index] != 0,
                        params);
                    const double second_opinion = pair_rr_perturbed_opinion_at(
                        innate,
                        sample.second_root,
                        sample.second_attack_reached[attack_index] != 0,
                        second_defense_reached[candidate_index] != 0,
                        params);
                    moments[candidate_index * attack_count + attack_index].add(
                        sample.coefficient * first_opinion * second_opinion);
                }
            }
        });

    PairRRStreamingEvaluationResult result;
    result.generated_samples = sample_count;
    result.candidate_results.reserve(defense_candidates.size());
    for (std::size_t candidate_index = 0;
         candidate_index < defense_candidates.size();
         ++candidate_index) {
        EmpiricalPoolEvaluationResult candidate_result;
        candidate_result.pool_value = -std::numeric_limits<double>::infinity();
        candidate_result.attack_results.reserve(attack_count);
        for (std::size_t attack_index = 0; attack_index < attack_count; ++attack_index) {
            const auto attack_result =
                moments[candidate_index * attack_count + attack_index].finish();
            candidate_result.attack_results.push_back(attack_result);
            if (attack_index == 0 || attack_result.mean > candidate_result.pool_value) {
                candidate_result.pool_value = attack_result.mean;
                candidate_result.worst_attack_index = attack_index;
                candidate_result.worst_attack_result = attack_result;
            }
        }
        result.candidate_results.push_back(std::move(candidate_result));
    }
    return result;
}

EmpiricalPoolEvaluationResult estimate_empirical_pool_value_pair_rr_dense(
    PropagationModel propagation_model,
    const DirectedGraph& graph,
    const DenseConflict& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options) {
    validate_problem_sizes(graph, conflict, innate);
    if (attack_pool.empty()) {
        throw std::invalid_argument("Pair-RR empirical pool evaluation requires a nonempty attack pool");
    }
    if (options.simulations == 0) {
        throw std::invalid_argument("Pair-RR empirical pool evaluation requires at least one sample");
    }

    const auto streaming_result = evaluate_empirical_pool_candidates_pair_rr_streaming(
        propagation_model,
        graph,
        conflict,
        innate,
        attack_pool,
        {defense_seeds},
        params,
        options.simulations,
        options.seed,
        "Pair-RR empirical evaluation sampling");
    return streaming_result.candidate_results.front();
}

EmpiricalDefenseSearchResult run_empirical_greedy_defense_search_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const CascadeParams& params,
    const DefenseOracleOptions& options) {
    validate_problem_sizes(graph, conflict, innate);
    validate_empirical_search_options(options);
    if (attack_pool.empty()) {
        throw std::invalid_argument("Empirical defense search requires a nonempty attack pool");
    }

    EmpiricalDefenseSearchResult result;
    std::vector<std::uint8_t> selected_mask(graph.num_nodes(), 0);
    result.defense_seeds.reserve(std::min(options.budget, graph.num_nodes()));
    result.round_best_values.reserve(std::min(options.budget, graph.num_nodes()));

    for (std::size_t round = 0; round < options.budget; ++round) {
        bool found_candidate = false;
        NodeId best_node = 0;
        double best_value = std::numeric_limits<double>::infinity();
        const MonteCarloOptions round_options{
            options.empirical_samples,
            options.seed + 0x517cc1b727220a95ULL * static_cast<std::uint64_t>(round + 1)
        };

        for (NodeId node = 0; static_cast<std::size_t>(node) < graph.num_nodes(); ++node) {
            if (selected_mask[node] != 0) {
                continue;
            }

            std::vector<NodeId> candidate = result.defense_seeds;
            candidate.push_back(node);
            const auto evaluation = estimate_empirical_pool_value(
                options.propagation_model,
                graph, conflict, innate, attack_pool, candidate, params, round_options);

            if (!found_candidate ||
                evaluation.pool_value < best_value ||
                (evaluation.pool_value == best_value && node < best_node)) {
                found_candidate = true;
                best_node = node;
                best_value = evaluation.pool_value;
            }
        }

        if (!found_candidate) {
            break;
        }

        selected_mask[best_node] = 1;
        result.defense_seeds.push_back(best_node);
        result.round_best_values.push_back(best_value);
    }

    std::sort(result.defense_seeds.begin(), result.defense_seeds.end());
    result.defense_seeds.erase(
        std::unique(result.defense_seeds.begin(), result.defense_seeds.end()),
        result.defense_seeds.end());

    const MonteCarloOptions final_options{
        options.empirical_samples,
        options.seed ^ 0xdb4f0b9175ae2165ULL
    };
    const auto final_evaluation = estimate_empirical_pool_value(
        options.propagation_model,
        graph, conflict, innate, attack_pool, result.defense_seeds, params, final_options);
    result.empirical_pool_value = final_evaluation.pool_value;
    return result;
}

EmpiricalDefenseSearchResult run_empirical_greedy_defense_search_pair_rr_dense(
    PropagationModel propagation_model,
    const DirectedGraph& graph,
    const DenseConflict& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const CascadeParams& params,
    const DefenseOracleOptions& options) {
    validate_problem_sizes(graph, conflict, innate);
    validate_empirical_search_options(options);
    if (attack_pool.empty()) {
        throw std::invalid_argument("Pair-RR empirical defense search requires a nonempty attack pool");
    }

    const auto pool_samples = build_pair_rr_empirical_pool_samples(
        propagation_model,
        graph,
        conflict,
        attack_pool,
        options.empirical_samples,
        options.seed,
        "Pair-RR empirical selection sampling");
    return run_empirical_greedy_defense_search_pair_rr_from_samples(
        graph.num_nodes(), innate, pool_samples, params, options);
}

namespace {

struct DefenseCandidate {
    DefenseCandidateSource source = DefenseCandidateSource::None;
    std::string label;
    std::vector<std::size_t> origin_rounds;
    std::vector<NodeId> seeds;
    double empirical_value = std::numeric_limits<double>::quiet_NaN();
};

std::string format_round_indices(const std::vector<std::size_t>& rounds) {
    if (rounds.empty()) {
        return "not_applicable";
    }
    std::string text = "[";
    for (std::size_t index = 0; index < rounds.size(); ++index) {
        if (index != 0) {
            text += ',';
        }
        text += std::to_string(rounds[index]);
    }
    text += ']';
    return text;
}

bool same_seed_set(
    const std::vector<NodeId>& lhs,
    const std::vector<NodeId>& rhs) {
    return lhs == rhs;
}

void add_unique_defense_candidate(
    std::vector<DefenseCandidate>& candidates,
    DefenseCandidateSource source,
    std::string label,
    std::vector<NodeId> seeds,
    std::vector<std::size_t> origin_rounds = {}) {
    std::sort(seeds.begin(), seeds.end());
    seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
    for (auto& candidate : candidates) {
        if (same_seed_set(candidate.seeds, seeds)) {
            for (const auto round : origin_rounds) {
                if (std::find(candidate.origin_rounds.begin(), candidate.origin_rounds.end(), round) ==
                    candidate.origin_rounds.end()) {
                    candidate.origin_rounds.push_back(round);
                }
            }
            return;
        }
    }
    candidates.push_back(DefenseCandidate{
        source,
        std::move(label),
        std::move(origin_rounds),
        std::move(seeds),
        std::numeric_limits<double>::quiet_NaN()
    });
}

}  // namespace

DefenseOracleResult run_defense_oracle(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<NodeId>& initial_defense,
    const CascadeParams& params,
    const DefenseOracleOptions& options) {
    DefenseOracleResult result;
    try {
        validate_problem_sizes(graph, conflict, innate);
        validate_certificate_options(options);
        if (options.candidate_policy == DefenseCandidatePolicy::EmpiricalAndSurrogate) {
            validate_empirical_search_options(options);
        }
        validate_mixed_distribution_options(options);

        const auto normalized_initial_defense =
            normalized_seed_set(graph.num_nodes(), initial_defense);
        if (normalized_initial_defense.size() > options.budget) {
            result.status = DefenseOracleStatus::InvalidInput;
            result.status_message = "Initial defense exceeds the defense budget";
            return result;
        }

        const bool paper_exact_mode =
            options.defense_oracle_mode == DefenseOracleMode::PaperExact;
        const bool use_empirical_search =
            options.candidate_policy == DefenseCandidatePolicy::EmpiricalAndSurrogate;
        if (use_empirical_search) {
            const bool dense_pair_rr = dynamic_cast<const DenseConflict*>(&conflict) != nullptr;
            result.empirical_evaluator = dense_pair_rr ? "pair_rr_dense" : "pair_rr_amgcl";
            if (options.progress_callback) {
                options.progress_callback("Defense Oracle: empirical search", 0.0);
            }
            {
                const auto selection_samples = build_pair_rr_empirical_pool_samples(
                    options.propagation_model,
                    graph,
                    conflict,
                    attack_pool,
                    options.empirical_samples,
                    options.seed,
                    "Defense Oracle: empirical selection pair-RR sampling",
                    options.progress_callback);
                result.empirical_selection_samples = selection_samples.samples.size();
                result.empirical_result =
                    run_empirical_greedy_defense_search_pair_rr_from_samples(
                        graph.num_nodes(),
                        innate,
                        selection_samples,
                        params,
                        options);
            }
            if (options.progress_callback) {
                options.progress_callback("Defense Oracle: empirical search", 100.0);
            }
        } else {
            result.empirical_evaluator = "not_used";
        }
        if (options.progress_callback) {
            options.progress_callback("Defense Oracle: lower certificate", 0.0);
        }

        if (!paper_exact_mode &&
            options.mixed_distribution_mode == MixedDistributionMode::Adaptive) {
            result.mixed_certificate = compute_adaptive_mixed_lower_certificate_ic(
                graph, conflict, innate, attack_pool, params, options);
        } else {
            result.mixed_certificate = compute_mixed_lower_certificate_ic(
                graph, conflict, innate, attack_pool, params, options);
        }
        if (options.progress_callback) {
            options.progress_callback("Defense Oracle: lower certificate", 100.0);
            options.progress_callback("Defense Oracle: candidate validation", 0.0);
        }

        std::vector<DefenseCandidate> candidates;
        if (use_empirical_search) {
            add_unique_defense_candidate(
                candidates,
                DefenseCandidateSource::Empirical,
                "Empirical",
                result.empirical_result.defense_seeds);
        }
        const bool adaptive_mixture =
            options.mixed_distribution_mode == MixedDistributionMode::Adaptive;
        const std::size_t adaptive_surrogate_count = adaptive_mixture
            ? result.mixed_certificate.round_greedy_defenses.size()
            : 1;
        result.adaptive_candidate_count_before_deduplication =
            (use_empirical_search ? 1 : 0) + adaptive_surrogate_count;
        if (adaptive_mixture) {
            for (std::size_t round = 0;
                 round < result.mixed_certificate.round_greedy_defenses.size();
                 ++round) {
                add_unique_defense_candidate(
                    candidates,
                    DefenseCandidateSource::Surrogate,
                    "SurrogateRound" + std::to_string(round + 1),
                    result.mixed_certificate.round_greedy_defenses[round],
                    {round + 1});
            }
        } else {
            add_unique_defense_candidate(
                candidates,
                DefenseCandidateSource::Surrogate,
                "Surrogate",
                result.mixed_certificate.greedy_defense);
        }

        result.empirical_value_empirical = std::numeric_limits<double>::quiet_NaN();
        result.empirical_value_surrogate = std::numeric_limits<double>::quiet_NaN();

        result.defense_candidate_count = candidates.size();
        result.adaptive_candidate_count_after_deduplication = candidates.size();
        result.lower_certificate = result.mixed_certificate.lower_certificate;
        if (use_empirical_search) {
            std::vector<std::vector<NodeId>> validation_candidates;
            validation_candidates.reserve(candidates.size());
            for (const auto& candidate : candidates) {
                validation_candidates.push_back(candidate.seeds);
            }
            const auto validation_result =
                evaluate_empirical_pool_candidates_pair_rr_streaming(
                    options.propagation_model,
                    graph,
                    conflict,
                    innate,
                    attack_pool,
                    validation_candidates,
                    params,
                    options.empirical_samples,
                    options.seed ^ 0xa0761d6478bd642fULL,
                    "Defense Oracle: empirical validation pair-RR sampling",
                    options.progress_callback);
            result.empirical_validation_samples = validation_result.generated_samples;
            result.empirical_validation_independent = true;
            result.empirical_validation_streaming = true;
            bool found_best = false;
            DefenseCandidate best_candidate;
            for (std::size_t candidate_index = 0;
                 candidate_index < candidates.size();
                 ++candidate_index) {
                auto& candidate = candidates[candidate_index];
                const auto& evaluation =
                    validation_result.candidate_results[candidate_index];
                candidate.empirical_value = evaluation.pool_value;
                result.defense_candidate_sources.push_back(candidate.label);
                result.defense_candidate_origin_rounds.push_back(
                    format_round_indices(candidate.origin_rounds));
                result.defense_candidate_values.push_back(candidate.empirical_value);

                switch (candidate.source) {
                    case DefenseCandidateSource::Empirical:
                        result.empirical_value_empirical = candidate.empirical_value;
                        break;
                    case DefenseCandidateSource::Surrogate:
                        if (std::isnan(result.empirical_value_surrogate) ||
                            candidate.empirical_value < result.empirical_value_surrogate) {
                            result.empirical_value_surrogate = candidate.empirical_value;
                        }
                        break;
                    case DefenseCandidateSource::None:
                        break;
                }

                if (!found_best ||
                    candidate.empirical_value < best_candidate.empirical_value ||
                    (candidate.empirical_value == best_candidate.empirical_value &&
                     candidate.label < best_candidate.label)) {
                    found_best = true;
                    best_candidate = candidate;
                }
            }
            if (found_best) {
                result.defense_seeds = best_candidate.seeds;
                result.empirical_value_selected = best_candidate.empirical_value;
                result.candidate_source = best_candidate.source;
                result.selected_candidate_mixed_round =
                    format_round_indices(best_candidate.origin_rounds);
            }
        } else if (!candidates.empty()) {
            const auto& surrogate = candidates.front();
            result.defense_seeds = surrogate.seeds;
            result.candidate_source = surrogate.source;
            result.defense_candidate_sources.push_back(surrogate.label);
            result.defense_candidate_origin_rounds.push_back(
                format_round_indices(surrogate.origin_rounds));
            result.selected_candidate_mixed_round =
                format_round_indices(surrogate.origin_rounds);
        }
        if (options.progress_callback) {
            options.progress_callback("Defense Oracle: candidate validation", 100.0);
        }
    } catch (const std::exception& error) {
        result.status = DefenseOracleStatus::InvalidInput;
        result.status_message = error.what();
        result.defense_seeds.clear();
        result.residual_pool.clear();
        result.fixed_attack_certificates.clear();
        result.candidate_source = DefenseCandidateSource::None;
        return result;
    }

    result.status = DefenseOracleStatus::SearchCompleted;
    result.status_message = "Defense search completed";
    return result;
}

}  // namespace rcd
