#include "rcd/attack_oracle.hpp"

#include "rcd/certificate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace rcd {
namespace {

std::vector<std::uint8_t> make_reached_mask(std::size_t num_nodes, const std::vector<NodeId>& reached) {
    std::vector<std::uint8_t> mask(num_nodes, 0);
    for (const auto node : reached) {
        if (static_cast<std::size_t>(node) >= num_nodes) {
            throw std::out_of_range("Reached node is out of range");
        }
        mask[node] = 1;
    }
    return mask;
}

void validate_seeds(std::size_t num_nodes, const std::vector<NodeId>& seeds) {
    for (const auto seed : seeds) {
        if (static_cast<std::size_t>(seed) >= num_nodes) {
            throw std::out_of_range("Seed node is out of range");
        }
    }
}

void validate_options(const AttackOracleOptions& options) {
    if (options.rr_samples == 0) {
        throw std::invalid_argument("Attack Oracle requires at least one RR sample");
    }
    if (options.max_rr_samples != 0 && options.max_rr_samples < options.rr_samples) {
        throw std::invalid_argument(
            "Attack Oracle max RR samples must be 0 or at least the base RR samples");
    }
    if (!std::isfinite(options.failure_probability) ||
        options.failure_probability <= 0.0 ||
        options.failure_probability >= 1.0) {
        throw std::invalid_argument("Attack Oracle failure probability must be in (0, 1)");
    }
    if (!std::isfinite(options.epsilon_rr) || options.epsilon_rr <= 0.0) {
        throw std::invalid_argument("Attack Oracle epsilon_rr must be finite and positive");
    }
    if (!std::isnan(options.oracle_epsilon) &&
        (!std::isfinite(options.oracle_epsilon) || options.oracle_epsilon <= 0.0)) {
        throw std::invalid_argument("Attack Oracle oracle_epsilon must be finite and positive");
    }
}

struct FailureAllocation {
    double oracle_delta = std::numeric_limits<double>::quiet_NaN();
    double distribution_delta = std::numeric_limits<double>::quiet_NaN();
    double value_delta = std::numeric_limits<double>::quiet_NaN();
    double constant_delta = std::numeric_limits<double>::quiet_NaN();
    double surrogate_delta = std::numeric_limits<double>::quiet_NaN();
};

FailureAllocation fixed_failure_allocation(const AttackOracleOptions& options) {
    FailureAllocation allocation;
    allocation.oracle_delta = options.failure_probability;
    allocation.distribution_delta = options.failure_probability;
    allocation.value_delta = options.failure_probability;
    allocation.constant_delta = allocation.value_delta / 2.0;
    allocation.surrogate_delta = allocation.value_delta / 2.0;
    return allocation;
}

double attack_oracle_epsilon(const AttackOracleOptions& options) {
    if (!std::isnan(options.oracle_epsilon)) {
        return options.oracle_epsilon;
    }
    return options.epsilon_rr;
}

void set_failure_fields(
    AttackOracleResult& result,
    const FailureAllocation& allocation) {
    result.oracle_failure_probability = allocation.oracle_delta;
    result.value_failure_probability = allocation.value_delta;
    result.constant_failure_probability = allocation.constant_delta;
    result.surrogate_failure_probability = allocation.surrogate_delta;
}

}  // namespace

namespace {

AttackOracleResult run_attack_oracle_with_sample_count(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const AttackOracleOptions& options,
    std::size_t sample_count,
    std::size_t required_sample_count,
    std::size_t rr_sample_cap,
    bool rr_sample_cap_triggered,
    std::uint64_t seed,
    const FailureAllocation& allocation) {
    std::mt19937_64 root_rng(seed);
    std::uniform_int_distribution<std::size_t> root_dist(0, graph.num_nodes() - 1);
    LazyLiveEdgeOracle live_oracle(graph.num_edges(), seed ^ 0xd1b54a32d192ed03ULL);
    LazyTriggeringEdgeOracle triggering_oracle(graph.num_nodes(), seed ^ 0x94d049bb133111ebULL);
    TraversalWorkspace forward_workspace(graph.num_nodes());
    TraversalWorkspace rr_workspace(graph.num_nodes());

    std::vector<WeightedRRSample> samples;
    samples.reserve(sample_count);
    double k_sum = 0.0;
    const std::size_t progress_interval = std::max<std::size_t>(1, sample_count / 10);

    // With no defender seeds, every realization has the same all-zero defense
    // reach mask. Reuse its exact conditional terms instead of solving the FJ
    // system once per RR sample.
    const bool empty_defense = defense_seeds.empty();
    Eigen::VectorXd empty_defense_upper_weights;
    double empty_defense_constant = 0.0;
    if (empty_defense) {
        const std::vector<std::uint8_t> empty_defense_reached(graph.num_nodes(), 0);
        const auto terms = build_conditional_terms(
            conflict,
            innate,
            empty_defense_reached,
            params,
            OptimizedPlayer::Attacker);
        empty_defense_upper_weights = upper_surrogate_weights(terms);
        empty_defense_constant = terms.constant;
    }

    if (options.progress_callback) {
        options.progress_callback(
            "Attack Oracle: RR sampling started (total_samples=" +
                std::to_string(sample_count) + ")",
            0.0);
    }

    for (std::size_t i = 0; i < sample_count; ++i) {
        Eigen::VectorXd per_sample_upper_weights;
        const Eigen::VectorXd* upper_weights = &empty_defense_upper_weights;
        if (empty_defense) {
            k_sum += empty_defense_constant;
        } else {
            const std::vector<NodeId> defense_reached = forward_reachable(
                options.propagation_model,
                graph,
                defense_seeds,
                live_oracle,
                triggering_oracle,
                forward_workspace);
            const std::vector<std::uint8_t> defense_reached_mask =
                make_reached_mask(graph.num_nodes(), defense_reached);

            const auto terms = build_conditional_terms(
                conflict, innate, defense_reached_mask, params, OptimizedPlayer::Attacker);
            per_sample_upper_weights = upper_surrogate_weights(terms);
            upper_weights = &per_sample_upper_weights;
            k_sum += terms.constant;
        }

        const NodeId root = static_cast<NodeId>(root_dist(root_rng));
        std::vector<NodeId> rr_set = reverse_rr_set(
            options.propagation_model,
            graph,
            root,
            live_oracle,
            triggering_oracle,
            rr_workspace);
        std::sort(rr_set.begin(), rr_set.end());
        const Weight weight = static_cast<Weight>(graph.num_nodes()) * (*upper_weights)[root];
        samples.push_back(WeightedRRSample{std::move(rr_set), root, weight});

        live_oracle.reset();
        triggering_oracle.reset();
        if (options.progress_callback &&
            ((i + 1) % progress_interval == 0 || i + 1 == sample_count)) {
            options.progress_callback(
                "Attack Oracle: RR sampling",
                100.0 * static_cast<double>(i + 1) /
                    static_cast<double>(std::max<std::size_t>(sample_count, 1)));
        }
    }

    if (options.progress_callback) {
        options.progress_callback("Attack Oracle: certificate preparation", 0.0);
    }
    auto attack_seeds = weighted_coverage_greedy(
        graph.num_nodes(), samples, options.budget, {}, options.progress_callback);

    std::sort(attack_seeds.begin(), attack_seeds.end());
    attack_seeds.erase(std::unique(attack_seeds.begin(), attack_seeds.end()), attack_seeds.end());

    const double estimated_k = k_sum / static_cast<double>(sample_count);
    const double estimated_surrogate = weighted_coverage_value(samples, attack_seeds);
    const double oracle_epsilon = attack_oracle_epsilon(options);
    const double factor = greedy_coverage_factor();
    const double raw_upper_certificate =
        estimated_k + (estimated_surrogate + oracle_epsilon) / factor;
    const double beta = conflict_beta(conflict);
    const double conservative_k = upper_confidence_bound(
        estimated_k,
        quadratic_constant_sample_bound(graph.num_nodes()),
        sample_count,
        allocation.constant_delta);
    const double conservative_surrogate = upper_confidence_bound(
        estimated_surrogate,
        attacker_rr_sample_bound(graph.num_nodes(), beta),
        sample_count,
        allocation.surrogate_delta);
    const double upper_certificate =
        conservative_k + (conservative_surrogate + oracle_epsilon) / factor;

    AttackOracleResult result;
    result.attack_seeds = std::move(attack_seeds);
    result.upper_certificate = upper_certificate;
    result.raw_upper_certificate = raw_upper_certificate;
    result.estimated_k = estimated_k;
    result.estimated_surrogate_value = estimated_surrogate;
    result.conservative_k = conservative_k;
    result.conservative_surrogate_value = conservative_surrogate;
    result.oracle_epsilon = oracle_epsilon;
    result.certificate_formula = "additive_oracle_error";
    result.candidate_ground_set = "all_nodes";
    result.required_rr_samples = required_sample_count;
    result.rr_sample_cap = rr_sample_cap;
    result.rr_sample_cap_triggered = rr_sample_cap_triggered;
    result.generated_rr_samples = sample_count;
    result.samples = std::move(samples);
    set_failure_fields(result, allocation);
    if (options.progress_callback) {
        options.progress_callback("Attack Oracle: certificate computed", 100.0);
    }
    return result;
}

}  // namespace

AttackOracleResult run_attack_oracle(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const AttackOracleOptions& options) {
    validate_options(options);
    if (conflict.num_nodes() != graph.num_nodes() ||
        static_cast<std::size_t>(innate.size()) != graph.num_nodes()) {
        throw std::invalid_argument("Graph, conflict backend, and opinion vector sizes must match");
    }

    validate_seeds(graph.num_nodes(), defense_seeds);

    const auto allocation = fixed_failure_allocation(options);
    auto result = run_attack_oracle_with_sample_count(
        graph,
        conflict,
        innate,
        defense_seeds,
        params,
        options,
        options.rr_samples,
        options.rr_samples,
        options.max_rr_samples,
        false,
        options.seed,
        allocation);
    result.certificate_status = "fixed_sample_estimate";
    return result;
}

LinearAttackResponseResult run_linear_attack_response(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const LinearAttackResponseOptions& options) {
    if (options.rr_samples == 0) {
        throw std::invalid_argument("Linear attack response requires at least one RR sample");
    }
    if (conflict.num_nodes() != graph.num_nodes() ||
        static_cast<std::size_t>(innate.size()) != graph.num_nodes()) {
        throw std::invalid_argument("Graph, conflict backend, and opinion vector sizes must match");
    }
    validate_seeds(graph.num_nodes(), defense_seeds);

    std::vector<std::uint8_t> defense_seed_mask(graph.num_nodes(), 0);
    for (const NodeId seed : defense_seeds) {
        defense_seed_mask[seed] = 1;
    }
    const ConditionalTerms terms = build_conditional_terms(
        conflict,
        innate,
        defense_seed_mask,
        params,
        OptimizedPlayer::Attacker);

    std::vector<Weight> root_weights(graph.num_nodes(), 0.0);
    std::size_t positive_root_weights = 0;
    for (Eigen::Index i = 0; i < terms.ell.size(); ++i) {
        const double weight = terms.ell[i];
        if (!std::isfinite(weight)) {
            throw std::runtime_error("Linear attack response produced a non-finite root weight");
        }
        if (weight > 0.0) {
            root_weights[static_cast<std::size_t>(i)] = weight;
            ++positive_root_weights;
        }
    }

    const auto samples = generate_weighted_rr_samples_from_node_weights(
        options.propagation_model,
        graph,
        options.rr_samples,
        options.seed,
        root_weights,
        options.progress_callback);
    auto attack_seeds = weighted_coverage_greedy(
        graph.num_nodes(),
        samples,
        options.budget,
        {},
        options.progress_callback);
    std::sort(attack_seeds.begin(), attack_seeds.end());

    LinearAttackResponseResult result;
    result.attack_seeds = std::move(attack_seeds);
    result.generated_rr_samples = samples.size();
    result.positive_root_weights = positive_root_weights;
    result.estimated_linear_gain = weighted_coverage_value(samples, result.attack_seeds);
    return result;
}

}  // namespace rcd
