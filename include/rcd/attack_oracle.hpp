#pragma once

#include "rcd/conflict.hpp"
#include "rcd/graph.hpp"
#include "rcd/objective.hpp"
#include "rcd/propagation.hpp"
#include "rcd/progress.hpp"
#include "rcd/rr_sets.hpp"
#include "rcd/types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace rcd {

struct AttackOracleOptions {
    std::size_t budget = 10;
    std::size_t rr_samples = 1000;
    std::uint64_t seed = 1;
    double epsilon_rr = 0.1;
    double failure_probability = 0.01;
    std::size_t max_rr_samples = 0;
    PropagationModel propagation_model = PropagationModel::IC;
    double oracle_epsilon = std::numeric_limits<double>::quiet_NaN();
    ProgressCallback progress_callback;
};

struct AttackOracleResult {
    std::vector<NodeId> attack_seeds;
    double upper_certificate = 0.0;
    double raw_upper_certificate = 0.0;
    double estimated_k = 0.0;
    double estimated_surrogate_value = 0.0;
    double conservative_k = 0.0;
    double conservative_surrogate_value = 0.0;
    double oracle_epsilon = std::numeric_limits<double>::quiet_NaN();
    std::string certificate_formula = "additive_oracle_error";
    std::string candidate_ground_set = "all_nodes";
    std::string certificate_status = "fixed_sample_estimate";
    std::size_t required_rr_samples = 0;
    std::size_t rr_sample_cap = 0;
    bool rr_sample_cap_triggered = false;
    std::size_t generated_rr_samples = 0;
    double oracle_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double value_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double constant_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double surrogate_failure_probability = std::numeric_limits<double>::quiet_NaN();
    std::vector<WeightedRRSample> samples;
};

// Large-graph engineering response: fixes the opponent at its seed mask,
// linearizes the attacker-side FJ term once, and optimizes weighted RR
// coverage. It intentionally has no certificate.
struct LinearAttackResponseOptions {
    std::size_t budget = 10;
    std::size_t rr_samples = 1000;
    std::uint64_t seed = 1;
    PropagationModel propagation_model = PropagationModel::IC;
    ProgressCallback progress_callback;
};

struct LinearAttackResponseResult {
    std::vector<NodeId> attack_seeds;
    std::size_t generated_rr_samples = 0;
    std::size_t positive_root_weights = 0;
    double estimated_linear_gain = 0.0;
};

AttackOracleResult run_attack_oracle(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const AttackOracleOptions& options);

LinearAttackResponseResult run_linear_attack_response(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const LinearAttackResponseOptions& options);

}  // namespace rcd
