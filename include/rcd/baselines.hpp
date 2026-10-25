#pragma once

#include "rcd/conflict.hpp"
#include "rcd/crd.hpp"
#include "rcd/graph.hpp"
#include "rcd/objective.hpp"
#include "rcd/propagation.hpp"
#include "rcd/types.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

namespace rcd {

enum class BaselineAlgorithm {
    FJCIter,
    FJCLinearIter,
    HISTIter,
    HIST,
    Degree,
    Random,
    Exact
};

BaselineAlgorithm parse_baseline_algorithm(const std::string& value);
std::string to_string(BaselineAlgorithm algorithm);

enum class BaselineSamplingMode {
    Fixed,
    SubsimAdaptive,
    ClassicFullRR,
    NotApplicable
};

std::string to_string(BaselineSamplingMode mode);

struct BaselineOptions {
    BaselineAlgorithm algorithm = BaselineAlgorithm::Degree;
    std::size_t attacker_budget = 10;
    std::size_t defender_budget = 10;
    std::size_t iterations = 3;
    std::size_t fjc_pair_rr_samples = 100;
    std::size_t hist_rr_samples_floor = 0;
    std::size_t mc_simulations = 10000;
    std::uint64_t seed = 1;
    PropagationModel propagation_model = PropagationModel::IC;
    double epsilon = 0.1;
    double validation_failure_probability = 0.01;
    std::size_t max_adaptive_rounds = 12;
    std::size_t max_rr_samples = 0;
    std::size_t fjc_linear_rr_samples = 100;
};

struct BaselineIterationLog {
    std::size_t iteration = 0;
    std::size_t attack_pool_size = 0;
    std::vector<NodeId> defense_seeds;
    std::vector<NodeId> attack_seeds;
    double selection_objective_value = 0.0;
    double iteration_runtime_sec = 0.0;
    bool duplicate_attack = false;
    std::size_t sentinel_size = 0;
    std::size_t selection_rr_samples = 0;
    std::size_t validation_rr_samples = 0;
    std::size_t validation_early_stopped_samples = 0;
    double validation_objective_value = std::numeric_limits<double>::quiet_NaN();
    double validation_upper_bound = std::numeric_limits<double>::quiet_NaN();
    std::string validation_status;
    std::string sampling_mode;
    std::size_t adaptive_rounds_used = 0;
    double target_quality = std::numeric_limits<double>::quiet_NaN();
    std::size_t required_rr_samples = 0;
    std::size_t generated_rr_samples = 0;
    std::size_t verification_samples = 0;
    double verification_bound = std::numeric_limits<double>::quiet_NaN();
    bool verification_accepted = false;
    bool rr_sample_cap_triggered = false;
    std::size_t attack_adaptive_rounds_used = 0;
    double attack_target_quality = std::numeric_limits<double>::quiet_NaN();
    std::size_t attack_required_rr_samples = 0;
    std::size_t attack_generated_rr_samples = 0;
    std::size_t attack_verification_samples = 0;
    std::size_t attack_validation_early_stopped_samples = 0;
    double attack_verification_lower_bound = std::numeric_limits<double>::quiet_NaN();
    bool attack_verification_accepted = false;
    bool attack_rr_sample_cap_triggered = false;
    std::string attack_validation_status;
};

struct BaselineResult {
    BaselineAlgorithm algorithm = BaselineAlgorithm::Degree;
    std::string selection_objective;
    bool uses_attack_pool = false;
    bool uses_fj_objective_in_selection = false;
    bool uses_certificate = false;
    std::string certificate_status = "not_applicable";
    std::string selection_backend = "not_applicable";
    std::string selection_context = "not_applicable";
    std::size_t fjc_pair_rr_samples = 0;
    std::size_t fjc_linear_rr_samples = 0;
    std::size_t hist_rr_samples_floor = 0;
    BaselineSamplingMode sampling_mode = BaselineSamplingMode::Fixed;
    std::size_t max_adaptive_rounds = 0;
    std::size_t selection_mc_simulations = 0;
    std::vector<NodeId> defense_seeds;
    std::vector<NodeId> attack_seeds;
    std::vector<BaselineIterationLog> iterations;
    double algorithm_runtime_sec = 0.0;
    CRDEvaluationSummary evaluation;
};

BaselineResult run_baseline(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const BaselineOptions& options,
    std::ostream* progress_os = nullptr);

}  // namespace rcd
