#pragma once

#include "rcd/attack_oracle.hpp"
#include "rcd/conflict.hpp"
#include "rcd/defense_oracle.hpp"
#include "rcd/evaluation.hpp"
#include "rcd/graph.hpp"
#include "rcd/objective.hpp"
#include "rcd/types.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

namespace rcd {

enum class CRDMode {
    Engineering,
    PaperExact
};

enum class CRDVariant {
    Full,
    SingleAttackPool,
    SurrogateOnly,
    AdaptiveMix,
    LinearApprox
};

std::string to_string(CRDVariant variant);
CRDVariant parse_crd_variant(const std::string& value);
bool is_crd_family_algorithm(const std::string& value);

struct CRDOptions {
    std::size_t max_iterations = 3;
    std::size_t defense_budget = 100;
    std::size_t attack_budget = 100;
    std::size_t attack_rr_samples = 1000000;
    std::size_t defense_rr_samples = 1000000;
    std::size_t empirical_samples = 1000000;
    std::uint64_t seed = 1;
    double epsilon_rr = 0.1;
    double epsilon_cert = 10.0;
    double failure_probability = 0.01;
    MixedDistributionMode mixed_distribution_mode = MixedDistributionMode::Uniform;
    std::size_t mixed_rounds = 5;
    double mixed_learning_rate = 1.0;
    std::size_t max_rr_samples = 0;
    PropagationModel propagation_model = PropagationModel::IC;
    CRDMode crd_mode = CRDMode::Engineering;
    CRDVariant variant = CRDVariant::Full;
    DefenseCandidatePolicy defense_candidate_policy =
        DefenseCandidatePolicy::EmpiricalAndSurrogate;
    std::size_t engineering_holdout_samples = 0;
};

struct CRDIterationLog {
    std::size_t iteration = 0;
    std::string crd_mode;
    std::string crd_variant;
    double epsilon_a = std::numeric_limits<double>::quiet_NaN();
    double epsilon_b = std::numeric_limits<double>::quiet_NaN();
    double delta_a = std::numeric_limits<double>::quiet_NaN();
    double delta_b = std::numeric_limits<double>::quiet_NaN();
    std::size_t theta_a = 0;
    std::size_t theta_b = 0;
    std::size_t theta_emp = 0;
    std::size_t theta_emp_selection = 0;
    std::size_t theta_emp_validation = 0;
    bool empirical_validation_independent = false;
    bool empirical_validation_streaming = false;
    std::size_t attack_pool_size = 0;
    std::size_t attack_pool_size_before_defense = 0;
    std::size_t attack_pool_size_after_attack = 0;
    std::string attack_pool_update;
    std::string propagation_model;
    std::vector<NodeId> defense_seeds;
    std::vector<NodeId> attack_seeds;
    DefenseOracleStatus defense_status = DefenseOracleStatus::InvalidInput;
    DefenseCandidateSource defense_candidate_source = DefenseCandidateSource::None;
    std::string defense_status_message;
    std::size_t defense_candidate_count = 0;
    std::vector<std::string> defense_candidate_sources;
    std::vector<std::string> defense_candidate_origin_rounds;
    std::vector<double> defense_candidate_values;
    std::string selected_candidate_mixed_round = "not_applicable";
    std::size_t adaptive_candidate_count_before_deduplication = 0;
    std::size_t adaptive_candidate_count_after_deduplication = 0;
    double empirical_value_empirical = std::numeric_limits<double>::quiet_NaN();
    double empirical_value_surrogate = std::numeric_limits<double>::quiet_NaN();
    double empirical_value_selected = std::numeric_limits<double>::quiet_NaN();
    std::string mixed_distribution_type;
    std::size_t mixed_rounds = 0;
    std::size_t selected_mixed_round = 0;
    std::vector<double> round_lower_certificates;
    std::vector<std::vector<double>> round_attack_distributions;
    std::vector<std::vector<NodeId>> round_surrogate_defenses;
    double attack_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double defense_failure_probability = std::numeric_limits<double>::quiet_NaN();
    std::string attack_certificate_status;
    std::size_t attack_required_rr_samples = 0;
    std::size_t attack_rr_sample_cap = 0;
    bool attack_rr_sample_cap_triggered = false;
    std::size_t attack_generated_rr_samples = 0;
    std::string defense_certificate_status;
    std::size_t defense_required_rr_samples = 0;
    std::size_t defense_rr_sample_cap = 0;
    bool defense_rr_sample_cap_triggered = false;
    std::size_t defense_generated_rr_samples = 0;
    double unclamped_lower_certificate = std::numeric_limits<double>::quiet_NaN();
    double raw_lower_certificate = std::numeric_limits<double>::quiet_NaN();
    std::string lower_certificate_formula;
    double defense_conservative_constant = std::numeric_limits<double>::quiet_NaN();
    double defense_conservative_surrogate = std::numeric_limits<double>::quiet_NaN();
    double raw_upper_certificate = std::numeric_limits<double>::quiet_NaN();
    std::string attack_certificate_formula;
    double attack_conservative_constant = std::numeric_limits<double>::quiet_NaN();
    double attack_conservative_surrogate = std::numeric_limits<double>::quiet_NaN();
    double attack_constant_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double attack_surrogate_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double defense_distribution_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double defense_constant_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double defense_surrogate_failure_probability = std::numeric_limits<double>::quiet_NaN();
    std::string empirical_evaluator;
    bool sample_cap_triggered = false;
    bool paper_certificate_valid = false;
    double lower_certificate = std::numeric_limits<double>::quiet_NaN();
    double upper_certificate = std::numeric_limits<double>::quiet_NaN();
    double certificate_gap = std::numeric_limits<double>::quiet_NaN();
    double iteration_runtime_sec = std::numeric_limits<double>::quiet_NaN();
};

struct CRDEvaluationSummary {
    bool available = false;
    std::string attack_response_policy = "not_available";
    std::size_t attack_response_rr_samples = 0;
    MonteCarloResult defense_vs_attack;
    MonteCarloResult no_defense_attack_only;
    MonteCarloResult defense_only;
    std::vector<NodeId> attack_only_seeds;
};

struct CRDResult {
    CRDVariant variant = CRDVariant::Full;
    std::vector<NodeId> defense_seeds;
    std::vector<NodeId> attack_seeds;
    std::vector<AttackPoolEntry> attack_pool;
    std::vector<CRDIterationLog> iterations;
    std::string initial_attack_pool_policy;
    std::size_t initial_attack_pool_size = 0;
    double best_lower_certificate = std::numeric_limits<double>::quiet_NaN();
    double best_upper_certificate = std::numeric_limits<double>::quiet_NaN();
    double best_certificate_gap = std::numeric_limits<double>::quiet_NaN();
    std::string final_selection_policy = "not_available";
    std::size_t final_selected_iteration = 0;
    std::size_t final_holdout_samples = 0;
    std::uint64_t final_holdout_seed = 0;
    bool final_holdout_independent = false;
    bool final_holdout_streaming = false;
    double final_selected_holdout_value = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> final_iteration_holdout_values;
    double algorithm_runtime_sec = std::numeric_limits<double>::quiet_NaN();
    CRDEvaluationSummary evaluation;
};

std::string to_string(CRDMode mode);
CRDMode parse_crd_mode(const std::string& value);

std::string format_seed_set(const std::vector<NodeId>& seeds);
std::string format_seed_set_preview(
    const std::vector<NodeId>& seeds,
    std::size_t max_displayed_seeds = 20);

std::string format_decimal(double value);

void print_crd_iteration_log(std::ostream& os, const CRDIterationLog& log);

CRDResult run_crd(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const CRDOptions& options,
    std::ostream* progress_os = nullptr);

// Large-graph engineering variant. It retains the CRD attack-pool outer loop,
// but selects seeds using only the reference-context linear FJ term and
// weighted RR coverage. It does not produce a paper certificate.
CRDResult run_crd_linear(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const CRDOptions& options,
    std::ostream* progress_os = nullptr);

}  // namespace rcd
