#pragma once

#include "rcd/conflict.hpp"
#include "rcd/evaluation.hpp"
#include "rcd/graph.hpp"
#include "rcd/objective.hpp"
#include "rcd/progress.hpp"
#include "rcd/rr_sets.hpp"
#include "rcd/types.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace rcd {

struct AttackPoolEntry {
    std::vector<NodeId> attack_seeds;
    double upper_certificate = std::numeric_limits<double>::quiet_NaN();
};

enum class MixedDistributionMode {
    Uniform,
    Adaptive
};

enum class DefenseOracleMode {
    Engineering,
    PaperExact
};

enum class DefenseCandidatePolicy {
    EmpiricalAndSurrogate,
    SurrogateOnly
};

struct DefenseOracleOptions {
    std::size_t budget = 10;
    std::size_t rr_samples = 1000;
    std::size_t empirical_samples = 1000000;
    std::uint64_t seed = 1;
    double epsilon_rr = 0.1;
    MixedDistributionMode mixed_distribution_mode = MixedDistributionMode::Uniform;
    std::size_t mixed_rounds = 5;
    double mixed_learning_rate = 1.0;
    double failure_probability = 0.025;
    std::size_t max_rr_samples = 0;
    PropagationModel propagation_model = PropagationModel::IC;
    double oracle_epsilon = std::numeric_limits<double>::quiet_NaN();
    DefenseOracleMode defense_oracle_mode = DefenseOracleMode::Engineering;
    DefenseCandidatePolicy candidate_policy = DefenseCandidatePolicy::EmpiricalAndSurrogate;
    ProgressCallback progress_callback;
};

enum class DefenseOracleStatus {
    SearchCompleted,
    InvalidInput
};

enum class DefenseCandidateSource {
    None,
    Empirical,
    Surrogate
};

struct ResidualAttackEntry {
    std::vector<NodeId> original_attack;
    std::vector<NodeId> residual_attack;
};

struct FixedAttackLowerCertificate {
    std::vector<NodeId> residual_attack;
    std::vector<NodeId> greedy_defense;
    double estimated_k = 0.0;
    double estimated_lower_surrogate_value = 0.0;
    double unclamped_lower_certificate = std::numeric_limits<double>::quiet_NaN();
    double lower_certificate = std::numeric_limits<double>::quiet_NaN();
    std::vector<WeightedRRSample> samples;
};

struct MixedLowerCertificate {
    std::vector<NodeId> greedy_defense;
    std::string distribution_type = "uniform";
    std::vector<double> attack_distribution;
    std::vector<double> final_attack_distribution;
    std::vector<double> round_lower_certificates;
    std::vector<std::vector<double>> round_attack_distributions;
    std::vector<std::vector<NodeId>> round_greedy_defenses;
    std::size_t mixed_rounds = 5;
    std::size_t selected_round = 0;
    std::string certificate_status = "fixed_sample_estimate";
    double failure_probability = 0.025;
    double estimated_constant = 0.0;
    double estimated_lower_surrogate_value = 0.0;
    double lower_certificate = std::numeric_limits<double>::quiet_NaN();
    double unclamped_lower_certificate = std::numeric_limits<double>::quiet_NaN();
    double raw_lower_certificate = std::numeric_limits<double>::quiet_NaN();
    double conservative_constant = std::numeric_limits<double>::quiet_NaN();
    double conservative_lower_surrogate_value = std::numeric_limits<double>::quiet_NaN();
    double oracle_epsilon = std::numeric_limits<double>::quiet_NaN();
    std::string certificate_formula = "additive_oracle_error";
    std::size_t required_rr_samples = 0;
    std::size_t rr_sample_cap = 0;
    bool rr_sample_cap_triggered = false;
    std::size_t generated_rr_samples = 0;
    double oracle_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double distribution_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double value_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double constant_failure_probability = std::numeric_limits<double>::quiet_NaN();
    double surrogate_failure_probability = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::size_t> sampled_attack_indices;
    std::vector<WeightedRRSample> samples;
};

struct EmpiricalPoolEvaluationResult {
    double pool_value = std::numeric_limits<double>::quiet_NaN();
    std::size_t worst_attack_index = 0;
    MonteCarloResult worst_attack_result;
    std::vector<MonteCarloResult> attack_results;
};

struct EmpiricalDefenseSearchResult {
    std::vector<NodeId> defense_seeds;
    double empirical_pool_value = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> round_best_values;
};

struct PairRRStreamingEvaluationResult {
    std::size_t generated_samples = 0;
    std::vector<EmpiricalPoolEvaluationResult> candidate_results;
};

struct DefenseOracleResult {
    DefenseOracleStatus status = DefenseOracleStatus::InvalidInput;
    std::string status_message;
    std::vector<NodeId> defense_seeds;
    std::vector<ResidualAttackEntry> residual_pool;
    std::vector<FixedAttackLowerCertificate> fixed_attack_certificates;
    EmpiricalDefenseSearchResult empirical_result;
    std::string empirical_evaluator;
    std::size_t empirical_selection_samples = 0;
    std::size_t empirical_validation_samples = 0;
    bool empirical_validation_independent = false;
    bool empirical_validation_streaming = false;
    MixedLowerCertificate mixed_certificate;
    DefenseCandidateSource candidate_source = DefenseCandidateSource::None;
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
    double lower_certificate = std::numeric_limits<double>::quiet_NaN();
};

std::string to_string(DefenseOracleStatus status);
std::string to_string(DefenseCandidateSource source);
std::string to_string(MixedDistributionMode mode);
std::string to_string(DefenseOracleMode mode);
MixedDistributionMode parse_mixed_distribution_mode(const std::string& value);
DefenseOracleMode parse_defense_oracle_mode(const std::string& value);

std::vector<NodeId> normalized_seed_set(
    std::size_t num_nodes,
    const std::vector<NodeId>& seeds);

std::vector<NodeId> residual_attack_set(
    std::size_t num_nodes,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds);

FixedAttackLowerCertificate compute_fixed_attack_lower_certificate_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& fixed_attack,
    const CascadeParams& params,
    const DefenseOracleOptions& options);

MixedLowerCertificate compute_mixed_lower_certificate_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const CascadeParams& params,
    const DefenseOracleOptions& options);

MixedLowerCertificate compute_mixed_lower_certificate_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<double>& attack_distribution,
    const CascadeParams& params,
    const DefenseOracleOptions& options);

MixedLowerCertificate compute_adaptive_mixed_lower_certificate_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const CascadeParams& params,
    const DefenseOracleOptions& options);

EmpiricalPoolEvaluationResult estimate_empirical_pool_value_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options);

EmpiricalPoolEvaluationResult estimate_empirical_pool_value(
    PropagationModel propagation_model,
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options);

EmpiricalPoolEvaluationResult estimate_empirical_pool_value_pair_rr_dense(
    PropagationModel propagation_model,
    const DirectedGraph& graph,
    const DenseConflict& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options);

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
    const ProgressCallback& progress_callback = {});

EmpiricalDefenseSearchResult run_empirical_greedy_defense_search_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const CascadeParams& params,
    const DefenseOracleOptions& options);

EmpiricalDefenseSearchResult run_empirical_greedy_defense_search_pair_rr_dense(
    PropagationModel propagation_model,
    const DirectedGraph& graph,
    const DenseConflict& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const CascadeParams& params,
    const DefenseOracleOptions& options);

DefenseOracleResult run_defense_oracle(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<NodeId>& initial_defense,
    const CascadeParams& params,
    const DefenseOracleOptions& options);

}  // namespace rcd
