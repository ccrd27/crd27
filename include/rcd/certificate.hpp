#pragma once

#include "rcd/conflict.hpp"

#include <cstddef>
#include <vector>

namespace rcd {

struct PaperExactSamplePlanOptions {
    std::size_t num_nodes = 0;
    std::size_t attack_budget = 0;
    std::size_t defense_budget = 0;
    double beta = 0.0;
    double epsilon = 0.0;
    double failure_probability = 0.0;
    std::size_t iterations = 0;
    std::size_t max_rr_samples = 0;
};

struct PaperExactRoundSamplePlan {
    double attack_oracle_epsilon = 0.0;
    double defense_oracle_epsilon = 0.0;
    double attack_failure_probability = 0.0;
    double defense_failure_probability = 0.0;
    std::size_t required_attack_rr_samples = 0;
    std::size_t required_defense_rr_samples = 0;
    std::size_t attack_rr_samples = 0;
    std::size_t defense_rr_samples = 0;
    std::size_t attack_rr_sample_cap = 0;
    std::size_t defense_rr_sample_cap = 0;
    bool attack_rr_sample_cap_triggered = false;
    bool defense_rr_sample_cap_triggered = false;
};

double log_binomial_upper_bound(std::size_t n, std::size_t k);
double log_uniform_concentration_term(std::size_t n, std::size_t k, double failure_probability);

double hoeffding_radius(
    double sample_bound,
    std::size_t sample_count,
    double failure_probability);

double lower_confidence_bound(
    double mean,
    double sample_bound,
    std::size_t sample_count,
    double failure_probability);

double upper_confidence_bound(
    double mean,
    double sample_bound,
    std::size_t sample_count,
    double failure_probability);

std::size_t required_rr_samples(
    double sample_bound,
    double epsilon_surrogate,
    double guessed_scale,
    double log_family_delta);

double greedy_coverage_factor();
double paper_exact_final_oracle_error(double epsilon);
std::size_t paper_exact_attack_rr_samples(
    std::size_t num_nodes,
    std::size_t attack_budget,
    double beta,
    double attack_oracle_epsilon,
    double attack_failure_probability);
std::size_t paper_exact_defense_rr_samples(
    std::size_t num_nodes,
    std::size_t defense_budget,
    double beta,
    double defense_oracle_epsilon,
    double defense_failure_probability);
std::vector<PaperExactRoundSamplePlan> plan_paper_exact_rr_samples(
    const PaperExactSamplePlanOptions& options);

double conflict_beta(const ConflictBackend& conflict);
double conflict_beta(const DenseConflict& conflict);
double conflict_beta(const AmgclConflict& conflict);
double attacker_rr_sample_bound(std::size_t num_nodes, double beta);
double defender_rr_sample_bound(std::size_t num_nodes, double beta);
double quadratic_constant_sample_bound(std::size_t num_nodes);

}  // namespace rcd
