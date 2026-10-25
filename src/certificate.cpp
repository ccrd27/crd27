#include "rcd/certificate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace rcd {
namespace {

void validate_probability(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0 || value >= 1.0) {
        throw std::invalid_argument(std::string(name) + " must be in (0, 1)");
    }
}

void validate_nonnegative_finite(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
    }
}

void validate_finite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

void validate_positive_finite(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }
}

void validate_budget(std::size_t n, std::size_t k, const char* name) {
    if (k > n) {
        throw std::invalid_argument(std::string(name) + " must not exceed num_nodes");
    }
}

std::size_t ceil_to_sample_count(double value) {
    if (!std::isfinite(value) ||
        value > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Required RR sample count exceeds size_t range");
    }
    return std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(value)));
}

std::size_t required_samples_for_absolute_error(
    double sample_bound,
    double epsilon_absolute,
    double log_family_delta) {
    validate_nonnegative_finite(sample_bound, "sample_bound");
    validate_positive_finite(epsilon_absolute, "epsilon_absolute");
    validate_positive_finite(log_family_delta, "log_family_delta");
    if (sample_bound == 0.0) {
        return 1;
    }
    const double numerator = sample_bound * sample_bound * log_family_delta;
    const double denominator = 2.0 * epsilon_absolute * epsilon_absolute;
    return ceil_to_sample_count(numerator / denominator);
}

std::size_t apply_sample_cap(
    std::size_t required,
    std::size_t cap,
    bool& cap_triggered) {
    cap_triggered = (cap != 0 && required > cap);
    return cap_triggered ? cap : required;
}

}  // namespace

double log_binomial_upper_bound(std::size_t n, std::size_t k) {
    if (k > n) {
        throw std::invalid_argument("k must not exceed n");
    }
    k = std::min(k, n - k);
    if (k == 0) {
        return 0.0;
    }
    return std::lgamma(static_cast<double>(n) + 1.0) -
           std::lgamma(static_cast<double>(k) + 1.0) -
           std::lgamma(static_cast<double>(n - k) + 1.0);
}

double log_uniform_concentration_term(
    std::size_t n,
    std::size_t k,
    double failure_probability) {
    validate_probability(failure_probability, "failure_probability");
    return std::log(2.0) + log_binomial_upper_bound(n, k) -
           std::log(failure_probability);
}

double hoeffding_radius(
    double sample_bound,
    std::size_t sample_count,
    double failure_probability) {
    validate_nonnegative_finite(sample_bound, "sample_bound");
    if (sample_count == 0) {
        throw std::invalid_argument("sample_count must be positive");
    }
    validate_probability(failure_probability, "failure_probability");
    if (sample_bound == 0.0) {
        return 0.0;
    }
    return sample_bound * std::sqrt(
        std::log(2.0 / failure_probability) /
        (2.0 * static_cast<double>(sample_count)));
}

double lower_confidence_bound(
    double mean,
    double sample_bound,
    std::size_t sample_count,
    double failure_probability) {
    validate_finite(mean, "mean");
    return mean - hoeffding_radius(sample_bound, sample_count, failure_probability);
}

double upper_confidence_bound(
    double mean,
    double sample_bound,
    std::size_t sample_count,
    double failure_probability) {
    validate_finite(mean, "mean");
    return mean + hoeffding_radius(sample_bound, sample_count, failure_probability);
}

std::size_t required_rr_samples(
    double sample_bound,
    double epsilon_surrogate,
    double guessed_scale,
    double log_family_delta) {
    validate_nonnegative_finite(sample_bound, "sample_bound");
    validate_positive_finite(epsilon_surrogate, "epsilon_surrogate");
    validate_positive_finite(guessed_scale, "guessed_scale");
    validate_positive_finite(log_family_delta, "log_family_delta");

    const double factor = 2.0 - 1.0 / std::exp(1.0);
    const double epsilon_absolute = epsilon_surrogate * guessed_scale / factor;
    validate_positive_finite(epsilon_absolute, "epsilon_absolute");
    if (sample_bound == 0.0) {
        return 1;
    }

    return required_samples_for_absolute_error(
        sample_bound, epsilon_absolute, log_family_delta);
}

double greedy_coverage_factor() {
    return 1.0 - 1.0 / std::exp(1.0);
}

double paper_exact_final_oracle_error(double epsilon) {
    validate_positive_finite(epsilon, "epsilon");
    return greedy_coverage_factor() * epsilon / 4.0;
}

std::size_t paper_exact_attack_rr_samples(
    std::size_t num_nodes,
    std::size_t attack_budget,
    double beta,
    double attack_oracle_epsilon,
    double attack_failure_probability) {
    if (num_nodes == 0) {
        throw std::invalid_argument("num_nodes must be positive");
    }
    validate_budget(num_nodes, attack_budget, "attack_budget");
    validate_nonnegative_finite(beta, "beta");
    validate_positive_finite(attack_oracle_epsilon, "attack_oracle_epsilon");
    const double log_term = log_uniform_concentration_term(
        num_nodes, attack_budget, attack_failure_probability);
    const double epsilon_absolute =
        attack_oracle_epsilon / (2.0 - 1.0 / std::exp(1.0));
    return required_samples_for_absolute_error(
        attacker_rr_sample_bound(num_nodes, beta), epsilon_absolute, log_term);
}

std::size_t paper_exact_defense_rr_samples(
    std::size_t num_nodes,
    std::size_t defense_budget,
    double beta,
    double defense_oracle_epsilon,
    double defense_failure_probability) {
    if (num_nodes == 0) {
        throw std::invalid_argument("num_nodes must be positive");
    }
    validate_budget(num_nodes, defense_budget, "defense_budget");
    validate_nonnegative_finite(beta, "beta");
    validate_positive_finite(defense_oracle_epsilon, "defense_oracle_epsilon");
    const double log_term = log_uniform_concentration_term(
        num_nodes, defense_budget, defense_failure_probability);
    const double epsilon_absolute =
        defense_oracle_epsilon / (2.0 - 1.0 / std::exp(1.0));
    return required_samples_for_absolute_error(
        defender_rr_sample_bound(num_nodes, beta), epsilon_absolute, log_term);
}

std::vector<PaperExactRoundSamplePlan> plan_paper_exact_rr_samples(
    const PaperExactSamplePlanOptions& options) {
    if (options.num_nodes == 0) {
        throw std::invalid_argument("num_nodes must be positive");
    }
    validate_budget(options.num_nodes, options.attack_budget, "attack_budget");
    validate_budget(options.num_nodes, options.defense_budget, "defense_budget");
    validate_nonnegative_finite(options.beta, "beta");
    validate_positive_finite(options.epsilon, "epsilon");
    validate_probability(options.failure_probability, "failure_probability");
    if (options.iterations == 0) {
        throw std::invalid_argument("paper-exact iteration count must be positive");
    }

    const double allocation_denominator =
        2.0 * static_cast<double>(options.iterations);
    validate_positive_finite(allocation_denominator, "allocation_denominator");
    const double round_failure_probability =
        options.failure_probability / allocation_denominator;
    validate_probability(round_failure_probability, "round_failure_probability");

    const double attack_epsilon = paper_exact_final_oracle_error(options.epsilon);
    const double defense_epsilon = attack_epsilon;
    const std::size_t required_attack = paper_exact_attack_rr_samples(
        options.num_nodes,
        options.attack_budget,
        options.beta,
        attack_epsilon,
        round_failure_probability);
    const std::size_t required_defense = paper_exact_defense_rr_samples(
        options.num_nodes,
        options.defense_budget,
        options.beta,
        defense_epsilon,
        round_failure_probability);

    const std::size_t total_rounds = options.iterations;
    std::vector<PaperExactRoundSamplePlan> plans;
    plans.reserve(total_rounds);

    for (std::size_t t = 1; t <= total_rounds; ++t) {
        PaperExactRoundSamplePlan plan;
        plan.attack_oracle_epsilon = attack_epsilon;
        plan.defense_oracle_epsilon = defense_epsilon;
        plan.attack_failure_probability = round_failure_probability;
        plan.defense_failure_probability = round_failure_probability;
        plan.required_attack_rr_samples = required_attack;
        plan.required_defense_rr_samples = required_defense;
        plan.attack_rr_sample_cap = options.max_rr_samples;
        plan.defense_rr_sample_cap = options.max_rr_samples;
        plan.attack_rr_samples = apply_sample_cap(
            required_attack,
            plan.attack_rr_sample_cap,
            plan.attack_rr_sample_cap_triggered);
        plan.defense_rr_samples = apply_sample_cap(
            required_defense,
            plan.defense_rr_sample_cap,
            plan.defense_rr_sample_cap_triggered);
        plans.push_back(plan);
    }
    return plans;
}

double conflict_beta(const ConflictBackend& conflict) {
    return conflict.beta_bound();
}

double conflict_beta(const DenseConflict& conflict) {
    return conflict.beta_bound();
}

double conflict_beta(const AmgclConflict& conflict) {
    return conflict.beta_bound();
}

double attacker_rr_sample_bound(std::size_t num_nodes, double beta) {
    validate_nonnegative_finite(beta, "beta");
    return static_cast<double>(num_nodes) * (2.0 * beta + 1.0);
}

double defender_rr_sample_bound(std::size_t num_nodes, double beta) {
    validate_nonnegative_finite(beta, "beta");
    return 2.0 * static_cast<double>(num_nodes) * beta;
}

double quadratic_constant_sample_bound(std::size_t num_nodes) {
    return static_cast<double>(num_nodes);
}

}  // namespace rcd
