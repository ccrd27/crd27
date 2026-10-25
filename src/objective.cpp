#include "rcd/objective.hpp"

#include <algorithm>
#include <stdexcept>

namespace rcd {
namespace {

void validate_reach_size(const std::vector<std::uint8_t>& reached, Eigen::Index expected, const char* name) {
    if (static_cast<Eigen::Index>(reached.size()) != expected) {
        throw std::invalid_argument(std::string(name) + " size does not match opinion vector size");
    }
}

void validate_terms_size(const ConditionalTerms& terms) {
    if (terms.eta.size() != terms.xi.size() || terms.eta.size() != terms.ell.size()) {
        throw std::invalid_argument("Conditional term vectors must have the same size");
    }
}

double reached_value(const std::vector<std::uint8_t>& reached, Eigen::Index index) {
    return reached[static_cast<std::size_t>(index)] ? 1.0 : 0.0;
}

double rho_value(const CascadeParams& params, Eigen::Index index, Eigen::Index expected_size) {
    if (params.rho_u_by_node.size() == 0) {
        return params.default_rho_u;
    }
    if (params.rho_u_by_node.size() != expected_size) {
        throw std::invalid_argument("rho_u_by_node size does not match opinion vector size");
    }
    return params.rho_u_by_node[index];
}

}  // namespace

double clip_unit_interval(double value) {
    return std::min(1.0, std::max(0.0, value));
}

Eigen::VectorXd perturb_opinions(
    const Eigen::VectorXd& innate,
    const std::vector<std::uint8_t>& reached_a,
    const std::vector<std::uint8_t>& reached_b,
    const CascadeParams& params) {
    validate_reach_size(reached_a, innate.size(), "reached_a");
    validate_reach_size(reached_b, innate.size(), "reached_b");

    Eigen::VectorXd perturbed(innate.size());
    for (Eigen::Index i = 0; i < innate.size(); ++i) {
        const double s = innate[i];
        const double r_a = reached_value(reached_a, i);
        const double r_b = reached_value(reached_b, i);
        const double rho = rho_value(params, i, innate.size());
        const double raw =
            s +
            r_a * (1.0 - rho * r_b) * params.alpha_A * (s - 0.5) +
            r_b * params.alpha_B * (0.5 - s);
        perturbed[i] = clip_unit_interval(raw);
    }
    return perturbed;
}

double realized_conflict_value(
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<std::uint8_t>& reached_a,
    const std::vector<std::uint8_t>& reached_b,
    const CascadeParams& params) {
    const Eigen::VectorXd perturbed = perturb_opinions(innate, reached_a, reached_b, params);
    return conflict.value(perturbed);
}

ConditionalTerms build_conditional_terms(
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<std::uint8_t>& fixed_opponent_reached,
    const CascadeParams& params,
    OptimizedPlayer optimized_player) {
    validate_reach_size(fixed_opponent_reached, innate.size(), "fixed_opponent_reached");
    if (conflict.num_nodes() != static_cast<std::size_t>(innate.size())) {
        throw std::invalid_argument("Conflict matrix size does not match opinion vector size");
    }

    ConditionalTerms terms;
    terms.optimized_player = optimized_player;
    terms.eta.resize(innate.size());
    terms.xi.resize(innate.size());

    for (Eigen::Index i = 0; i < innate.size(); ++i) {
        const double s = innate[i];
        const double r_opponent = reached_value(fixed_opponent_reached, i);
        const double rho = rho_value(params, i, innate.size());

        if (optimized_player == OptimizedPlayer::Attacker) {
            const double eta = clip_unit_interval(
                s + r_opponent * params.alpha_B * (0.5 - s));
            const double with_attacker = clip_unit_interval(
                s +
                (1.0 - rho * r_opponent) * params.alpha_A * (s - 0.5) +
                r_opponent * params.alpha_B * (0.5 - s));
            terms.eta[i] = eta;
            terms.xi[i] = with_attacker - eta;
        } else {
            const double eta = clip_unit_interval(
                s + r_opponent * params.alpha_A * (s - 0.5));
            const double with_defender = clip_unit_interval(
                s +
                r_opponent * (1.0 - rho) * params.alpha_A * (s - 0.5) +
                params.alpha_B * (0.5 - s));
            terms.eta[i] = eta;
            terms.xi[i] = with_defender - eta;
        }
    }

    const Eigen::VectorXd c_eta = conflict.apply(terms.eta);
    terms.ell = 2.0 * terms.xi.array() * c_eta.array();
    // Reuse C*eta: evaluating eta^T C eta through conflict.value() would
    // repeat the same sparse FJ solve(s), which is especially costly for AMGCL.
    terms.constant = terms.eta.dot(c_eta);
    return terms;
}

Eigen::VectorXd apply_conditional_reach(
    const ConditionalTerms& terms,
    const std::vector<std::uint8_t>& optimized_reached) {
    validate_terms_size(terms);
    validate_reach_size(optimized_reached, terms.eta.size(), "optimized_reached");

    Eigen::VectorXd result = terms.eta;
    for (Eigen::Index i = 0; i < result.size(); ++i) {
        if (optimized_reached[static_cast<std::size_t>(i)]) {
            result[i] += terms.xi[i];
        }
    }
    return result;
}

double decomposed_conflict_value(
    const ConflictBackend& conflict,
    const ConditionalTerms& terms,
    const std::vector<std::uint8_t>& optimized_reached) {
    validate_terms_size(terms);
    validate_reach_size(optimized_reached, terms.eta.size(), "optimized_reached");

    Eigen::VectorXd delta = Eigen::VectorXd::Zero(terms.eta.size());
    double first_order = 0.0;
    for (Eigen::Index i = 0; i < terms.eta.size(); ++i) {
        if (optimized_reached[static_cast<std::size_t>(i)]) {
            delta[i] = terms.xi[i];
            first_order += terms.ell[i];
        }
    }

    const double second_order = conflict.value(delta);
    return terms.constant + first_order + second_order;
}

Eigen::VectorXd upper_surrogate_weights(const ConditionalTerms& terms) {
    validate_terms_size(terms);
    Eigen::VectorXd weights(terms.eta.size());
    for (Eigen::Index i = 0; i < weights.size(); ++i) {
        weights[i] = std::max(0.0, terms.ell[i] + terms.xi[i] * terms.xi[i]);
    }
    return weights;
}

Eigen::VectorXd lower_surrogate_weights(const ConditionalTerms& terms) {
    validate_terms_size(terms);
    Eigen::VectorXd weights(terms.eta.size());
    for (Eigen::Index i = 0; i < weights.size(); ++i) {
        weights[i] = std::max(0.0, -terms.ell[i]);
    }
    return weights;
}

}  // namespace rcd
