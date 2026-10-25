#pragma once

#include "rcd/conflict.hpp"
#include "rcd/types.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <vector>

namespace rcd {

enum class OptimizedPlayer {
    Attacker,
    Defender
};

struct CascadeParams {
    double alpha_A = 0.5;
    double alpha_B = 0.5;
    double default_rho_u = 0.5;
    Eigen::VectorXd rho_u_by_node;
};

struct ConditionalTerms {
    OptimizedPlayer optimized_player = OptimizedPlayer::Attacker;
    Eigen::VectorXd eta;
    Eigen::VectorXd xi;
    Eigen::VectorXd ell;
    double constant = 0.0;
};

double clip_unit_interval(double value);

Eigen::VectorXd perturb_opinions(
    const Eigen::VectorXd& innate,
    const std::vector<std::uint8_t>& reached_a,
    const std::vector<std::uint8_t>& reached_b,
    const CascadeParams& params);

double realized_conflict_value(
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<std::uint8_t>& reached_a,
    const std::vector<std::uint8_t>& reached_b,
    const CascadeParams& params);

ConditionalTerms build_conditional_terms(
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<std::uint8_t>& fixed_opponent_reached,
    const CascadeParams& params,
    OptimizedPlayer optimized_player);

Eigen::VectorXd apply_conditional_reach(
    const ConditionalTerms& terms,
    const std::vector<std::uint8_t>& optimized_reached);

double decomposed_conflict_value(
    const ConflictBackend& conflict,
    const ConditionalTerms& terms,
    const std::vector<std::uint8_t>& optimized_reached);

Eigen::VectorXd upper_surrogate_weights(const ConditionalTerms& terms);
Eigen::VectorXd lower_surrogate_weights(const ConditionalTerms& terms);

}  // namespace rcd
