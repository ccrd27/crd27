#pragma once

#include "rcd/conflict.hpp"
#include "rcd/graph.hpp"
#include "rcd/objective.hpp"
#include "rcd/propagation.hpp"
#include "rcd/progress.hpp"
#include "rcd/types.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rcd {

enum class MonteCarloEvaluationMode {
    Fixed,
    Adaptive
};

struct MonteCarloOptions {
    std::size_t simulations = 10000;
    std::uint64_t seed = 1;
    ProgressCallback progress_callback = {};
    MonteCarloEvaluationMode evaluation_mode = MonteCarloEvaluationMode::Fixed;
    std::size_t min_simulations = 500;
    std::size_t check_interval = 100;
    double ci_relative_half_width = 0.1;
};

struct MonteCarloResult {
    double mean = 0.0;
    double variance = 0.0;
    double standard_error = 0.0;
    double ci95_half_width = 0.0;
    std::size_t simulations = 0;
    std::string stop_status = "fixed_budget";
};

std::string to_string(MonteCarloEvaluationMode mode);

double evaluate_realized_conflict_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    LazyLiveEdgeOracle& oracle,
    TraversalWorkspace& attack_workspace,
    TraversalWorkspace& defense_workspace);

MonteCarloResult estimate_expected_conflict_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options);

double evaluate_realized_conflict(
    PropagationModel model,
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    LazyLiveEdgeOracle& live_oracle,
    LazyTriggeringEdgeOracle& triggering_oracle,
    TraversalWorkspace& attack_workspace,
    TraversalWorkspace& defense_workspace);

MonteCarloResult estimate_expected_conflict(
    PropagationModel model,
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options);

MonteCarloResult estimate_expected_conflict_pair_rr_dense(
    PropagationModel model,
    const DirectedGraph& graph,
    const DenseConflict& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options);

}  // namespace rcd
