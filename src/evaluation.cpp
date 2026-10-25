#include "rcd/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace rcd {
namespace {

void validate_sizes(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate) {
    if (conflict.num_nodes() != graph.num_nodes() ||
        static_cast<std::size_t>(innate.size()) != graph.num_nodes()) {
        throw std::invalid_argument("Graph, conflict backend, and opinion vector sizes must match");
    }
}

void validate_monte_carlo_options(const MonteCarloOptions& options) {
    if (options.simulations == 0) {
        throw std::invalid_argument("Monte Carlo evaluation requires at least one simulation");
    }
    if (options.evaluation_mode == MonteCarloEvaluationMode::Adaptive) {
        if (options.min_simulations < 2 || options.min_simulations > options.simulations) {
            throw std::invalid_argument(
                "Adaptive Monte Carlo requires min_simulations in [2, simulations]");
        }
        if (options.check_interval == 0) {
            throw std::invalid_argument("Adaptive Monte Carlo check interval must be positive");
        }
        if (!std::isfinite(options.ci_relative_half_width) ||
            options.ci_relative_half_width <= 0.0) {
            throw std::invalid_argument(
                "Adaptive Monte Carlo relative CI half-width must be finite and positive");
        }
    }
}

MonteCarloResult finalize_monte_carlo_result(
    double mean,
    double m2,
    std::size_t simulations,
    MonteCarloEvaluationMode mode,
    bool stopped_early) {
    MonteCarloResult result;
    result.mean = mean;
    result.simulations = simulations;
    if (simulations > 1) {
        result.variance = m2 / static_cast<double>(simulations - 1);
        result.standard_error = std::sqrt(result.variance / static_cast<double>(simulations));
        result.ci95_half_width = 1.96 * result.standard_error;
    }
    result.stop_status = mode == MonteCarloEvaluationMode::Fixed
        ? "fixed_budget"
        : (stopped_early ? "ci_target_reached" : "maximum_samples_reached");
    return result;
}

bool adaptive_ci_target_reached(
    const MonteCarloOptions& options,
    std::size_t simulations,
    double mean,
    double m2) {
    if (options.evaluation_mode != MonteCarloEvaluationMode::Adaptive ||
        simulations < options.min_simulations ||
        simulations < 2 ||
        simulations % options.check_interval != 0) {
        return false;
    }
    const double variance = m2 / static_cast<double>(simulations - 1);
    const double standard_error = std::sqrt(variance / static_cast<double>(simulations));
    return 1.96 * standard_error <=
        options.ci_relative_half_width * std::abs(mean);
}

std::size_t monte_carlo_progress_interval(const MonteCarloOptions& options) {
    if (options.evaluation_mode == MonteCarloEvaluationMode::Adaptive) {
        return options.check_interval;
    }
    return std::max<std::size_t>(1, options.simulations / 10);
}

void report_monte_carlo_start(const MonteCarloOptions& options) {
    if (!options.progress_callback) {
        return;
    }
    std::string stage =
        "Monte Carlo simulation started (maximum_samples=" +
        std::to_string(options.simulations);
    if (options.evaluation_mode == MonteCarloEvaluationMode::Adaptive) {
        stage +=
            ", minimum_samples=" + std::to_string(options.min_simulations) +
            ", check_interval=" + std::to_string(options.check_interval);
    }
    stage += ")";
    options.progress_callback(stage, 0.0);
}

void report_monte_carlo_progress(
    const MonteCarloOptions& options,
    std::size_t completed,
    bool confidence_target_reached) {
    if (!options.progress_callback) {
        return;
    }
    const std::string sample_status =
        "completed_samples=" + std::to_string(completed) +
        ", maximum_samples=" + std::to_string(options.simulations);
    if (confidence_target_reached) {
        options.progress_callback(
            "Monte Carlo confidence target reached (" + sample_status + ")",
            100.0);
        return;
    }
    const std::size_t interval = monte_carlo_progress_interval(options);
    if (completed % interval == 0 || completed == options.simulations) {
        options.progress_callback(
            "Monte Carlo simulation (" + sample_status + ")",
            100.0 * static_cast<double>(completed) /
                static_cast<double>(options.simulations));
    }
}

std::vector<std::uint8_t> make_reached_mask(
    std::size_t num_nodes,
    const std::vector<NodeId>& reached) {
    std::vector<std::uint8_t> mask(num_nodes, 0);
    for (const auto node : reached) {
        if (static_cast<std::size_t>(node) >= num_nodes) {
            throw std::out_of_range("Reached node is out of range");
        }
        mask[node] = 1;
    }
    return mask;
}

std::vector<std::uint8_t> make_seed_mask(
    std::size_t num_nodes,
    const std::vector<NodeId>& seeds) {
    std::vector<std::uint8_t> mask(num_nodes, 0);
    for (const auto seed : seeds) {
        if (static_cast<std::size_t>(seed) >= num_nodes) {
            throw std::out_of_range("Seed node is out of range");
        }
        mask[seed] = 1;
    }
    return mask;
}

bool rr_set_intersects_mask(
    const std::vector<NodeId>& rr_set,
    const std::vector<std::uint8_t>& seed_mask) {
    for (const auto node : rr_set) {
        if (static_cast<std::size_t>(node) >= seed_mask.size()) {
            throw std::out_of_range("RR-set node is out of range");
        }
        if (seed_mask[node] != 0) {
            return true;
        }
    }
    return false;
}

double rho_value(const CascadeParams& params, NodeId node, std::size_t expected_size) {
    if (params.rho_u_by_node.size() == 0) {
        return params.default_rho_u;
    }
    if (static_cast<std::size_t>(params.rho_u_by_node.size()) != expected_size) {
        throw std::invalid_argument("rho_u_by_node size does not match opinion vector size");
    }
    return params.rho_u_by_node[static_cast<Eigen::Index>(node)];
}

double perturbed_opinion_at(
    const Eigen::VectorXd& innate,
    NodeId node,
    bool attack_reached,
    bool defense_reached,
    const CascadeParams& params) {
    if (static_cast<Eigen::Index>(node) >= innate.size()) {
        throw std::out_of_range("Node id is out of range");
    }
    const double s = innate[static_cast<Eigen::Index>(node)];
    const double r_a = attack_reached ? 1.0 : 0.0;
    const double r_b = defense_reached ? 1.0 : 0.0;
    const double rho = rho_value(params, node, static_cast<std::size_t>(innate.size()));
    return clip_unit_interval(
        s +
        r_a * (1.0 - rho * r_b) * params.alpha_A * (s - 0.5) +
        r_b * params.alpha_B * (0.5 - s));
}

}  // namespace

std::string to_string(MonteCarloEvaluationMode mode) {
    switch (mode) {
    case MonteCarloEvaluationMode::Fixed:
        return "fixed";
    case MonteCarloEvaluationMode::Adaptive:
        return "adaptive";
    }
    return "unknown";
}

double evaluate_realized_conflict_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    LazyLiveEdgeOracle& oracle,
    TraversalWorkspace& attack_workspace,
    TraversalWorkspace& defense_workspace) {
    validate_sizes(graph, conflict, innate);

    const std::vector<NodeId> attack_reached = forward_reachable_ic(
        graph, attack_seeds, oracle, attack_workspace);
    const std::vector<NodeId> defense_reached = forward_reachable_ic(
        graph, defense_seeds, oracle, defense_workspace);

    const auto attack_mask = make_reached_mask(graph.num_nodes(), attack_reached);
    const auto defense_mask = make_reached_mask(graph.num_nodes(), defense_reached);
    return realized_conflict_value(conflict, innate, attack_mask, defense_mask, params);
}

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
    TraversalWorkspace& defense_workspace) {
    validate_sizes(graph, conflict, innate);

    const std::vector<NodeId> attack_reached = forward_reachable(
        model, graph, attack_seeds, live_oracle, triggering_oracle, attack_workspace);
    const std::vector<NodeId> defense_reached = forward_reachable(
        model, graph, defense_seeds, live_oracle, triggering_oracle, defense_workspace);

    const auto attack_mask = make_reached_mask(graph.num_nodes(), attack_reached);
    const auto defense_mask = make_reached_mask(graph.num_nodes(), defense_reached);
    return realized_conflict_value(conflict, innate, attack_mask, defense_mask, params);
}

MonteCarloResult estimate_expected_conflict_ic(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options) {
    validate_sizes(graph, conflict, innate);
    validate_monte_carlo_options(options);
    report_monte_carlo_start(options);

    LazyLiveEdgeOracle oracle(graph.num_edges(), options.seed ^ 0x510e527fade682d1ULL);
    TraversalWorkspace attack_workspace(graph.num_nodes());
    TraversalWorkspace defense_workspace(graph.num_nodes());

    double mean = 0.0;
    double m2 = 0.0;
    std::size_t completed = 0;
    bool stopped_early = false;
    for (std::size_t i = 0; i < options.simulations; ++i) {
        const double value = evaluate_realized_conflict_ic(
            graph,
            conflict,
            innate,
            attack_seeds,
            defense_seeds,
            params,
            oracle,
            attack_workspace,
            defense_workspace);
        oracle.reset();

        const double count = static_cast<double>(i + 1);
        const double delta = value - mean;
        mean += delta / count;
        const double delta2 = value - mean;
        m2 += delta * delta2;
        completed = i + 1;
        if (adaptive_ci_target_reached(options, completed, mean, m2)) {
            stopped_early = true;
            report_monte_carlo_progress(options, completed, true);
            break;
        }
        report_monte_carlo_progress(options, completed, false);
    }
    return finalize_monte_carlo_result(
        mean, m2, completed, options.evaluation_mode, stopped_early);
}

MonteCarloResult estimate_expected_conflict(
    PropagationModel model,
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options) {
    validate_sizes(graph, conflict, innate);
    validate_monte_carlo_options(options);
    report_monte_carlo_start(options);

    LazyLiveEdgeOracle live_oracle(graph.num_edges(), options.seed ^ 0x510e527fade682d1ULL);
    LazyTriggeringEdgeOracle triggering_oracle(graph.num_nodes(), options.seed ^ 0x9b05688c2b3e6c1fULL);
    TraversalWorkspace attack_workspace(graph.num_nodes());
    TraversalWorkspace defense_workspace(graph.num_nodes());

    double mean = 0.0;
    double m2 = 0.0;
    std::size_t completed = 0;
    bool stopped_early = false;
    for (std::size_t i = 0; i < options.simulations; ++i) {
        const double value = evaluate_realized_conflict(
            model,
            graph,
            conflict,
            innate,
            attack_seeds,
            defense_seeds,
            params,
            live_oracle,
            triggering_oracle,
            attack_workspace,
            defense_workspace);
        live_oracle.reset();
        triggering_oracle.reset();

        const double count = static_cast<double>(i + 1);
        const double delta = value - mean;
        mean += delta / count;
        const double delta2 = value - mean;
        m2 += delta * delta2;
        completed = i + 1;
        if (adaptive_ci_target_reached(options, completed, mean, m2)) {
            stopped_early = true;
            report_monte_carlo_progress(options, completed, true);
            break;
        }
        report_monte_carlo_progress(options, completed, false);
    }
    return finalize_monte_carlo_result(
        mean, m2, completed, options.evaluation_mode, stopped_early);
}

MonteCarloResult estimate_expected_conflict_pair_rr_dense(
    PropagationModel model,
    const DirectedGraph& graph,
    const DenseConflict& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<NodeId>& attack_seeds,
    const std::vector<NodeId>& defense_seeds,
    const CascadeParams& params,
    const MonteCarloOptions& options) {
    validate_sizes(graph, conflict, innate);
    if (graph.num_nodes() == 0) {
        throw std::invalid_argument("Pair-RR empirical evaluation requires a nonempty graph");
    }
    if (options.simulations == 0) {
        throw std::invalid_argument("Pair-RR empirical evaluation requires at least one sample");
    }

    const auto attack_seed_mask = make_seed_mask(graph.num_nodes(), attack_seeds);
    const auto defense_seed_mask = make_seed_mask(graph.num_nodes(), defense_seeds);

    std::mt19937_64 root_rng(options.seed ^ 0x6a09e667f3bcc909ULL);
    std::uniform_int_distribution<std::size_t> root_dist(0, graph.num_nodes() - 1);
    LazyLiveEdgeOracle live_oracle(graph.num_edges(), options.seed ^ 0xbb67ae8584caa73bULL);
    LazyTriggeringEdgeOracle triggering_oracle(graph.num_nodes(), options.seed ^ 0x3c6ef372fe94f82bULL);
    TraversalWorkspace first_rr_workspace(graph.num_nodes());
    TraversalWorkspace second_rr_workspace(graph.num_nodes());

    double mean = 0.0;
    double m2 = 0.0;
    const double n_squared =
        static_cast<double>(graph.num_nodes()) * static_cast<double>(graph.num_nodes());
    const auto& c = conflict.conflict_matrix();

    for (std::size_t i = 0; i < options.simulations; ++i) {
        const NodeId first_root = static_cast<NodeId>(root_dist(root_rng));
        const NodeId second_root = static_cast<NodeId>(root_dist(root_rng));

        std::vector<NodeId> first_rr_set = reverse_rr_set(
            model,
            graph,
            first_root,
            live_oracle,
            triggering_oracle,
            first_rr_workspace);
        std::vector<NodeId> second_rr_set = reverse_rr_set(
            model,
            graph,
            second_root,
            live_oracle,
            triggering_oracle,
            second_rr_workspace);

        const bool first_attack_reached =
            rr_set_intersects_mask(first_rr_set, attack_seed_mask);
        const bool first_defense_reached =
            rr_set_intersects_mask(first_rr_set, defense_seed_mask);
        const bool second_attack_reached =
            rr_set_intersects_mask(second_rr_set, attack_seed_mask);
        const bool second_defense_reached =
            rr_set_intersects_mask(second_rr_set, defense_seed_mask);

        const double first_opinion = perturbed_opinion_at(
            innate, first_root, first_attack_reached, first_defense_reached, params);
        const double second_opinion = perturbed_opinion_at(
            innate, second_root, second_attack_reached, second_defense_reached, params);
        const double value =
            n_squared *
            c(static_cast<Eigen::Index>(first_root), static_cast<Eigen::Index>(second_root)) *
            first_opinion *
            second_opinion;

        live_oracle.reset();
        triggering_oracle.reset();

        const double count = static_cast<double>(i + 1);
        const double delta = value - mean;
        mean += delta / count;
        const double delta2 = value - mean;
        m2 += delta * delta2;
    }

    MonteCarloResult result;
    result.mean = mean;
    result.simulations = options.simulations;
    if (options.simulations > 1) {
        result.variance = m2 / static_cast<double>(options.simulations - 1);
        result.standard_error = std::sqrt(result.variance / static_cast<double>(options.simulations));
    }
    return result;
}

}  // namespace rcd
