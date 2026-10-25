#include "rcd/baselines.hpp"
#include "rcd/config.hpp"
#include "rcd/certificate.hpp"
#include "rcd/conflict.hpp"
#include "rcd/crd.hpp"
#include "rcd/evaluation.hpp"
#include "rcd/graph.hpp"
#include "rcd/objective.hpp"
#include "rcd/progress.hpp"
#include "rcd/results_writer.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

std::uint64_t current_rss_kib() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    std::uint64_t total_pages = 0;
    std::uint64_t resident_pages = 0;
    if (!(statm >> total_pages >> resident_pages)) {
        return 0;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return 0;
    }
    return resident_pages * static_cast<std::uint64_t>(page_size) / 1024ULL;
#else
    return 0;
#endif
}

std::uint64_t peak_rss_kib() {
#if defined(__linux__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss) / 1024ULL;
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#endif
#else
    return 0;
#endif
}

rcd::MonteCarloOptions make_evaluation_mc_options(
    const rcd::ProgramOptions& options,
    std::uint64_t seed,
    const std::string& scenario) {
    rcd::MonteCarloOptions result;
    result.simulations = options.mc_simulations;
    result.seed = seed;
    result.evaluation_mode = options.mc_evaluation_mode == "adaptive"
        ? rcd::MonteCarloEvaluationMode::Adaptive
        : rcd::MonteCarloEvaluationMode::Fixed;
    result.min_simulations = options.mc_min_simulations;
    result.check_interval = options.mc_check_interval;
    result.ci_relative_half_width = options.mc_ci_relative_half_width;
    result.progress_callback = [scenario](const std::string& stage, double percent) {
        rcd::print_progress_line(
            std::cout,
            "Evaluation: " + scenario + ": " + stage,
            percent);
    };
    return result;
}

struct FinalAttackResponse {
    std::vector<rcd::NodeId> seeds;
    std::string policy;
    std::size_t rr_samples = 0;
};

rcd::ProgressCallback make_final_attack_response_progress(const std::string& scenario) {
    return [scenario](const std::string& stage, double percent) {
        rcd::print_progress_line(
            std::cout,
            "Evaluation: " + scenario + ": final attack response: " + stage,
            percent);
    };
}

FinalAttackResponse select_final_attack_response(
    const rcd::DirectedGraph& graph,
    const rcd::ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const std::vector<rcd::NodeId>& defense_seeds,
    const rcd::CascadeParams& params,
    const rcd::ProgramOptions& program_options,
    std::uint64_t seed,
    const std::string& scenario) {
    const auto propagation_model = rcd::parse_propagation_model(program_options.propagation_model);
    const auto progress = make_final_attack_response_progress(scenario);

    if (conflict.is_approximate()) {
        const rcd::LinearAttackResponseOptions options{
            program_options.k_a,
            program_options.attack_rr_samples,
            seed,
            propagation_model,
            progress
        };
        const auto response = rcd::run_linear_attack_response(
            graph, conflict, innate, defense_seeds, params, options);
        return FinalAttackResponse{
            response.attack_seeds,
            "linear_rr_reference_context",
            response.generated_rr_samples
        };
    }

    rcd::AttackOracleOptions options{
        program_options.k_a,
        program_options.attack_rr_samples,
        seed,
        program_options.epsilon_cert,
        program_options.failure_probability,
        program_options.max_rr_samples,
        propagation_model
    };
    options.progress_callback = progress;
    const auto response = rcd::run_attack_oracle(
        graph, conflict, innate, defense_seeds, params, options);
    return FinalAttackResponse{
        response.attack_seeds,
        "full_conditional_weighted_rr",
        response.generated_rr_samples
    };
}

rcd::CRDEvaluationSummary evaluate_final_solution(
    const rcd::DirectedGraph& graph,
    const rcd::ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const rcd::CascadeParams& params,
    const rcd::CRDOptions& crd_options,
    const rcd::ProgramOptions& program_options,
    rcd::CRDResult& crd_result) {
    rcd::CRDEvaluationSummary summary;

    // All methods use the same backend-specific final response policy. Dense
    // evaluation uses the full conditional oracle, while approximate backends
    // use the common scalable linear-RR response.
    const auto final_attack = select_final_attack_response(
        graph,
        conflict,
        innate,
        crd_result.defense_seeds,
        params,
        program_options,
        program_options.seed ^ 0x9e3779b97f4a7c15ULL,
        "defense-vs-attack");
    crd_result.attack_seeds = final_attack.seeds;
    summary.attack_response_policy = final_attack.policy;
    summary.attack_response_rr_samples = final_attack.rr_samples;

    const auto mc_options = make_evaluation_mc_options(
        program_options,
        program_options.seed ^ 0x3c6ef372fe94f82bULL,
        "defense-vs-attack");

    summary.defense_vs_attack = rcd::estimate_expected_conflict(
        crd_options.propagation_model,
        graph,
        conflict,
        innate,
        crd_result.attack_seeds,
        crd_result.defense_seeds,
        params,
        mc_options);

    const auto attack_only = select_final_attack_response(
        graph,
        conflict,
        innate,
        {},
        params,
        program_options,
        program_options.seed ^ 0xbb67ae8584caa73bULL,
        "attack-only");
    summary.attack_only_seeds = attack_only.seeds;
    summary.no_defense_attack_only = rcd::estimate_expected_conflict(
        crd_options.propagation_model,
        graph,
        conflict,
        innate,
        summary.attack_only_seeds,
        {},
        params,
        make_evaluation_mc_options(
            program_options,
            program_options.seed ^ 0xa54ff53a5f1d36f1ULL,
            "attack-only"));

    summary.defense_only = rcd::estimate_expected_conflict(
        crd_options.propagation_model,
        graph,
        conflict,
        innate,
        {},
        crd_result.defense_seeds,
        params,
        make_evaluation_mc_options(
            program_options,
            program_options.seed ^ 0x510e527fade682d1ULL,
            "defense-only"));
    summary.available = true;
    return summary;
}

rcd::CRDEvaluationSummary evaluate_baseline_solution(
    const rcd::DirectedGraph& graph,
    const rcd::ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const rcd::CascadeParams& params,
    const rcd::ProgramOptions& program_options,
    rcd::BaselineResult& baseline_result) {
    rcd::CRDEvaluationSummary summary;
    const auto propagation_model = rcd::parse_propagation_model(program_options.propagation_model);

    if (baseline_result.algorithm != rcd::BaselineAlgorithm::Exact) {
        const auto final_attack = select_final_attack_response(
            graph,
            conflict,
            innate,
            baseline_result.defense_seeds,
            params,
            program_options,
            program_options.seed ^ 0x9e3779b97f4a7c15ULL,
            "defense-vs-attack");
        baseline_result.attack_seeds = final_attack.seeds;
        summary.attack_response_policy = final_attack.policy;
        summary.attack_response_rr_samples = final_attack.rr_samples;
    } else {
        summary.attack_response_policy = "exact_enumeration_attack";
    }

    summary.defense_vs_attack = rcd::estimate_expected_conflict(
        propagation_model,
        graph,
        conflict,
        innate,
        baseline_result.attack_seeds,
        baseline_result.defense_seeds,
        params,
        make_evaluation_mc_options(
            program_options,
            program_options.seed ^ 0x3c6ef372fe94f82bULL,
            "defense-vs-attack"));

    const auto attack_only = select_final_attack_response(
        graph,
        conflict,
        innate,
        {},
        params,
        program_options,
        program_options.seed ^ 0xbb67ae8584caa73bULL,
        "attack-only");
    summary.attack_only_seeds = attack_only.seeds;
    summary.no_defense_attack_only = rcd::estimate_expected_conflict(
        propagation_model,
        graph,
        conflict,
        innate,
        summary.attack_only_seeds,
        {},
        params,
        make_evaluation_mc_options(
            program_options,
            program_options.seed ^ 0xa54ff53a5f1d36f1ULL,
            "attack-only"));

    summary.defense_only = rcd::estimate_expected_conflict(
        propagation_model,
        graph,
        conflict,
        innate,
        {},
        baseline_result.defense_seeds,
        params,
        make_evaluation_mc_options(
            program_options,
            program_options.seed ^ 0x510e527fade682d1ULL,
            "defense-only"));
    summary.available = true;
    return summary;
}

rcd::ExperimentMetadata make_metadata(
    const rcd::ProgramOptions& options,
    const rcd::DirectedGraph& graph,
    const rcd::UndirectedFJGraph& fj_graph,
    const rcd::ConflictBackend& conflict) {
    rcd::ExperimentMetadata metadata;
    metadata.dataset_path = options.dataset_path;
    metadata.output_dir = options.output_dir;
    metadata.output_prefix = options.output_prefix;
    metadata.innate_opinion_file = options.innate_opinion_file;
    metadata.edge_weight_file = options.edge_weight_file.empty() ? "default_weighted_cascade" : options.edge_weight_file;
    metadata.rho_file = options.rho_file.empty() ? "default_constant_rho" : options.rho_file;
    metadata.fj_weight_file = options.fj_weight_file.empty() ? "default_unit_weight" : options.fj_weight_file;
    metadata.algorithm = options.algorithm;
    metadata.metric = options.metric;
    metadata.propagation_model = options.propagation_model;
    metadata.conflict_backend = options.conflict_backend;
    metadata.conflict_backend_approximate = conflict.is_approximate();
    metadata.conflict_beta_bound = rcd::conflict_beta(conflict);
    metadata.seed = options.seed;
    metadata.graph_nodes = graph.num_nodes();
    metadata.graph_directed_edges = graph.num_edges();
    metadata.graph_skipped_self_loops = graph.skipped_self_loops;
    metadata.graph_skipped_duplicate_edges = graph.skipped_duplicate_edges;
    metadata.fj_undirected_edges = fj_graph.num_edges();
    metadata.mc_simulations = options.mc_simulations;
    metadata.mc_evaluation_mode = options.mc_evaluation_mode;
    metadata.mc_min_simulations = options.mc_min_simulations;
    metadata.mc_check_interval = options.mc_check_interval;
    metadata.mc_ci_relative_half_width = options.mc_ci_relative_half_width;
    metadata.k_a = options.k_a;
    metadata.k_b = options.k_b;
    metadata.crd_mode = options.crd_mode;
    metadata.crd_variant = rcd::is_crd_family_algorithm(options.algorithm)
        ? rcd::to_string(rcd::parse_crd_variant(options.algorithm))
        : "not_applicable";
    metadata.crd_iterations = options.crd_iterations;
    metadata.paper_crd_schedule = options.crd_mode == "paper-exact"
        ? "single_layer"
        : "not_applicable";
    metadata.initial_attack_pool_policy = "not_recorded";
    metadata.baseline_iterations = options.baseline_iterations;
    metadata.fjc_pair_rr_samples = options.algorithm == "fjc-iter"
        ? options.fjc_pair_rr_samples
        : 0;
    metadata.fjc_linear_rr_samples = options.algorithm == "fjc-linear-iter"
        ? options.fjc_linear_rr_samples
        : 0;
    metadata.hist_rr_samples_floor =
        (options.algorithm == "hist-iter" || options.algorithm == "hist")
        ? options.hist_rr_samples_floor
        : 0;
    metadata.baseline_max_adaptive_rounds = options.baseline_max_adaptive_rounds;
    metadata.attack_rr_samples = options.attack_rr_samples;
    metadata.defense_rr_samples = options.defense_rr_samples;
    metadata.max_rr_samples = options.max_rr_samples;
    metadata.empirical_samples = options.empirical_samples;
    metadata.mixed_distribution = options.mixed_distribution;
    metadata.mixed_rounds = options.mixed_rounds;
    metadata.mixed_learning_rate = options.mixed_learning_rate;
    metadata.alpha_a = options.alpha_a;
    metadata.alpha_b = options.alpha_b;
    metadata.rho = options.rho;
    metadata.epsilon_cert = options.epsilon_cert;
    metadata.failure_probability = options.failure_probability;
    if (const auto* amgcl = dynamic_cast<const rcd::AmgclConflict*>(&conflict)) {
        metadata.amgcl_tolerance = amgcl->tolerance();
        metadata.amgcl_max_iterations = amgcl->max_iterations();
        metadata.amgcl_last_iterations = amgcl->last_iterations();
        metadata.amgcl_last_relative_error = amgcl->last_relative_error();
    }
    return metadata;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto options = rcd::parse_program_options(argc, argv);
        if (options.show_help) {
            rcd::print_usage(std::cout, argv[0]);
            return 0;
        }

        rcd::ensure_output_directory(options.output_dir);
        rcd::print_progress_line(std::cout, "Input: reading graph", 5.0);
        auto graph = rcd::read_directed_edge_list(options.dataset_path);
        options.k_a = rcd::resolved_attack_budget(options, graph.num_nodes());
        options.k_b = rcd::resolved_defense_budget(options, graph.num_nodes());
        rcd::print_options(std::cout, options);
        if (!options.edge_weight_file.empty()) {
            rcd::apply_edge_probability_file(graph, options.edge_weight_file);
        }
        auto fj_graph = rcd::build_fj_graph(graph);
        rcd::print_progress_line(std::cout, "Input: graph and innate opinions loaded", 20.0);
        if (!options.fj_weight_file.empty()) {
            rcd::apply_fj_weight_file(fj_graph, graph, options.fj_weight_file);
        }
        const Eigen::VectorXd innate = rcd::read_node_value_file(
            graph, options.innate_opinion_file, "innate opinion", 0.0, 1.0);
        Eigen::VectorXd rho_by_node;
        if (!options.rho_file.empty()) {
            rho_by_node = rcd::read_node_value_file(
                graph, options.rho_file, "rho", 0.0, 1.0);
        }

        std::cout << "Directed propagation graph\n";
        std::cout << "  nodes: " << graph.num_nodes() << '\n';
        std::cout << "  directed_edges: " << graph.num_edges() << '\n';
        std::cout << "  skipped_self_loops: " << graph.skipped_self_loops << '\n';
        std::cout << "  skipped_duplicate_edges: " << graph.skipped_duplicate_edges << '\n';
        std::cout << "FJ undirected graph\n";
        std::cout << "  nodes: " << fj_graph.num_nodes << '\n';
        std::cout << "  undirected_edges: " << fj_graph.num_edges() << '\n';
        const auto metric = rcd::parse_conflict_metric(options.metric);
        rcd::print_progress_line(std::cout, "Model: FJ graph constructed", 25.0);

        const auto run_full_crd = [&](const rcd::ConflictBackend& conflict) {
            rcd::CascadeParams params{
                options.alpha_a,
                options.alpha_b,
                options.rho
            };
            params.rho_u_by_node = rho_by_node;
            rcd::CRDOptions crd_options{
                options.crd_iterations,
                options.k_b,
                options.k_a,
                options.attack_rr_samples,
                options.defense_rr_samples,
                options.empirical_samples,
                options.seed,
                rcd::resolved_epsilon(options, graph.num_nodes()),
                rcd::resolved_epsilon(options, graph.num_nodes()),
                options.failure_probability,
                rcd::parse_mixed_distribution_mode(options.mixed_distribution),
                options.mixed_rounds,
                options.mixed_learning_rate,
                options.max_rr_samples,
                rcd::parse_propagation_model(options.propagation_model),
                rcd::parse_crd_mode(options.crd_mode),
                rcd::parse_crd_variant(options.algorithm)
            };
            crd_options.engineering_holdout_samples = options.empirical_samples;
            if (crd_options.variant == rcd::CRDVariant::SurrogateOnly) {
                crd_options.defense_candidate_policy = rcd::DefenseCandidatePolicy::SurrogateOnly;
                crd_options.mixed_distribution_mode = rcd::MixedDistributionMode::Uniform;
                crd_options.empirical_samples = 0;
            } else if (crd_options.variant == rcd::CRDVariant::AdaptiveMix) {
                crd_options.mixed_distribution_mode = rcd::MixedDistributionMode::Adaptive;
            }

            std::cout << (crd_options.variant == rcd::CRDVariant::LinearApprox
                              ? "CRD-Linear outer loop\n"
                              : "CRD outer loop\n");
            rcd::print_progress_line(std::cout, "Backend: conflict solver initialized", 30.0);
            const auto rss_before_algorithm = current_rss_kib();
            auto crd_result = crd_options.variant == rcd::CRDVariant::LinearApprox
                ? rcd::run_crd_linear(graph, conflict, innate, params, crd_options, &std::cout)
                : rcd::run_crd(graph, conflict, innate, params, crd_options, &std::cout);
            const auto rss_after_algorithm = current_rss_kib();
            const auto peak_rss_after_algorithm = peak_rss_kib();
            std::cout << "Monte Carlo evaluation\n" << std::flush;
            crd_result.evaluation = evaluate_final_solution(
                graph, conflict, innate, params, crd_options, options, crd_result);
            auto metadata = make_metadata(options, graph, fj_graph, conflict);
            metadata.final_attack_response = crd_result.evaluation.attack_response_policy;
            metadata.final_attack_response_rr_samples =
                crd_result.evaluation.attack_response_rr_samples;
            metadata.epsilon_cert = rcd::resolved_epsilon(options, graph.num_nodes());
            metadata.empirical_samples = crd_options.empirical_samples;
            metadata.mixed_distribution = rcd::to_string(crd_options.mixed_distribution_mode);
            metadata.initial_attack_pool_policy = crd_result.initial_attack_pool_policy;
            metadata.memory_rss_before_algorithm_kib = rss_before_algorithm;
            metadata.memory_rss_after_algorithm_kib = rss_after_algorithm;
            metadata.memory_peak_rss_kib = peak_rss_after_algorithm;
            std::cout << "CRD result\n";
            std::cout << "  crd_variant: " << rcd::to_string(crd_result.variant) << '\n';
            std::cout << "  iterations: " << crd_result.iterations.size() << '\n';
            std::cout << "  initial_attack_pool_size: " << crd_result.initial_attack_pool_size << '\n';
            std::cout << "  attack_pool_size: " << crd_result.attack_pool.size() << '\n';
            if (crd_result.variant == rcd::CRDVariant::LinearApprox) {
                std::cout << "  certificate_status: not_applicable_linear_approx\n";
            }
            std::cout << "  best_lower_certificate: "
                      << rcd::format_decimal(crd_result.best_lower_certificate) << '\n';
            std::cout << "  best_upper_certificate: "
                      << rcd::format_decimal(crd_result.best_upper_certificate) << '\n';
            std::cout << "  best_certificate_gap: "
                      << rcd::format_decimal(crd_result.best_certificate_gap) << '\n';
            std::cout << "  final_selection_policy: "
                      << crd_result.final_selection_policy << '\n';
            std::cout << "  final_selected_iteration: "
                      << crd_result.final_selected_iteration << '\n';
            std::cout << "  final_holdout_samples: "
                      << crd_result.final_holdout_samples << '\n';
            std::cout << "  final_holdout_streaming: "
                      << (crd_result.final_holdout_streaming ? "true" : "false") << '\n';
            std::cout << "  final_selected_holdout_value: "
                      << rcd::format_decimal(crd_result.final_selected_holdout_value) << '\n';
            std::cout << "  algorithm_runtime_sec: "
                      << rcd::format_decimal(crd_result.algorithm_runtime_sec) << '\n';
            std::cout << "  mc_value_defense_vs_attack: "
                      << rcd::format_decimal(crd_result.evaluation.defense_vs_attack.mean) << '\n';
            std::cout << "  mc_value_no_defense_attack_only: "
                      << rcd::format_decimal(crd_result.evaluation.no_defense_attack_only.mean) << '\n';
            std::cout << "  mc_value_defense_only: "
                      << rcd::format_decimal(crd_result.evaluation.defense_only.mean) << '\n';
            std::cout << "  defense_seeds: "
                      << rcd::format_seed_set_preview(crd_result.defense_seeds) << '\n';
            std::cout << "  attack_seeds: "
                      << rcd::format_seed_set_preview(crd_result.attack_seeds) << '\n';
            std::cout << "  attack_only_seeds: "
                      << rcd::format_seed_set_preview(
                             crd_result.evaluation.attack_only_seeds) << '\n';
            std::cout << "  memory_rss_before_algorithm_kib: "
                      << metadata.memory_rss_before_algorithm_kib << '\n';
            std::cout << "  memory_rss_after_algorithm_kib: "
                      << metadata.memory_rss_after_algorithm_kib << '\n';
            std::cout << "  memory_peak_rss_kib: " << metadata.memory_peak_rss_kib << '\n';

            const auto output_paths = rcd::make_result_output_paths(
                options.output_dir, options.output_prefix);
            rcd::write_crd_result_files(crd_result, graph, output_paths, metadata);
            std::cout << "Result files\n";
            std::cout << "  txt: " << output_paths.txt_path << '\n';
            std::cout << "  csv: " << output_paths.csv_path << '\n';
            std::cout << "  seeds: " << output_paths.seeds_path << '\n';
            rcd::print_progress_line(std::cout, "Output: result files written", 100.0);
        };

        const auto run_full_baseline = [&](const rcd::ConflictBackend& conflict) {
            rcd::CascadeParams params{
                options.alpha_a,
                options.alpha_b,
                options.rho
            };
            params.rho_u_by_node = rho_by_node;
            rcd::BaselineOptions baseline_options{
                rcd::parse_baseline_algorithm(options.algorithm),
                options.k_a,
                options.k_b,
                options.baseline_iterations,
                options.fjc_pair_rr_samples,
                options.hist_rr_samples_floor,
                options.mc_simulations,
                options.seed,
                rcd::parse_propagation_model(options.propagation_model)
            };
            baseline_options.epsilon =
                (options.algorithm == "hist-iter" || options.algorithm == "hist")
                ? options.epsilon_cert
                : rcd::resolved_epsilon(options, graph.num_nodes());
            baseline_options.validation_failure_probability = options.failure_probability;
            baseline_options.max_adaptive_rounds = options.baseline_max_adaptive_rounds;
            baseline_options.max_rr_samples = options.max_rr_samples;
            baseline_options.fjc_linear_rr_samples = options.fjc_linear_rr_samples;

            rcd::print_progress_line(std::cout, "Backend: conflict solver initialized", 30.0);
            std::cout << "Baseline selection\n";
            const auto rss_before_algorithm = current_rss_kib();
            auto baseline_result = rcd::run_baseline(
                graph, conflict, innate, params, baseline_options, &std::cout);
            const auto rss_after_algorithm = current_rss_kib();
            const auto peak_rss_after_algorithm = peak_rss_kib();
            std::cout << "Monte Carlo evaluation\n" << std::flush;
            baseline_result.evaluation = evaluate_baseline_solution(
                graph, conflict, innate, params, options, baseline_result);
            auto metadata = make_metadata(options, graph, fj_graph, conflict);
            metadata.final_attack_response = baseline_result.evaluation.attack_response_policy;
            metadata.final_attack_response_rr_samples =
                baseline_result.evaluation.attack_response_rr_samples;
            metadata.epsilon_cert = rcd::resolved_epsilon(options, graph.num_nodes());
            metadata.memory_rss_before_algorithm_kib = rss_before_algorithm;
            metadata.memory_rss_after_algorithm_kib = rss_after_algorithm;
            metadata.memory_peak_rss_kib = peak_rss_after_algorithm;
            std::cout << "Baseline result\n";
            std::cout << "  algorithm: " << rcd::to_string(baseline_result.algorithm) << '\n';
            std::cout << "  iterations: " << baseline_result.iterations.size() << '\n';
            std::cout << "  selection_objective: " << baseline_result.selection_objective << '\n';
            std::cout << "  algorithm_runtime_sec: "
                      << rcd::format_decimal(baseline_result.algorithm_runtime_sec) << '\n';
            std::cout << "  mc_value_defense_vs_attack: "
                      << rcd::format_decimal(baseline_result.evaluation.defense_vs_attack.mean) << '\n';
            std::cout << "  mc_value_no_defense_attack_only: "
                      << rcd::format_decimal(baseline_result.evaluation.no_defense_attack_only.mean) << '\n';
            std::cout << "  mc_value_defense_only: "
                      << rcd::format_decimal(baseline_result.evaluation.defense_only.mean) << '\n';
            std::cout << "  defense_seeds: "
                      << rcd::format_seed_set_preview(baseline_result.defense_seeds) << '\n';
            std::cout << "  attack_seeds: "
                      << rcd::format_seed_set_preview(baseline_result.attack_seeds) << '\n';
            std::cout << "  attack_only_seeds: "
                      << rcd::format_seed_set_preview(
                             baseline_result.evaluation.attack_only_seeds) << '\n';
            std::cout << "  memory_rss_before_algorithm_kib: "
                      << metadata.memory_rss_before_algorithm_kib << '\n';
            std::cout << "  memory_rss_after_algorithm_kib: "
                      << metadata.memory_rss_after_algorithm_kib << '\n';
            std::cout << "  memory_peak_rss_kib: " << metadata.memory_peak_rss_kib << '\n';

            const auto output_paths = rcd::make_result_output_paths(
                options.output_dir, options.output_prefix);
            rcd::write_baseline_result_files(baseline_result, graph, output_paths, metadata);
            std::cout << "Result files\n";
            std::cout << "  txt: " << output_paths.txt_path << '\n';
            std::cout << "  csv: " << output_paths.csv_path << '\n';
            std::cout << "  seeds: " << output_paths.seeds_path << '\n';
            rcd::print_progress_line(std::cout, "Output: result files written", 100.0);
        };

        if (options.conflict_backend == "dense") {
            const auto conflict = rcd::DenseConflict::build(fj_graph, metric);
            const Eigen::VectorXd constant_opinion = Eigen::VectorXd::Ones(
                static_cast<Eigen::Index>(conflict.num_nodes()));
            std::cout << "Dense conflict backend\n";
            std::cout << "  metric: " << rcd::to_string(metric) << '\n';
            std::cout << "  matrix_size: " << conflict.num_nodes() << " x " << conflict.num_nodes() << '\n';
            std::cout << "  constant_vector_value: "
                      << rcd::format_decimal(conflict.value(constant_opinion)) << '\n';
            if (rcd::is_crd_family_algorithm(options.algorithm)) {
                run_full_crd(conflict);
            } else {
                run_full_baseline(conflict);
            }
        } else if (options.conflict_backend == "amgcl") {
            const auto conflict = rcd::AmgclConflict::build(fj_graph, metric);
            const Eigen::VectorXd constant_opinion = Eigen::VectorXd::Ones(
                static_cast<Eigen::Index>(conflict.num_nodes()));
            std::cout << "AMGCL conflict backend\n";
            std::cout << "  metric: " << rcd::to_string(metric) << '\n';
            std::cout << "  solver_type: approximate\n";
            std::cout << "  linear_solver: amg-preconditioned-cg\n";
            std::cout << "  tolerance: " << conflict.tolerance() << '\n';
            std::cout << "  max_iterations: " << conflict.max_iterations() << '\n';
            std::cout << "  nodes: " << conflict.num_nodes() << '\n';
            std::cout << "  conflict_beta_bound: "
                      << rcd::format_decimal(rcd::conflict_beta(conflict)) << '\n';
            std::cout << "  constant_vector_value: "
                      << rcd::format_decimal(conflict.value(constant_opinion)) << '\n';
            std::cout << "  last_iterations: " << conflict.last_iterations() << '\n';
            std::cout << "  last_relative_error: "
                      << rcd::format_decimal(conflict.last_relative_error()) << '\n';
            if (rcd::is_crd_family_algorithm(options.algorithm)) {
                run_full_crd(conflict);
            } else {
                run_full_baseline(conflict);
            }
        } else {
            std::cout << "Conflict backend\n";
            std::cout << "  status: skipped (use --conflict-backend dense or amgcl to run an active backend)\n";
            std::cout << "Algorithm\n";
            std::cout << "  status: skipped (requires an active conflict backend)\n";
        }
        std::cout << "Experiment initialized successfully.\n";
        std::cout << "Output files will use prefix: "
                  << options.output_dir << '/' << options.output_prefix << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n\n";
        rcd::print_usage(std::cerr, argv[0]);
        return 1;
    }
}
