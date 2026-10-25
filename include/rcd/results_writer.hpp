#pragma once

#include "rcd/baselines.hpp"
#include "rcd/crd.hpp"
#include "rcd/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace rcd {

struct ResultOutputPaths {
    std::string txt_path;
    std::string csv_path;
    std::string seeds_path;
};

struct ExperimentMetadata {
    std::string dataset_path;
    std::string output_dir;
    std::string output_prefix;
    std::string innate_opinion_file;
    std::string edge_weight_file;
    std::string rho_file;
    std::string fj_weight_file;
    std::string algorithm = "crd";
    std::string metric;
    std::string propagation_model;
    std::string conflict_backend;
    bool conflict_backend_approximate = false;
    double conflict_beta_bound = 0.0;
    double amgcl_tolerance = 0.0;
    int amgcl_max_iterations = 0;
    int amgcl_last_iterations = 0;
    double amgcl_last_relative_error = 0.0;
    std::uint64_t seed = 0;
    std::size_t graph_nodes = 0;
    std::size_t graph_directed_edges = 0;
    std::size_t graph_skipped_self_loops = 0;
    std::size_t graph_skipped_duplicate_edges = 0;
    std::size_t fj_undirected_edges = 0;
    std::size_t mc_simulations = 0;
    std::string mc_evaluation_mode = "fixed";
    std::size_t mc_min_simulations = 0;
    std::size_t mc_check_interval = 0;
    double mc_ci_relative_half_width = 0.0;
    std::string final_attack_response = "not_available";
    std::size_t final_attack_response_rr_samples = 0;
    std::size_t k_a = 0;
    std::size_t k_b = 0;
    std::string crd_mode;
    std::string crd_variant = "not_applicable";
    std::size_t crd_iterations = 0;
    std::string paper_crd_schedule;
    std::string initial_attack_pool_policy;
    std::size_t baseline_iterations = 0;
    std::size_t fjc_pair_rr_samples = 0;
    std::size_t fjc_linear_rr_samples = 0;
    std::size_t hist_rr_samples_floor = 0;
    std::size_t baseline_max_adaptive_rounds = 0;
    std::size_t attack_rr_samples = 0;
    std::size_t defense_rr_samples = 0;
    std::size_t max_rr_samples = 0;
    std::size_t empirical_samples = 0;
    std::string mixed_distribution;
    std::size_t mixed_rounds = 0;
    double mixed_learning_rate = 0.0;
    double alpha_a = 0.0;
    double alpha_b = 0.0;
    double rho = 0.0;
    double epsilon_cert = 0.0;
    double failure_probability = 0.0;
    std::uint64_t memory_rss_before_algorithm_kib = 0;
    std::uint64_t memory_rss_after_algorithm_kib = 0;
    std::uint64_t memory_peak_rss_kib = 0;
};

ResultOutputPaths make_result_output_paths(
    const std::string& output_dir,
    const std::string& output_prefix);

void write_crd_result_files(
    const CRDResult& result,
    const DirectedGraph& graph,
    const ResultOutputPaths& paths,
    const ExperimentMetadata& metadata = ExperimentMetadata{});

void write_baseline_result_files(
    const BaselineResult& result,
    const DirectedGraph& graph,
    const ResultOutputPaths& paths,
    const ExperimentMetadata& metadata = ExperimentMetadata{});

}  // namespace rcd
