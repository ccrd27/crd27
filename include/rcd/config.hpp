#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <string>

namespace rcd {

struct ProgramOptions {
    std::string dataset_path = "dataset/facebook/facebook.txt";
    std::string output_dir = "results";
    std::string output_prefix = "crd_result";
    std::string innate_opinion_file;
    std::string edge_weight_file;
    std::string rho_file;
    std::string fj_weight_file;
    std::string algorithm = "crd";
    std::string metric = "polarization";
    std::string propagation_model = "ic";
    std::uint64_t seed = 1;
    std::size_t mc_simulations = 10000;
    std::string mc_evaluation_mode = "fixed";
    std::size_t mc_min_simulations = 500;
    std::size_t mc_check_interval = 100;
    double mc_ci_relative_half_width = 0.1;
    std::size_t k_a = 100;
    std::size_t k_b = 100;
    double k_a_relative = 0.01;
    double k_b_relative = 0.01;
    bool k_a_explicit_set = false;
    bool k_b_explicit_set = false;
    bool k_a_relative_set = false;
    bool k_b_relative_set = false;
    std::string crd_mode = "engineering";
    std::size_t crd_iterations = 3;
    std::size_t baseline_iterations = 3;
    std::size_t fjc_pair_rr_samples = 100;
    std::size_t fjc_linear_rr_samples = 100;
    std::size_t hist_rr_samples_floor = 0;
    bool fjc_pair_rr_samples_set = false;
    bool fjc_linear_rr_samples_set = false;
    bool hist_rr_samples_floor_set = false;
    std::size_t baseline_max_adaptive_rounds = 12;
    std::size_t attack_rr_samples = 1000000;
    std::size_t defense_rr_samples = 1000000;
    std::size_t max_rr_samples = 0;
    std::size_t empirical_samples = 1000000;
    std::string mixed_distribution = "uniform";
    bool mixed_distribution_set = false;
    std::size_t mixed_rounds = 5;
    double alpha_a = 0.5;
    double alpha_b = 0.5;
    double rho = 0.5;
    double epsilon_cert = 10.0;
    double epsilon_relative = std::numeric_limits<double>::quiet_NaN();
    bool epsilon_absolute_set = false;
    bool epsilon_relative_set = false;
    double failure_probability = 0.01;
    double mixed_learning_rate = 1.0;
    std::string conflict_backend = "amgcl";
    bool show_help = false;
};

ProgramOptions parse_program_options(int argc, char** argv);
double resolved_epsilon(const ProgramOptions& options, std::size_t num_nodes);
std::size_t resolved_attack_budget(const ProgramOptions& options, std::size_t num_nodes);
std::size_t resolved_defense_budget(const ProgramOptions& options, std::size_t num_nodes);
void print_usage(std::ostream& os, const char* executable);
void print_options(std::ostream& os, const ProgramOptions& options);
void ensure_output_directory(const std::string& output_dir);

}  // namespace rcd
