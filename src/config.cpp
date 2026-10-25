#include "rcd/config.hpp"

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace rcd {
namespace {

std::string require_value(int argc, char** argv, int& index, const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + flag);
    }
    ++index;
    return argv[index];
}

std::size_t parse_size(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument("Invalid integer for " + flag + ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

std::uint64_t parse_u64(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument("Invalid integer for " + flag + ": " + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

double parse_double(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0;
    const auto parsed = std::stod(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument("Invalid number for " + flag + ": " + value);
    }
    return parsed;
}

double parse_percentage_fraction(const std::string& value, const std::string& flag) {
    if (value.empty() || value.back() != '%') {
        throw std::invalid_argument(flag + " must use percentage syntax such as 10%");
    }
    const double percentage = parse_double(value.substr(0, value.size() - 1), flag);
    if (!std::isfinite(percentage) || percentage < 0.0 || percentage > 100.0) {
        throw std::invalid_argument(flag + " must be in [0%, 100%]");
    }
    return percentage / 100.0;
}

void validate_probability(double value, const std::string& flag) {
    if (value < 0.0 || value > 1.0) {
        throw std::invalid_argument(flag + " must be in [0, 1]");
    }
}

}  // namespace

ProgramOptions parse_program_options(int argc, char** argv) {
    ProgramOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            options.show_help = true;
        } else if (flag == "--graph") {
            options.dataset_path = require_value(argc, argv, i, flag);
        } else if (flag == "--output-dir") {
            options.output_dir = require_value(argc, argv, i, flag);
        } else if (flag == "--output-prefix") {
            options.output_prefix = require_value(argc, argv, i, flag);
        } else if (flag == "--s-file") {
            options.innate_opinion_file = require_value(argc, argv, i, flag);
        } else if (flag == "--p-file") {
            options.edge_weight_file = require_value(argc, argv, i, flag);
        } else if (flag == "--rho-u-file") {
            options.rho_file = require_value(argc, argv, i, flag);
        } else if (flag == "--w-file") {
            options.fj_weight_file = require_value(argc, argv, i, flag);
        } else if (flag == "--algorithm") {
            options.algorithm = require_value(argc, argv, i, flag);
        } else if (flag == "--metric") {
            options.metric = require_value(argc, argv, i, flag);
        } else if (flag == "--propagation-model") {
            options.propagation_model = require_value(argc, argv, i, flag);
        } else if (flag == "--conflict-backend") {
            options.conflict_backend = require_value(argc, argv, i, flag);
        } else if (flag == "--seed") {
            options.seed = parse_u64(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--mc-simulations") {
            options.mc_simulations = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--mc-evaluation-mode") {
            options.mc_evaluation_mode = require_value(argc, argv, i, flag);
        } else if (flag == "--mc-min-simulations") {
            options.mc_min_simulations = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--mc-check-interval") {
            options.mc_check_interval = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--mc-ci-relative-half-width") {
            options.mc_ci_relative_half_width = parse_double(
                require_value(argc, argv, i, flag), flag);
        } else if (flag == "--k-A") {
            if (options.k_a_relative_set) {
                throw std::invalid_argument("--k-A and --k-A-relative are mutually exclusive");
            }
            options.k_a = parse_size(require_value(argc, argv, i, flag), flag);
            options.k_a_explicit_set = true;
        } else if (flag == "--k-B") {
            if (options.k_b_relative_set) {
                throw std::invalid_argument("--k-B and --k-B-relative are mutually exclusive");
            }
            options.k_b = parse_size(require_value(argc, argv, i, flag), flag);
            options.k_b_explicit_set = true;
        } else if (flag == "--k-A-relative") {
            if (options.k_a_explicit_set) {
                throw std::invalid_argument("--k-A and --k-A-relative are mutually exclusive");
            }
            options.k_a_relative = parse_percentage_fraction(
                require_value(argc, argv, i, flag), flag);
            options.k_a_relative_set = true;
        } else if (flag == "--k-B-relative") {
            if (options.k_b_explicit_set) {
                throw std::invalid_argument("--k-B and --k-B-relative are mutually exclusive");
            }
            options.k_b_relative = parse_percentage_fraction(
                require_value(argc, argv, i, flag), flag);
            options.k_b_relative_set = true;
        } else if (flag == "--crd-mode") {
            options.crd_mode = require_value(argc, argv, i, flag);
        } else if (flag == "--T") {
            options.crd_iterations = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--baseline-iterations") {
            options.baseline_iterations = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--baseline-pair-rr-samples") {
            options.fjc_pair_rr_samples = parse_size(require_value(argc, argv, i, flag), flag);
            options.fjc_pair_rr_samples_set = true;
        } else if (flag == "--fjc-linear-rr-samples") {
            options.fjc_linear_rr_samples = parse_size(require_value(argc, argv, i, flag), flag);
            options.fjc_linear_rr_samples_set = true;
        } else if (flag == "--hist-rr-samples") {
            options.hist_rr_samples_floor = parse_size(require_value(argc, argv, i, flag), flag);
            options.hist_rr_samples_floor_set = true;
        } else if (flag == "--baseline-max-adaptive-rounds") {
            options.baseline_max_adaptive_rounds = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--theta-A") {
            options.attack_rr_samples = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--theta-B") {
            options.defense_rr_samples = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--max-rr-samples") {
            options.max_rr_samples = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--theta-emp") {
            options.empirical_samples = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--mixed-distribution") {
            options.mixed_distribution = require_value(argc, argv, i, flag);
            options.mixed_distribution_set = true;
        } else if (flag == "--mixed-rounds") {
            options.mixed_rounds = parse_size(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--alpha-A") {
            options.alpha_a = parse_double(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--alpha-B") {
            options.alpha_b = parse_double(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--rho-u") {
            options.rho = parse_double(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--epsilon") {
            if (options.epsilon_relative_set) {
                throw std::invalid_argument("--epsilon and --epsilon-relative are mutually exclusive");
            }
            options.epsilon_cert = parse_double(require_value(argc, argv, i, flag), flag);
            options.epsilon_absolute_set = true;
        } else if (flag == "--epsilon-relative") {
            if (options.epsilon_absolute_set) {
                throw std::invalid_argument("--epsilon and --epsilon-relative are mutually exclusive");
            }
            options.epsilon_relative = parse_percentage_fraction(
                require_value(argc, argv, i, flag), flag);
            options.epsilon_relative_set = true;
        } else if (flag == "--delta") {
            options.failure_probability = parse_double(require_value(argc, argv, i, flag), flag);
        } else if (flag == "--mixed-learning-rate") {
            options.mixed_learning_rate = parse_double(require_value(argc, argv, i, flag), flag);
        } else {
            throw std::invalid_argument("Unknown option: " + flag);
        }
    }

    if (options.show_help) {
        return options;
    }

    if (options.algorithm != "crd" &&
        options.algorithm != "crd-single" &&
        options.algorithm != "crd-sur" &&
        options.algorithm != "crd-adamix" &&
        options.algorithm != "crd-linear" &&
        options.algorithm != "fjc-iter" &&
        options.algorithm != "fjc-linear-iter" &&
        options.algorithm != "hist-iter" &&
        options.algorithm != "hist" &&
        options.algorithm != "degree" &&
        options.algorithm != "random" &&
        options.algorithm != "exact") {
        throw std::invalid_argument(
            "--algorithm must be one of crd, crd-single, crd-sur, crd-adamix, crd-linear, fjc-iter, fjc-linear-iter, hist-iter, hist, degree, random, or exact");
    }
    if (options.metric != "polarization" && options.metric != "disagreement") {
        throw std::invalid_argument("--metric must be either polarization or disagreement");
    }
    if (options.innate_opinion_file.empty()) {
        throw std::invalid_argument("--s-file is required");
    }
    if (options.propagation_model != "ic" && options.propagation_model != "lt") {
        throw std::invalid_argument("--propagation-model must be either ic or lt");
    }
    if (options.conflict_backend != "none" &&
        options.conflict_backend != "dense" &&
        options.conflict_backend != "amgcl") {
        throw std::invalid_argument("--conflict-backend must be one of none, dense, or amgcl");
    }
    if (options.algorithm == "fjc-iter" && options.conflict_backend != "dense") {
        throw std::invalid_argument(
            "FJC-Iter requires --conflict-backend dense because it uses pair-RR selection");
    }
    if (options.mixed_distribution != "uniform" && options.mixed_distribution != "adaptive") {
        throw std::invalid_argument("--mixed-distribution must be either uniform or adaptive");
    }
    if (options.algorithm == "crd-sur" && options.mixed_distribution != "uniform") {
        throw std::invalid_argument("CRD-Sur requires --mixed-distribution uniform");
    }
    if (options.algorithm == "crd-adamix" && options.mixed_distribution_set &&
        options.mixed_distribution != "adaptive") {
        throw std::invalid_argument("CRD-AdaMix requires --mixed-distribution adaptive when specified");
    }
    if (options.crd_mode != "engineering" && options.crd_mode != "paper-exact") {
        throw std::invalid_argument("--crd-mode must be either engineering or paper-exact");
    }
    const bool crd_family =
        options.algorithm == "crd" || options.algorithm == "crd-single" ||
        options.algorithm == "crd-sur" || options.algorithm == "crd-adamix" ||
        options.algorithm == "crd-linear";
    const bool crd_ablation = crd_family && options.algorithm != "crd";
    const bool crd_paper_exact =
        options.algorithm == "crd" && options.crd_mode == "paper-exact";
    if (crd_ablation && options.crd_mode != "engineering") {
        throw std::invalid_argument("CRD ablation variants currently require --crd-mode engineering");
    }
    if (options.mc_simulations == 0) {
        throw std::invalid_argument("--mc-simulations must be positive");
    }
    if (options.mc_evaluation_mode != "fixed" && options.mc_evaluation_mode != "adaptive") {
        throw std::invalid_argument("--mc-evaluation-mode must be either fixed or adaptive");
    }
    if (options.mc_evaluation_mode == "adaptive") {
        if (options.mc_min_simulations < 2 ||
            options.mc_min_simulations > options.mc_simulations) {
            throw std::invalid_argument(
                "--mc-min-simulations must be in [2, --mc-simulations] for adaptive evaluation");
        }
        if (options.mc_check_interval == 0) {
            throw std::invalid_argument("--mc-check-interval must be positive");
        }
        if (!std::isfinite(options.mc_ci_relative_half_width) ||
            options.mc_ci_relative_half_width <= 0.0) {
            throw std::invalid_argument(
                "--mc-ci-relative-half-width must be finite and positive");
        }
    }
    if (options.crd_iterations == 0) {
        throw std::invalid_argument("--T must be positive");
    }
    if (options.baseline_iterations == 0) {
        throw std::invalid_argument("--baseline-iterations must be positive");
    }
    if (options.fjc_pair_rr_samples == 0) {
        throw std::invalid_argument("--baseline-pair-rr-samples must be positive");
    }
    if (options.fjc_linear_rr_samples == 0) {
        throw std::invalid_argument("--fjc-linear-rr-samples must be positive");
    }
    if (options.baseline_max_adaptive_rounds == 0) {
        throw std::invalid_argument("--baseline-max-adaptive-rounds must be positive");
    }
    if (options.fjc_pair_rr_samples_set && options.algorithm != "fjc-iter") {
        throw std::invalid_argument("--baseline-pair-rr-samples is only valid for FJC-Iter");
    }
    if (options.fjc_linear_rr_samples_set && options.algorithm != "fjc-linear-iter") {
        throw std::invalid_argument("--fjc-linear-rr-samples is only valid for FJC-Linear-Iter");
    }
    if (options.hist_rr_samples_floor_set &&
        options.algorithm != "hist-iter" && options.algorithm != "hist") {
        throw std::invalid_argument("--hist-rr-samples is only valid for HIST or HIST-Iter");
    }
    if (options.algorithm == "hist-iter" || options.algorithm == "hist") {
        if (options.epsilon_relative_set) {
            throw std::invalid_argument(
                "HIST requires absolute --epsilon in (0, 1); --epsilon-relative is not supported");
        }
        if (options.epsilon_cert <= 0.0 || options.epsilon_cert >= 1.0) {
            throw std::invalid_argument("HIST requires --epsilon in (0, 1)");
        }
    }
    if (options.attack_rr_samples == 0) {
        throw std::invalid_argument("--theta-A must be positive");
    }
    if (options.defense_rr_samples == 0) {
        throw std::invalid_argument("--theta-B must be positive");
    }
    if (options.max_rr_samples != 0 &&
        ((crd_family && !crd_paper_exact &&
          (options.max_rr_samples < options.attack_rr_samples ||
           options.max_rr_samples < options.defense_rr_samples)) ||
         (options.algorithm == "fjc-iter" &&
          options.max_rr_samples < options.fjc_pair_rr_samples) ||
         (options.algorithm == "fjc-linear-iter" &&
          options.max_rr_samples < options.fjc_linear_rr_samples))) {
        throw std::invalid_argument(
            "--max-rr-samples must be 0 or at least the active RR sample base counts");
    }
    if (options.empirical_samples == 0 && options.algorithm != "crd-sur" &&
        options.algorithm != "crd-linear") {
        throw std::invalid_argument("--theta-emp must be positive");
    }
    if (options.mixed_rounds == 0) {
        throw std::invalid_argument("--mixed-rounds must be positive");
    }
    validate_probability(options.alpha_a, "--alpha-A");
    validate_probability(options.alpha_b, "--alpha-B");
    validate_probability(options.rho, "--rho-u");
    if (!std::isfinite(options.epsilon_cert) || options.epsilon_cert < 0.0) {
        throw std::invalid_argument("--epsilon must be finite and nonnegative");
    }
    if (options.epsilon_relative_set &&
        (!std::isfinite(options.epsilon_relative) || options.epsilon_relative < 0.0)) {
        throw std::invalid_argument("--epsilon-relative must be finite and nonnegative");
    }
    if (crd_paper_exact && options.epsilon_cert <= 0.0) {
        throw std::invalid_argument("--epsilon must be positive in paper-exact CRD mode");
    }
    if (!std::isfinite(options.failure_probability) ||
        options.failure_probability <= 0.0 ||
        options.failure_probability >= 1.0) {
        throw std::invalid_argument("--delta must be in (0, 1)");
    }
    if (!std::isfinite(options.mixed_learning_rate) || options.mixed_learning_rate < 0.0) {
        throw std::invalid_argument("--mixed-learning-rate must be finite and nonnegative");
    }
    return options;
}

double resolved_epsilon(const ProgramOptions& options, std::size_t num_nodes) {
    if (num_nodes == 0) {
        throw std::invalid_argument("Cannot resolve epsilon for an empty graph");
    }
    if (options.epsilon_relative_set) {
        return static_cast<double>(num_nodes) * options.epsilon_relative;
    }
    return options.epsilon_cert;
}

namespace {

std::size_t resolve_relative_budget(
    double relative_fraction,
    bool relative_set,
    std::size_t explicit_budget,
    std::size_t num_nodes,
    const char* flag) {
    if (num_nodes == 0) {
        throw std::invalid_argument("Cannot resolve " + std::string(flag) + " for an empty graph");
    }
    if (!relative_set) {
        return explicit_budget;
    }
    const auto scaled_budget = static_cast<std::size_t>(
        std::ceil(relative_fraction * static_cast<double>(num_nodes)));
    return std::max<std::size_t>(1, scaled_budget);
}

}  // namespace

std::size_t resolved_attack_budget(const ProgramOptions& options, std::size_t num_nodes) {
    return resolve_relative_budget(
        options.k_a_relative,
        options.k_a_relative_set,
        options.k_a,
        num_nodes,
        "--k-A-relative");
}

std::size_t resolved_defense_budget(const ProgramOptions& options, std::size_t num_nodes) {
    return resolve_relative_budget(
        options.k_b_relative,
        options.k_b_relative_set,
        options.k_b,
        num_nodes,
        "--k-B-relative");
}

void print_usage(std::ostream& os, const char* executable) {
    os << "Usage: " << executable << " [options]\n"
       << "\n"
       << "Options:\n"
       << "  --graph PATH             Paper notation G: directed propagation graph\n"
       << "  --output-dir DIR         Output directory (default: results)\n"
       << "  --output-prefix PREFIX   Output file prefix (default: crd_result)\n"
       << "  --s-file PATH            Paper notation s: innate opinions, node_id value\n"
       << "  --p-file PATH            Paper notation p_uv: propagation probabilities\n"
       << "  --rho-u-file PATH        Paper notation rho_u: per-node suppression values\n"
       << "  --w-file PATH            Paper notation w_uv: FJ edge weights\n"
       << "  --algorithm NAME         crd, crd-single, crd-sur, crd-adamix, crd-linear, fjc-iter, fjc-linear-iter, hist-iter, hist, degree, random, or exact (default: crd)\n"
       << "  --metric NAME            polarization or disagreement (default: polarization)\n"
       << "  --propagation-model M    ic or lt propagation model (default: ic)\n"
       << "  --conflict-backend M     none, dense, or amgcl FJ/conflict backend (default: amgcl)\n"
       << "  --seed N                 Base random seed (default: 1)\n"
       << "  --mc-simulations N       Final MC count in fixed mode, maximum count in adaptive mode (default: 10000)\n"
       << "  --mc-evaluation-mode M   fixed or adaptive final MC evaluation mode (default: fixed)\n"
       << "  --mc-min-simulations N   Adaptive final MC minimum count (default: 500)\n"
       << "  --mc-check-interval N    Adaptive final MC CI check interval (default: 100)\n"
       << "  --mc-ci-relative-half-width X Adaptive relative 95% CI half-width target (default: 0.1)\n"
       << "  --k-A N                  Paper attack budget k_A (default: 100)\n"
       << "  --k-B N                  Paper defense budget k_B (default: 100)\n"
       << "  --k-A-relative X%        Attack budget as a percentage of n, rounded up (relative default: 1%; mutually exclusive with --k-A)\n"
       << "  --k-B-relative X%        Defense budget as a percentage of n, rounded up (relative default: 1%; mutually exclusive with --k-B)\n"
       << "  --crd-mode M             engineering or paper-exact CRD loop (default: engineering)\n"
       << "  --T N                    Paper CRD iteration limit T (default: 3)\n"
       << "  --baseline-iterations N  Baseline outer iterations (default: 3)\n"
       << "  --baseline-pair-rr-samples N FJC-Iter pair-RR sample count (default: 100)\n"
       << "  --fjc-linear-rr-samples N FJC-Linear-Iter ordinary RR sample count (default: 100)\n"
       << "  --hist-rr-samples N     Optional HIST/HIST-Iter R1 sample floor; 0 uses the formula (default: 0)\n"
       << "  --baseline-max-adaptive-rounds N HIST SUBSIM safety limit on R1 doubling rounds (default: 12)\n"
       << "  --theta-A N              Engineering attack RR count theta_A (default: 1000000)\n"
       << "  --theta-B N              Engineering defense RR count theta_B (default: 1000000)\n"
       << "  --max-rr-samples N       Maximum formula-driven RR samples, 0 means unlimited (default: 0)\n"
       << "  --theta-emp N            Samples per empirical pool and engineering final holdout (default: 1000000)\n"
       << "  --mixed-distribution M   uniform or adaptive mixed attack distribution (default: uniform)\n"
       << "  --mixed-rounds N         Mixed distribution rounds for adaptive mode (default: 5)\n"
       << "  --alpha-A X              Paper attacker strength alpha_A (default: 0.5)\n"
       << "  --alpha-B X              Paper defender strength alpha_B (default: 0.5)\n"
       << "  --rho-u X                Default rho_u when no file is supplied (default: 0.5)\n"
       << "  --epsilon X              Paper tolerance epsilon (default: 10)\n"
       << "  --epsilon-relative X%   Relative epsilon as a percentage of n (mutually exclusive with --epsilon)\n"
       << "  --delta X                Paper failure probability delta (default: 0.01)\n"
       << "  --mixed-learning-rate X  Adaptive mixed distribution learning rate (default: 1.0)\n"
       << "  -h, --help               Show this help message\n";
}

void print_options(std::ostream& os, const ProgramOptions& options) {
    os << "Configuration\n"
       << "  graph_G: " << options.dataset_path << '\n'
       << "  output_dir: " << options.output_dir << '\n'
       << "  output_prefix: " << options.output_prefix << '\n'
       << "  s_file: " << options.innate_opinion_file << '\n'
       << "  p_file: " << (options.edge_weight_file.empty() ? "default" : options.edge_weight_file) << '\n'
       << "  rho_u_file: " << (options.rho_file.empty() ? "default" : options.rho_file) << '\n'
       << "  w_file: " << (options.fj_weight_file.empty() ? "default" : options.fj_weight_file) << '\n'
       << "  algorithm: " << options.algorithm << '\n'
       << "  metric: " << options.metric << '\n'
       << "  propagation_model: " << options.propagation_model << '\n'
       << "  conflict_backend: " << options.conflict_backend << '\n'
       << "  seed: " << options.seed << '\n'
       << "  mc_simulations: " << options.mc_simulations << '\n'
       << "  mc_evaluation_mode: " << options.mc_evaluation_mode << '\n'
       << "  mc_min_simulations: " << options.mc_min_simulations << '\n'
       << "  mc_check_interval: " << options.mc_check_interval << '\n'
       << "  mc_ci_relative_half_width: " << options.mc_ci_relative_half_width << '\n'
       << "  k_A: " << options.k_a << '\n'
       << "  k_B: " << options.k_b << '\n'
       << "  k_A_relative: "
       << (options.k_a_relative_set
           ? std::to_string(options.k_a_relative * 100.0) + "%"
           : "disabled") << '\n'
       << "  k_B_relative: "
       << (options.k_b_relative_set
           ? std::to_string(options.k_b_relative * 100.0) + "%"
           : "disabled") << '\n'
       << "  crd_mode: " << options.crd_mode << '\n'
       << "  T: " << options.crd_iterations << '\n'
       << "  baseline_iterations: " << options.baseline_iterations << '\n'
       << "  fjc_pair_rr_samples: " << options.fjc_pair_rr_samples << '\n'
       << "  fjc_linear_rr_samples: " << options.fjc_linear_rr_samples << '\n'
       << "  hist_rr_samples_floor: " << options.hist_rr_samples_floor << '\n'
       << "  baseline_max_adaptive_rounds: " << options.baseline_max_adaptive_rounds << '\n'
       << "  theta_A: " << options.attack_rr_samples << '\n'
       << "  theta_B: " << options.defense_rr_samples << '\n'
       << "  max_rr_samples: " << options.max_rr_samples << '\n'
       << "  theta_emp: "
       << (options.algorithm == "crd-sur" ? 0 : options.empirical_samples) << '\n'
       << "  mixed_distribution: "
       << (options.algorithm == "crd-adamix" ? "adaptive" : options.mixed_distribution) << '\n'
       << "  mixed_rounds: " << options.mixed_rounds << '\n'
       << "  alpha_A: " << options.alpha_a << '\n'
       << "  alpha_B: " << options.alpha_b << '\n'
       << "  default_rho_u: " << options.rho << '\n'
       << "  epsilon: " << options.epsilon_cert << '\n'
       << "  epsilon_relative: "
       << (options.epsilon_relative_set ? "enabled" : "disabled") << '\n'
       << "  delta: " << options.failure_probability << '\n'
       << "  mixed_learning_rate: " << options.mixed_learning_rate << '\n'
       << "  conflict_backend_note: explicit experiment-entry selection; no graph-size auto switch\n";
}

void ensure_output_directory(const std::string& output_dir) {
    const std::filesystem::path path(output_dir);
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        if (!std::filesystem::is_directory(path, ec)) {
            throw std::runtime_error("Output path exists but is not a directory: " + output_dir);
        }
        return;
    }
    if (!std::filesystem::create_directories(path, ec)) {
        throw std::runtime_error("Failed to create output directory: " + output_dir);
    }
}

}  // namespace rcd
