#include "rcd/crd.hpp"

#include "rcd/certificate.hpp"
#include "rcd/progress.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace rcd {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

double certificate_gap(double lower, double upper) {
    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return upper - lower;
}

bool finite_gap_within_tolerance(double gap, double epsilon_cert) {
    return std::isfinite(gap) && gap >= 0.0 && gap <= epsilon_cert;
}

bool strict_paper_certificate_mode(
    const ConflictBackend& conflict,
    const CRDOptions& options) {
    return options.crd_mode == CRDMode::PaperExact &&
           options.variant == CRDVariant::Full &&
           dynamic_cast<const DenseConflict*>(&conflict) != nullptr;
}

AttackOracleOptions make_attack_options(const CRDOptions& options, std::size_t iteration) {
    AttackOracleOptions attack_options;
    attack_options.budget = options.attack_budget;
    attack_options.rr_samples = options.attack_rr_samples;
    attack_options.seed = options.seed + 0x9e3779b97f4a7c15ULL * (iteration + 1);
    attack_options.epsilon_rr = options.epsilon_rr;
    attack_options.failure_probability =
        options.failure_probability / (2.0 * static_cast<double>(options.max_iterations));
    attack_options.max_rr_samples = options.max_rr_samples;
    attack_options.propagation_model = options.propagation_model;
    return attack_options;
}

DefenseOracleOptions make_defense_options(const CRDOptions& options, std::size_t iteration) {
    DefenseOracleOptions defense_options;
    defense_options.budget = options.defense_budget;
    defense_options.rr_samples = options.defense_rr_samples;
    defense_options.empirical_samples = options.empirical_samples;
    defense_options.seed = options.seed + 0xbf58476d1ce4e5b9ULL * (iteration + 1);
    defense_options.epsilon_rr = options.epsilon_rr;
    defense_options.mixed_distribution_mode = options.mixed_distribution_mode;
    defense_options.mixed_rounds = options.mixed_rounds;
    defense_options.mixed_learning_rate = options.mixed_learning_rate;
    defense_options.failure_probability =
        options.failure_probability / (2.0 * static_cast<double>(options.max_iterations));
    defense_options.max_rr_samples = options.max_rr_samples;
    defense_options.propagation_model = options.propagation_model;
    defense_options.candidate_policy = options.defense_candidate_policy;
    return defense_options;
}

PaperExactSamplePlanOptions make_paper_exact_sample_plan_options(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const CRDOptions& options) {
    PaperExactSamplePlanOptions plan_options;
    plan_options.num_nodes = graph.num_nodes();
    plan_options.attack_budget = options.attack_budget;
    plan_options.defense_budget = options.defense_budget;
    plan_options.beta = conflict_beta(conflict);
    plan_options.epsilon = options.epsilon_cert;
    plan_options.failure_probability = options.failure_probability;
    plan_options.iterations = options.max_iterations;
    plan_options.max_rr_samples = options.max_rr_samples;
    return plan_options;
}

AttackOracleOptions make_paper_exact_attack_options(
    const CRDOptions& options,
    const PaperExactRoundSamplePlan& plan,
    std::size_t iteration) {
    AttackOracleOptions attack_options;
    attack_options.budget = options.attack_budget;
    attack_options.rr_samples = plan.attack_rr_samples;
    attack_options.seed = options.seed + 0x9e3779b97f4a7c15ULL * (iteration + 1);
    attack_options.epsilon_rr = options.epsilon_rr;
    attack_options.failure_probability = plan.attack_failure_probability;
    attack_options.max_rr_samples = plan.attack_rr_sample_cap;
    attack_options.propagation_model = options.propagation_model;
    attack_options.oracle_epsilon = plan.attack_oracle_epsilon;
    return attack_options;
}

DefenseOracleOptions make_paper_exact_defense_options(
    const CRDOptions& options,
    const PaperExactRoundSamplePlan& plan,
    std::size_t iteration) {
    DefenseOracleOptions defense_options;
    defense_options.budget = options.defense_budget;
    defense_options.rr_samples = plan.defense_rr_samples;
    defense_options.empirical_samples = plan.defense_rr_samples;
    defense_options.seed = options.seed + 0xbf58476d1ce4e5b9ULL * (iteration + 1);
    defense_options.epsilon_rr = options.epsilon_rr;
    defense_options.mixed_distribution_mode = MixedDistributionMode::Uniform;
    defense_options.mixed_rounds = 1;
    defense_options.mixed_learning_rate = options.mixed_learning_rate;
    defense_options.failure_probability = plan.defense_failure_probability;
    defense_options.max_rr_samples = plan.defense_rr_sample_cap;
    defense_options.propagation_model = options.propagation_model;
    defense_options.oracle_epsilon = plan.defense_oracle_epsilon;
    defense_options.defense_oracle_mode = DefenseOracleMode::PaperExact;
    return defense_options;
}

double allocated_attack_failure_probability(const CRDOptions& options) {
    return options.failure_probability / (2.0 * static_cast<double>(options.max_iterations));
}

double allocated_defense_failure_probability(const CRDOptions& options) {
    return options.failure_probability / (2.0 * static_cast<double>(options.max_iterations));
}

std::vector<NodeId> degree_out_attack_seed_set(
    const DirectedGraph& graph,
    std::size_t budget) {
    std::vector<NodeId> nodes;
    nodes.reserve(graph.num_nodes());
    for (NodeId node = 0; static_cast<std::size_t>(node) < graph.num_nodes(); ++node) {
        nodes.push_back(node);
    }
    std::sort(nodes.begin(), nodes.end(), [&](NodeId lhs, NodeId rhs) {
        const auto lhs_degree = graph.out_adj[lhs].size();
        const auto rhs_degree = graph.out_adj[rhs].size();
        if (lhs_degree != rhs_degree) {
            return lhs_degree > rhs_degree;
        }
        return lhs < rhs;
    });
    if (nodes.size() > budget) {
        nodes.resize(budget);
    }
    std::sort(nodes.begin(), nodes.end());
    return nodes;
}

bool attack_pool_contains(
    const std::vector<AttackPoolEntry>& attack_pool,
    const std::vector<NodeId>& attack_seeds) {
    std::vector<NodeId> normalized_attack = attack_seeds;
    std::sort(normalized_attack.begin(), normalized_attack.end());
    normalized_attack.erase(
        std::unique(normalized_attack.begin(), normalized_attack.end()),
        normalized_attack.end());
    return std::any_of(
        attack_pool.begin(),
        attack_pool.end(),
        [&normalized_attack](const AttackPoolEntry& entry) {
            std::vector<NodeId> existing = entry.attack_seeds;
            std::sort(existing.begin(), existing.end());
            existing.erase(std::unique(existing.begin(), existing.end()), existing.end());
            return existing == normalized_attack;
        });
}

void select_result_iteration(CRDResult& result, std::size_t iteration_index) {
    if (iteration_index >= result.iterations.size()) {
        throw std::out_of_range("Selected CRD iteration index is out of range");
    }
    const auto& log = result.iterations[iteration_index];
    result.defense_seeds = log.defense_seeds;
    result.attack_seeds = log.attack_seeds;
    result.best_lower_certificate = log.lower_certificate;
    result.best_upper_certificate = log.upper_certificate;
    result.best_certificate_gap = log.certificate_gap;
    result.final_selected_iteration = log.iteration;
}

void validate_crd_inputs(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CRDOptions& options) {
    if (options.crd_mode == CRDMode::Engineering && options.max_iterations == 0) {
        throw std::invalid_argument("CRD requires at least one iteration");
    }
    if (options.crd_mode == CRDMode::PaperExact && options.max_iterations == 0) {
        throw std::invalid_argument("Paper-exact CRD requires at least one iteration");
    }
    if (options.variant != CRDVariant::Full &&
        options.crd_mode != CRDMode::Engineering) {
        throw std::invalid_argument(
            "CRD ablation variants currently support engineering mode only");
    }
    if (options.variant == CRDVariant::LinearApprox) {
        throw std::invalid_argument("CRD-Linear must be executed through run_crd_linear");
    }
    if (options.crd_mode == CRDMode::Engineering && options.attack_rr_samples == 0) {
        throw std::invalid_argument("CRD requires at least one attack RR sample");
    }
    if (!std::isfinite(options.failure_probability) ||
        options.failure_probability <= 0.0 ||
        options.failure_probability >= 1.0) {
        throw std::invalid_argument("CRD failure probability must be in (0, 1)");
    }
    if (options.crd_mode == CRDMode::Engineering && options.defense_rr_samples == 0) {
        throw std::invalid_argument("CRD requires at least one defense RR sample");
    }
    if (options.crd_mode == CRDMode::Engineering &&
        options.max_rr_samples != 0 &&
        (options.max_rr_samples < options.attack_rr_samples ||
         options.max_rr_samples < options.defense_rr_samples)) {
        throw std::invalid_argument(
            "CRD max RR samples must be 0 or at least the attack and defense base RR samples");
    }
    if (options.defense_candidate_policy == DefenseCandidatePolicy::EmpiricalAndSurrogate &&
        options.empirical_samples == 0) {
        throw std::invalid_argument("CRD requires at least one empirical sample");
    }
    if (options.mixed_rounds == 0) {
        throw std::invalid_argument("CRD requires at least one mixed distribution round");
    }
    if (!std::isfinite(options.mixed_learning_rate) || options.mixed_learning_rate < 0.0) {
        throw std::invalid_argument("CRD mixed learning rate must be finite and nonnegative");
    }
    if (!std::isfinite(options.epsilon_rr) || options.epsilon_rr <= 0.0) {
        throw std::invalid_argument("CRD epsilon_rr must be finite and positive");
    }
    if (!std::isfinite(options.epsilon_cert) || options.epsilon_cert < 0.0) {
        throw std::invalid_argument("CRD certificate tolerance must be finite and nonnegative");
    }
    if (options.crd_mode == CRDMode::PaperExact && options.epsilon_cert <= 0.0) {
        throw std::invalid_argument("Paper-exact CRD certificate tolerance must be positive");
    }
    if (conflict.num_nodes() != graph.num_nodes() ||
        static_cast<std::size_t>(innate.size()) != graph.num_nodes()) {
        throw std::invalid_argument("Graph, conflict backend, and opinion vector sizes must match");
    }
}

}  // namespace

std::string to_string(CRDMode mode) {
    switch (mode) {
        case CRDMode::Engineering:
            return "engineering";
        case CRDMode::PaperExact:
            return "paper-exact";
    }
    return "unknown";
}

CRDMode parse_crd_mode(const std::string& value) {
    if (value == "engineering") {
        return CRDMode::Engineering;
    }
    if (value == "paper-exact") {
        return CRDMode::PaperExact;
    }
    throw std::invalid_argument("CRD mode must be either engineering or paper-exact");
}

std::string to_string(CRDVariant variant) {
    switch (variant) {
    case CRDVariant::Full:
        return "full";
    case CRDVariant::SingleAttackPool:
        return "single_attack_pool";
    case CRDVariant::SurrogateOnly:
        return "surrogate_only";
    case CRDVariant::AdaptiveMix:
        return "adaptive_mix";
    case CRDVariant::LinearApprox:
        return "linear_approx";
    }
    return "unknown";
}

bool is_crd_family_algorithm(const std::string& value) {
    return value == "crd" || value == "crd-single" || value == "crd-sur" ||
        value == "crd-adamix" || value == "crd-linear";
}

CRDVariant parse_crd_variant(const std::string& value) {
    if (value == "crd") {
        return CRDVariant::Full;
    }
    if (value == "crd-single") {
        return CRDVariant::SingleAttackPool;
    }
    if (value == "crd-sur") {
        return CRDVariant::SurrogateOnly;
    }
    if (value == "crd-adamix") {
        return CRDVariant::AdaptiveMix;
    }
    if (value == "crd-linear") {
        return CRDVariant::LinearApprox;
    }
    throw std::invalid_argument(
        "CRD algorithm must be one of crd, crd-single, crd-sur, crd-adamix, or crd-linear");
}

std::string format_seed_set(const std::vector<NodeId>& seeds) {
    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << seeds[i];
    }
    os << ']';
    return os.str();
}

std::string format_seed_set_preview(
    const std::vector<NodeId>& seeds,
    std::size_t max_displayed_seeds) {
    if (seeds.size() <= max_displayed_seeds) {
        return format_seed_set(seeds);
    }

    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < max_displayed_seeds; ++i) {
        if (i > 0) {
            os << ',';
        }
        os << seeds[i];
    }
    os << "] (total_seeds=" << seeds.size()
       << "; showing_first=" << max_displayed_seeds << ')';
    return os.str();
}

std::string format_decimal(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    if (value == 0.0) {
        return "0";
    }

    const double abs_value = std::fabs(value);
    const int integer_digits = static_cast<int>(std::floor(std::log10(abs_value))) + 1;
    const int significant_digits = std::numeric_limits<double>::max_digits10;
    const int precision = std::max(0, significant_digits - integer_digits);

    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    std::string text = os.str();

    const auto dot = text.find('.');
    if (dot != std::string::npos) {
        while (!text.empty() && text.back() == '0') {
            text.pop_back();
        }
        if (!text.empty() && text.back() == '.') {
            text.pop_back();
        }
    }
    return text == "-0" ? "0" : text;
}

void print_crd_iteration_log(std::ostream& os, const CRDIterationLog& log) {
    os << "CRD iteration " << log.iteration << '\n'
       << "  crd_mode: " << log.crd_mode << '\n'
       << "  crd_variant: " << log.crd_variant << '\n'
       << "  epsilon_A: " << format_decimal(log.epsilon_a) << '\n'
       << "  epsilon_B: " << format_decimal(log.epsilon_b) << '\n'
       << "  delta_A: " << format_decimal(log.delta_a) << '\n'
       << "  delta_B: " << format_decimal(log.delta_b) << '\n'
       << "  theta_A: " << log.theta_a << '\n'
       << "  theta_B: " << log.theta_b << '\n'
       << "  theta_emp: " << log.theta_emp << '\n'
       << "  theta_emp_selection: " << log.theta_emp_selection << '\n'
       << "  theta_emp_validation: " << log.theta_emp_validation << '\n'
       << "  empirical_validation_independent: "
       << (log.empirical_validation_independent ? "true" : "false") << '\n'
       << "  empirical_validation_streaming: "
       << (log.empirical_validation_streaming ? "true" : "false") << '\n'
       << "  attack_pool_size: " << log.attack_pool_size << '\n'
       << "  attack_pool_size_before_defense: " << log.attack_pool_size_before_defense << '\n'
       << "  attack_pool_size_after_attack: " << log.attack_pool_size_after_attack << '\n'
       << "  attack_pool_update: " << log.attack_pool_update << '\n'
       << "  propagation_model: " << log.propagation_model << '\n'
       << "  defense_status: " << to_string(log.defense_status) << '\n'
       << "  defense_candidate_source: " << to_string(log.defense_candidate_source) << '\n'
       << "  defense_candidate_count: " << log.defense_candidate_count << '\n'
       << "  selected_candidate_mixed_round: "
       << log.selected_candidate_mixed_round << '\n'
       << "  adaptive_candidate_count_before_deduplication: "
       << log.adaptive_candidate_count_before_deduplication << '\n'
       << "  adaptive_candidate_count_after_deduplication: "
       << log.adaptive_candidate_count_after_deduplication << '\n'
       << "  empirical_value_empirical: " << format_decimal(log.empirical_value_empirical) << '\n'
       << "  empirical_value_surrogate: " << format_decimal(log.empirical_value_surrogate) << '\n'
       << "  empirical_value_selected: " << format_decimal(log.empirical_value_selected) << '\n'
       << "  mixed_distribution_type: " << log.mixed_distribution_type << '\n'
       << "  mixed_rounds: " << log.mixed_rounds << '\n'
       << "  selected_mixed_round: " << log.selected_mixed_round << '\n'
       << "  attack_failure_probability: " << format_decimal(log.attack_failure_probability) << '\n'
       << "  defense_failure_probability: " << format_decimal(log.defense_failure_probability) << '\n'
       << "  attack_certificate_status: " << log.attack_certificate_status << '\n'
       << "  attack_certificate_formula: " << log.attack_certificate_formula << '\n'
       << "  attack_required_rr_samples: " << log.attack_required_rr_samples << '\n'
       << "  attack_rr_sample_cap: " << log.attack_rr_sample_cap << '\n'
       << "  attack_rr_sample_cap_triggered: "
       << (log.attack_rr_sample_cap_triggered ? "true" : "false") << '\n'
       << "  attack_generated_rr_samples: " << log.attack_generated_rr_samples << '\n'
       << "  defense_certificate_status: " << log.defense_certificate_status << '\n'
       << "  lower_certificate_formula: " << log.lower_certificate_formula << '\n'
       << "  defense_required_rr_samples: " << log.defense_required_rr_samples << '\n'
       << "  defense_rr_sample_cap: " << log.defense_rr_sample_cap << '\n'
       << "  defense_rr_sample_cap_triggered: "
       << (log.defense_rr_sample_cap_triggered ? "true" : "false") << '\n'
       << "  defense_generated_rr_samples: " << log.defense_generated_rr_samples << '\n'
       << "  unclamped_lower_certificate: "
       << format_decimal(log.unclamped_lower_certificate) << '\n'
       << "  raw_lower_certificate: " << format_decimal(log.raw_lower_certificate) << '\n'
       << "  defense_conservative_constant: "
       << format_decimal(log.defense_conservative_constant) << '\n'
       << "  defense_conservative_surrogate: "
       << format_decimal(log.defense_conservative_surrogate) << '\n'
       << "  raw_upper_certificate: " << format_decimal(log.raw_upper_certificate) << '\n'
       << "  attack_conservative_constant: "
       << format_decimal(log.attack_conservative_constant) << '\n'
       << "  attack_conservative_surrogate: "
       << format_decimal(log.attack_conservative_surrogate) << '\n'
       << "  attack_constant_failure_probability: "
       << format_decimal(log.attack_constant_failure_probability) << '\n'
       << "  attack_surrogate_failure_probability: "
       << format_decimal(log.attack_surrogate_failure_probability) << '\n'
       << "  defense_distribution_failure_probability: "
       << format_decimal(log.defense_distribution_failure_probability) << '\n'
       << "  defense_constant_failure_probability: "
       << format_decimal(log.defense_constant_failure_probability) << '\n'
       << "  defense_surrogate_failure_probability: "
       << format_decimal(log.defense_surrogate_failure_probability) << '\n'
       << "  empirical_evaluator: " << log.empirical_evaluator << '\n'
       << "  sample_cap_triggered: " << (log.sample_cap_triggered ? "true" : "false") << '\n'
       << "  paper_certificate_valid: "
       << (log.paper_certificate_valid ? "true" : "false") << '\n'
       << "  lower_certificate: " << format_decimal(log.lower_certificate) << '\n'
       << "  upper_certificate: " << format_decimal(log.upper_certificate) << '\n'
       << "  certificate_gap: " << format_decimal(log.certificate_gap) << '\n'
       << "  iteration_runtime_sec: " << format_decimal(log.iteration_runtime_sec) << '\n'
       << "  defense_seeds: " << format_seed_set_preview(log.defense_seeds) << '\n'
       << "  attack_seeds: " << format_seed_set_preview(log.attack_seeds) << '\n';
}

CRDResult run_crd(
    const DirectedGraph& graph,
    const ConflictBackend& conflict,
    const Eigen::VectorXd& innate,
    const CascadeParams& params,
    const CRDOptions& options,
    std::ostream* progress_os) {
    validate_crd_inputs(graph, conflict, innate, options);

    AlgorithmTimer algorithm_timer;
    CRDResult result;
    result.variant = options.variant;
    std::vector<NodeId> current_defense;
    const ProgressCallback progress_callback = [&](const std::string& stage, double percent) {
        if (progress_os == nullptr) {
            return;
        }
        algorithm_timer.pause_for_output();
        print_progress_line(*progress_os, stage, percent);
        algorithm_timer.resume_after_output();
    };
    const bool paper_exact_mode = options.crd_mode == CRDMode::PaperExact;
    const bool strict_paper_certificates = strict_paper_certificate_mode(conflict, options);
    const std::vector<PaperExactRoundSamplePlan> paper_plans = paper_exact_mode
        ? plan_paper_exact_rr_samples(
              make_paper_exact_sample_plan_options(graph, conflict, options))
        : std::vector<PaperExactRoundSamplePlan>{};
    if (strict_paper_certificates) {
        for (const auto& plan : paper_plans) {
            if (plan.attack_rr_sample_cap_triggered || plan.defense_rr_sample_cap_triggered) {
                throw std::runtime_error(
                    "Strict paper certificate mode forbids capped RR sample plans");
            }
        }
    }

    std::vector<NodeId> initial_attack_seeds;
    double initial_attack_certificate = std::numeric_limits<double>::quiet_NaN();
    if (paper_exact_mode) {
        initial_attack_seeds = degree_out_attack_seed_set(graph, options.attack_budget);
        result.initial_attack_pool_policy = "degree_out_attack";
    } else {
        auto initial_attack_options = make_attack_options(options, 0);
        initial_attack_options.progress_callback = progress_callback;
        const auto initial_attack = run_attack_oracle(
            graph,
            conflict,
            innate,
            current_defense,
            params,
            initial_attack_options);
        initial_attack_seeds = initial_attack.attack_seeds;
        initial_attack_certificate = initial_attack.upper_certificate;
        result.initial_attack_pool_policy = "empty_defense_attack_oracle";
    }
    result.attack_pool.push_back(AttackPoolEntry{
        initial_attack_seeds,
        initial_attack_certificate
    });
    result.initial_attack_pool_size = result.attack_pool.size();

    if (progress_os != nullptr) {
        algorithm_timer.pause_for_output();
        *progress_os << "CRD warm start\n"
                     << "  initial_attack_pool_size: " << result.initial_attack_pool_size << '\n'
                     << "  initial_attack_pool_policy: " << result.initial_attack_pool_policy << '\n'
                     << "  initial_attack_seeds: "
                     << format_seed_set_preview(initial_attack_seeds) << '\n';
        print_progress_line(*progress_os, "CRD warm start", 35.0);
        algorithm_timer.resume_after_output();
    }

    const std::size_t total_iterations =
        paper_exact_mode ? paper_plans.size() : options.max_iterations;
    for (std::size_t iteration = 1; iteration <= total_iterations; ++iteration) {
        const auto iteration_start = Clock::now();
        const std::size_t attack_pool_size_before_defense = result.attack_pool.size();
        const PaperExactRoundSamplePlan* paper_plan =
            paper_exact_mode ? &paper_plans[iteration - 1] : nullptr;
        auto defense_options = paper_exact_mode
            ? make_paper_exact_defense_options(options, *paper_plan, iteration)
            : make_defense_options(options, iteration);
        auto attack_options = paper_exact_mode
            ? make_paper_exact_attack_options(options, *paper_plan, iteration)
            : make_attack_options(options, iteration);
        defense_options.progress_callback = progress_callback;
        attack_options.progress_callback = progress_callback;
        const auto defense_result = run_defense_oracle(
            graph,
            conflict,
            innate,
            result.attack_pool,
            current_defense,
            params,
            defense_options);
        current_defense = defense_result.defense_seeds;

        const auto attack_result = run_attack_oracle(
            graph,
            conflict,
            innate,
            current_defense,
            params,
            attack_options);

        const bool duplicate_attack =
            attack_pool_contains(result.attack_pool, attack_result.attack_seeds);
        std::string attack_pool_update;
        if (!paper_exact_mode) {
            const AttackPoolEntry new_attack{
                attack_result.attack_seeds,
                attack_result.upper_certificate
            };
            if (options.variant == CRDVariant::SingleAttackPool) {
                result.attack_pool.assign(1, new_attack);
                attack_pool_update = duplicate_attack
                    ? "replace_latest_duplicate"
                    : "replace_latest";
            } else if (duplicate_attack) {
                attack_pool_update = "duplicate_ignored";
            } else {
                result.attack_pool.push_back(new_attack);
                attack_pool_update = "append_cumulative";
            }
        } else {
            attack_pool_update = duplicate_attack
                ? "duplicate_ignored"
                : "append_if_continuing";
        }

        CRDIterationLog log;
        log.iteration = iteration;
        log.crd_mode = to_string(options.crd_mode);
        log.crd_variant = to_string(options.variant);
        if (paper_plan != nullptr) {
            log.epsilon_a = paper_plan->attack_oracle_epsilon;
            log.epsilon_b = paper_plan->defense_oracle_epsilon;
        } else {
            log.epsilon_a = attack_result.oracle_epsilon;
            log.epsilon_b = defense_result.mixed_certificate.oracle_epsilon;
        }
        log.attack_pool_size = result.attack_pool.size();
        log.attack_pool_size_before_defense = attack_pool_size_before_defense;
        log.attack_pool_size_after_attack = result.attack_pool.size();
        log.attack_pool_update = attack_pool_update;
        log.propagation_model = to_string(options.propagation_model);
        log.defense_seeds = current_defense;
        log.attack_seeds = attack_result.attack_seeds;
        log.defense_status = defense_result.status;
        log.defense_candidate_source = defense_result.candidate_source;
        log.defense_status_message = defense_result.status_message;
        log.defense_candidate_count = defense_result.defense_candidate_count;
        log.defense_candidate_sources = defense_result.defense_candidate_sources;
        log.defense_candidate_origin_rounds = defense_result.defense_candidate_origin_rounds;
        log.defense_candidate_values = defense_result.defense_candidate_values;
        log.selected_candidate_mixed_round = defense_result.selected_candidate_mixed_round;
        log.adaptive_candidate_count_before_deduplication =
            defense_result.adaptive_candidate_count_before_deduplication;
        log.adaptive_candidate_count_after_deduplication =
            defense_result.adaptive_candidate_count_after_deduplication;
        log.empirical_value_empirical = defense_result.empirical_value_empirical;
        log.empirical_value_surrogate = defense_result.empirical_value_surrogate;
        log.empirical_value_selected = defense_result.empirical_value_selected;
        log.mixed_distribution_type = defense_result.mixed_certificate.distribution_type;
        log.mixed_rounds = defense_result.mixed_certificate.mixed_rounds;
        log.selected_mixed_round =
            defense_result.mixed_certificate.distribution_type == "adaptive"
            ? defense_result.mixed_certificate.selected_round + 1
            : 0;
        log.round_lower_certificates = defense_result.mixed_certificate.round_lower_certificates;
        log.round_attack_distributions = defense_result.mixed_certificate.round_attack_distributions;
        log.round_surrogate_defenses = defense_result.mixed_certificate.round_greedy_defenses;
        log.attack_failure_probability = paper_plan != nullptr
            ? paper_plan->attack_failure_probability
            : allocated_attack_failure_probability(options);
        log.defense_failure_probability = paper_plan != nullptr
            ? paper_plan->defense_failure_probability
            : allocated_defense_failure_probability(options);
        log.delta_a = log.attack_failure_probability;
        log.delta_b = log.defense_failure_probability;
        log.attack_certificate_status = attack_result.certificate_status;
        log.attack_certificate_formula = attack_result.certificate_formula;
        log.attack_required_rr_samples = paper_plan != nullptr
            ? paper_plan->required_attack_rr_samples
            : attack_result.required_rr_samples;
        log.attack_rr_sample_cap = paper_plan != nullptr
            ? paper_plan->attack_rr_sample_cap
            : attack_result.rr_sample_cap;
        log.attack_rr_sample_cap_triggered = paper_plan != nullptr
            ? paper_plan->attack_rr_sample_cap_triggered
            : attack_result.rr_sample_cap_triggered;
        log.attack_generated_rr_samples = attack_result.generated_rr_samples;
        log.theta_a = log.attack_generated_rr_samples;
        log.defense_certificate_status =
            defense_result.mixed_certificate.certificate_status;
        log.lower_certificate_formula =
            defense_result.mixed_certificate.certificate_formula;
        log.defense_required_rr_samples = paper_plan != nullptr
            ? paper_plan->required_defense_rr_samples
            : defense_result.mixed_certificate.required_rr_samples;
        log.defense_rr_sample_cap = paper_plan != nullptr
            ? paper_plan->defense_rr_sample_cap
            : defense_result.mixed_certificate.rr_sample_cap;
        log.defense_rr_sample_cap_triggered = paper_plan != nullptr
            ? paper_plan->defense_rr_sample_cap_triggered
            : defense_result.mixed_certificate.rr_sample_cap_triggered;
        log.defense_generated_rr_samples = defense_result.mixed_certificate.generated_rr_samples;
        log.theta_b = log.defense_generated_rr_samples;
        log.theta_emp = defense_options.empirical_samples;
        log.theta_emp_selection = defense_result.empirical_selection_samples;
        log.theta_emp_validation = defense_result.empirical_validation_samples;
        log.empirical_validation_independent =
            defense_result.empirical_validation_independent;
        log.empirical_validation_streaming =
            defense_result.empirical_validation_streaming;
        log.unclamped_lower_certificate =
            defense_result.mixed_certificate.unclamped_lower_certificate;
        log.raw_lower_certificate = defense_result.mixed_certificate.raw_lower_certificate;
        log.defense_conservative_constant =
            defense_result.mixed_certificate.conservative_constant;
        log.defense_conservative_surrogate =
            defense_result.mixed_certificate.conservative_lower_surrogate_value;
        log.raw_upper_certificate = attack_result.raw_upper_certificate;
        log.attack_conservative_constant = attack_result.conservative_k;
        log.attack_conservative_surrogate = attack_result.conservative_surrogate_value;
        log.attack_constant_failure_probability = attack_result.constant_failure_probability;
        log.attack_surrogate_failure_probability = attack_result.surrogate_failure_probability;
        log.defense_distribution_failure_probability =
            defense_result.mixed_certificate.distribution_failure_probability;
        log.defense_constant_failure_probability =
            defense_result.mixed_certificate.constant_failure_probability;
        log.defense_surrogate_failure_probability =
            defense_result.mixed_certificate.surrogate_failure_probability;
        log.empirical_evaluator = defense_result.empirical_evaluator;
        log.sample_cap_triggered =
            log.attack_rr_sample_cap_triggered || log.defense_rr_sample_cap_triggered;
        log.paper_certificate_valid =
            paper_exact_mode &&
            !log.sample_cap_triggered &&
            log.empirical_evaluator == "pair_rr_dense" &&
            log.defense_status == DefenseOracleStatus::SearchCompleted;
        log.lower_certificate = defense_result.lower_certificate;
        log.upper_certificate = attack_result.upper_certificate;
        log.certificate_gap = certificate_gap(log.lower_certificate, log.upper_certificate);
        log.iteration_runtime_sec = elapsed_seconds(iteration_start, Clock::now());
        if (strict_paper_certificates && !log.paper_certificate_valid) {
            throw std::runtime_error(
                "Strict paper certificate mode encountered a non-paper-valid CRD iteration");
        }
        result.iterations.push_back(log);

        if (progress_os != nullptr) {
            algorithm_timer.pause_for_output();
            print_crd_iteration_log(*progress_os, log);
            print_progress_line(
                *progress_os,
                "CRD iteration " + std::to_string(iteration),
                35.0 + 55.0 * static_cast<double>(iteration) /
                    static_cast<double>(options.max_iterations));
            algorithm_timer.resume_after_output();
        }

        if ((!paper_exact_mode || log.paper_certificate_valid) &&
            finite_gap_within_tolerance(log.certificate_gap, options.epsilon_cert)) {
            break;
        }
        if (paper_exact_mode && !duplicate_attack) {
            result.attack_pool.push_back(AttackPoolEntry{
                attack_result.attack_seeds,
                attack_result.upper_certificate
            });
        }
    }

    if (paper_exact_mode) {
        bool found_valid = false;
        std::size_t selected_index = 0;
        double selected_gap = std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < result.iterations.size(); ++index) {
            const auto& log = result.iterations[index];
            if (!log.paper_certificate_valid || !std::isfinite(log.certificate_gap)) {
                continue;
            }
            if (!found_valid || log.certificate_gap < selected_gap) {
                found_valid = true;
                selected_index = index;
                selected_gap = log.certificate_gap;
            }
        }

        if (found_valid) {
            result.final_selection_policy = "minimum_valid_certificate_gap";
            select_result_iteration(result, selected_index);
        } else {
            bool found_finite = false;
            for (std::size_t index = 0; index < result.iterations.size(); ++index) {
                const double gap = result.iterations[index].certificate_gap;
                if (!std::isfinite(gap)) {
                    continue;
                }
                if (!found_finite || gap < selected_gap) {
                    found_finite = true;
                    selected_index = index;
                    selected_gap = gap;
                }
            }
            if (!found_finite) {
                selected_index = result.iterations.size() - 1;
            }
            result.final_selection_policy = found_finite
                ? "minimum_diagnostic_gap_no_valid_certificate"
                : "last_iteration_no_finite_certificate_gap";
            select_result_iteration(result, selected_index);
        }
        if (progress_os != nullptr) {
            algorithm_timer.pause_for_output();
            *progress_os << "CRD paper-exact final selection\n"
                         << "  policy: " << result.final_selection_policy << '\n'
                         << "  selected_iteration: "
                         << result.final_selected_iteration << '\n'
                         << "  selected_certificate_gap: "
                         << format_decimal(result.best_certificate_gap) << '\n';
            algorithm_timer.resume_after_output();
        }
    } else {
        const std::size_t holdout_samples = options.engineering_holdout_samples != 0
            ? options.engineering_holdout_samples
            : (options.empirical_samples != 0
                ? options.empirical_samples
                : options.defense_rr_samples);
        std::vector<std::vector<NodeId>> defense_candidates;
        defense_candidates.reserve(result.iterations.size());
        for (const auto& log : result.iterations) {
            defense_candidates.push_back(log.defense_seeds);
        }

        result.final_holdout_seed = options.seed ^ 0xd6e8feb86659fd93ULL;
        const auto holdout_result =
            evaluate_empirical_pool_candidates_pair_rr_streaming(
                options.propagation_model,
                graph,
                conflict,
                innate,
                result.attack_pool,
                defense_candidates,
                params,
                holdout_samples,
                result.final_holdout_seed,
                "CRD engineering final holdout pair-RR sampling",
                progress_callback);
        result.final_selection_policy = "minimum_independent_holdout_value";
        result.final_holdout_samples = holdout_result.generated_samples;
        result.final_holdout_independent = true;
        result.final_holdout_streaming = true;
        result.final_iteration_holdout_values.reserve(
            holdout_result.candidate_results.size());

        bool found_best = false;
        std::size_t selected_index = 0;
        double selected_value = std::numeric_limits<double>::infinity();
        for (std::size_t index = 0;
             index < holdout_result.candidate_results.size();
             ++index) {
            const double value = holdout_result.candidate_results[index].pool_value;
            result.final_iteration_holdout_values.push_back(value);
            if (std::isfinite(value) && (!found_best || value < selected_value)) {
                found_best = true;
                selected_index = index;
                selected_value = value;
            }
        }
        if (!found_best) {
            throw std::runtime_error(
                "CRD engineering holdout evaluation produced no finite candidate value");
        }
        result.final_selected_holdout_value = selected_value;
        select_result_iteration(result, selected_index);

        if (progress_os != nullptr) {
            algorithm_timer.pause_for_output();
            *progress_os << "CRD engineering final selection\n"
                         << "  policy: " << result.final_selection_policy << '\n'
                         << "  holdout_samples: " << result.final_holdout_samples << '\n'
                         << "  holdout_streaming: true\n"
                         << "  selected_iteration: "
                         << result.final_selected_iteration << '\n'
                         << "  selected_holdout_value: "
                         << format_decimal(result.final_selected_holdout_value) << '\n';
            algorithm_timer.resume_after_output();
        }
    }

    result.algorithm_runtime_sec = algorithm_timer.elapsed_seconds();
    return result;
}

}  // namespace rcd
