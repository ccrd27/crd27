#include "rcd/results_writer.hpp"

#include "rcd/defense_oracle.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace rcd {
namespace {

std::string join_path(const std::string& output_dir, const std::string& filename) {
    return (std::filesystem::path(output_dir) / filename).string();
}

void ensure_parent_directory(const std::string& path_string) {
    const std::filesystem::path path(path_string);
    const auto parent = path.parent_path();
    if (parent.empty()) {
        return;
    }

    std::error_code ec;
    if (std::filesystem::exists(parent, ec)) {
        if (!std::filesystem::is_directory(parent, ec)) {
            throw std::runtime_error("Output parent path exists but is not a directory: " + parent.string());
        }
        return;
    }
    if (!std::filesystem::create_directories(parent, ec)) {
        throw std::runtime_error("Failed to create output parent directory: " + parent.string());
    }
}

std::ofstream open_output_file(const std::string& path) {
    ensure_parent_directory(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    return output;
}

std::string csv_escape(const std::string& value) {
    bool needs_quotes = false;
    for (const char ch : value) {
        if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

std::vector<std::vector<std::string>> parse_csv_rows(const std::string& text) {
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (quoted) {
            if (ch == '"') {
                if (i + 1 < text.size() && text[i + 1] == '"') {
                    field.push_back('"');
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                field.push_back(ch);
            }
        } else if (ch == '"' && field.empty()) {
            quoted = true;
        } else if (ch == ',') {
            row.push_back(field);
            field.clear();
        } else if (ch == '\n') {
            if (!field.empty() && field.back() == '\r') {
                field.pop_back();
            }
            row.push_back(field);
            field.clear();
            rows.push_back(std::move(row));
            row.clear();
        } else {
            field.push_back(ch);
        }
    }
    if (!field.empty() || !row.empty()) {
        row.push_back(field);
        rows.push_back(std::move(row));
    }
    return rows;
}

void transpose_csv_file(const std::string& path, const std::string& scope) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to reopen CSV file for transposition: " + path);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto rows = parse_csv_rows(contents.str());
    if (rows.empty()) {
        return;
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to rewrite transposed CSV file: " + path);
    }
    output << "field,value\n";
    const auto& header = rows.front();
    for (std::size_t row_id = 1; row_id < rows.size(); ++row_id) {
        const auto& row = rows[row_id];
        for (std::size_t column = 0; column < header.size(); ++column) {
            const std::string column_name = header[column].empty()
                ? "record"
                : header[column];
            const std::string field_name = scope + ".row[" +
                std::to_string(row_id - 1) + "]." + column_name;
            const std::string value = column < row.size() ? row[column] : "";
            output << csv_escape(field_name) << ',' << csv_escape(value) << '\n';
        }
    }
}

std::string external_seed_set(
    const DirectedGraph& graph,
    const std::vector<NodeId>& seeds,
    bool full_output = false) {
    std::ostringstream os;
    os << '[';
    const std::size_t displayed_count = full_output
        ? seeds.size()
        : std::min<std::size_t>(seeds.size(), 20);
    for (std::size_t i = 0; i < displayed_count; ++i) {
        if (i > 0) {
            os << ',';
        }
        const auto internal_id = static_cast<std::size_t>(seeds[i]);
        if (internal_id >= graph.external_ids.size()) {
            throw std::out_of_range("Seed node is out of range for external id mapping");
        }
        os << graph.external_ids[internal_id];
    }
    os << ']';
    if (!full_output && seeds.size() > displayed_count) {
        os << " (total_seeds=" << seeds.size()
           << "; showing_first=" << displayed_count << ')';
    }
    return os.str();
}

void write_seed_lines(
    std::ostream& os,
    const DirectedGraph& graph,
    const std::string& label,
    const std::vector<NodeId>& seeds,
    bool full_output = false) {
    const std::string internal = full_output
        ? format_seed_set(seeds)
        : format_seed_set_preview(seeds);
    os << label << "_internal=" << internal << '\n';
    os << label << "_external=" << external_seed_set(graph, seeds, full_output) << '\n';
}

void write_mc_result_lines(
    std::ostream& os,
    const std::string& label,
    const MonteCarloResult& result) {
    os << label << "_mean=" << format_decimal(result.mean) << '\n';
    os << label << "_variance=" << format_decimal(result.variance) << '\n';
    os << label << "_standard_error=" << format_decimal(result.standard_error) << '\n';
    os << label << "_ci95_half_width=" << format_decimal(result.ci95_half_width) << '\n';
    os << label << "_simulations=" << result.simulations << '\n';
    os << label << "_stop_status=" << result.stop_status << '\n';
}

std::string optional_mc_mean_to_string(bool available, const MonteCarloResult& result) {
    if (!available) {
        return "nan";
    }
    return format_decimal(result.mean);
}

std::string optional_mc_value_to_string(bool available, double value) {
    return available ? format_decimal(value) : "nan";
}

std::string optional_mc_count_to_string(bool available, std::size_t value) {
    return available ? std::to_string(value) : "0";
}

std::string optional_mc_status_to_string(bool available, const std::string& value) {
    return available ? value : "not_available";
}

std::string string_list(const std::vector<std::string>& values) {
    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << values[i];
    }
    os << ']';
    return os.str();
}

std::string double_list(const std::vector<double>& values) {
    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << format_decimal(values[i]);
    }
    os << ']';
    return os.str();
}

std::string nested_double_list(const std::vector<std::vector<double>>& values) {
    std::vector<std::string> formatted;
    formatted.reserve(values.size());
    for (const auto& value : values) {
        formatted.push_back(double_list(value));
    }
    return string_list(formatted);
}

std::string seed_set_list(const std::vector<std::vector<NodeId>>& values) {
    std::vector<std::string> formatted;
    formatted.reserve(values.size());
    for (const auto& value : values) {
        formatted.push_back(format_seed_set_preview(value));
    }
    return string_list(formatted);
}

void write_metadata_lines(std::ostream& output, const ExperimentMetadata& metadata) {
    output << "graph_G_file=" << metadata.dataset_path << '\n';
    output << "output_dir=" << metadata.output_dir << '\n';
    output << "output_prefix=" << metadata.output_prefix << '\n';
    output << "s_file=" << metadata.innate_opinion_file << '\n';
    output << "p_file=" << metadata.edge_weight_file << '\n';
    output << "rho_u_file=" << metadata.rho_file << '\n';
    output << "w_file=" << metadata.fj_weight_file << '\n';
    output << "algorithm=" << metadata.algorithm << '\n';
    output << "metric=" << metadata.metric << '\n';
    output << "propagation_model=" << metadata.propagation_model << '\n';
    output << "conflict_backend=" << metadata.conflict_backend << '\n';
    output << "conflict_backend_approximate="
           << (metadata.conflict_backend_approximate ? "true" : "false") << '\n';
    output << "conflict_beta_bound=" << format_decimal(metadata.conflict_beta_bound) << '\n';
    output << "amgcl_tolerance=" << format_decimal(metadata.amgcl_tolerance) << '\n';
    output << "amgcl_max_iterations=" << metadata.amgcl_max_iterations << '\n';
    output << "amgcl_last_iterations=" << metadata.amgcl_last_iterations << '\n';
    output << "amgcl_last_relative_error="
           << format_decimal(metadata.amgcl_last_relative_error) << '\n';
    output << "seed=" << metadata.seed << '\n';
    output << "graph_nodes=" << metadata.graph_nodes << '\n';
    output << "graph_directed_edges=" << metadata.graph_directed_edges << '\n';
    output << "graph_skipped_self_loops=" << metadata.graph_skipped_self_loops << '\n';
    output << "graph_skipped_duplicate_edges=" << metadata.graph_skipped_duplicate_edges << '\n';
    output << "fj_undirected_edges=" << metadata.fj_undirected_edges << '\n';
    output << "mc_simulations=" << metadata.mc_simulations << '\n';
    output << "mc_evaluation_mode=" << metadata.mc_evaluation_mode << '\n';
    output << "mc_min_simulations=" << metadata.mc_min_simulations << '\n';
    output << "mc_check_interval=" << metadata.mc_check_interval << '\n';
    output << "mc_ci_relative_half_width="
           << format_decimal(metadata.mc_ci_relative_half_width) << '\n';
    output << "final_attack_response=" << metadata.final_attack_response << '\n';
    output << "final_attack_response_rr_samples="
           << metadata.final_attack_response_rr_samples << '\n';
    output << "k_A=" << metadata.k_a << '\n';
    output << "k_B=" << metadata.k_b << '\n';
    output << "crd_mode=" << metadata.crd_mode << '\n';
    output << "crd_variant=" << metadata.crd_variant << '\n';
    output << "T=" << metadata.crd_iterations << '\n';
    output << "paper_crd_schedule=" << metadata.paper_crd_schedule << '\n';
    output << "initial_attack_pool_policy=" << metadata.initial_attack_pool_policy << '\n';
    output << "baseline_iterations=" << metadata.baseline_iterations << '\n';
    output << "fjc_pair_rr_samples=" << metadata.fjc_pair_rr_samples << '\n';
    output << "fjc_linear_rr_samples=" << metadata.fjc_linear_rr_samples << '\n';
    output << "hist_rr_samples_floor=" << metadata.hist_rr_samples_floor << '\n';
    output << "baseline_max_adaptive_rounds=" << metadata.baseline_max_adaptive_rounds << '\n';
    output << "theta_A=" << metadata.attack_rr_samples << '\n';
    output << "theta_B=" << metadata.defense_rr_samples << '\n';
    output << "max_rr_samples=" << metadata.max_rr_samples << '\n';
    output << "theta_emp=" << metadata.empirical_samples << '\n';
    output << "mixed_distribution=" << metadata.mixed_distribution << '\n';
    output << "mixed_rounds=" << metadata.mixed_rounds << '\n';
    output << "mixed_learning_rate=" << format_decimal(metadata.mixed_learning_rate) << '\n';
    output << "alpha_A=" << format_decimal(metadata.alpha_a) << '\n';
    output << "alpha_B=" << format_decimal(metadata.alpha_b) << '\n';
    output << "default_rho_u=" << format_decimal(metadata.rho) << '\n';
    output << "epsilon=" << format_decimal(metadata.epsilon_cert) << '\n';
    output << "delta=" << format_decimal(metadata.failure_probability) << '\n';
    output << "memory_rss_before_algorithm_kib="
           << metadata.memory_rss_before_algorithm_kib << '\n';
    output << "memory_rss_after_algorithm_kib="
           << metadata.memory_rss_after_algorithm_kib << '\n';
    output << "memory_peak_rss_kib=" << metadata.memory_peak_rss_kib << '\n';
}

void write_metadata_csv_header(std::ostream& output) {
    output << ",graph_G_file,output_dir,output_prefix,s_file,p_file,"
           << "rho_u_file,w_file,algorithm,metric,propagation_model,conflict_backend,"
           << "conflict_backend_approximate,conflict_beta_bound,amgcl_tolerance,"
           << "amgcl_max_iterations,amgcl_last_iterations,amgcl_last_relative_error,"
           << "seed,graph_nodes,graph_directed_edges,graph_skipped_self_loops,"
           << "graph_skipped_duplicate_edges,fj_undirected_edges,"
           << "mc_simulations,mc_evaluation_mode,mc_min_simulations,mc_check_interval,mc_ci_relative_half_width,"
           << "final_attack_response,final_attack_response_rr_samples,"
           << "k_A,k_B,crd_mode,crd_variant,T,paper_crd_schedule,"
           << "initial_attack_pool_policy,"
           << "baseline_iterations,"
           << "fjc_pair_rr_samples,fjc_linear_rr_samples,hist_rr_samples_floor,"
           << "baseline_max_adaptive_rounds,"
           << "theta_A,"
           << "theta_B,max_rr_samples,theta_emp,"
           << "mixed_distribution,mixed_rounds,mixed_learning_rate,"
           << "alpha_A,alpha_B,default_rho_u,epsilon,delta,"
           << "memory_rss_before_algorithm_kib,memory_rss_after_algorithm_kib,"
           << "memory_peak_rss_kib";
}

void write_metadata_csv_values(std::ostream& output, const ExperimentMetadata& metadata) {
    output << ','
           << csv_escape(metadata.dataset_path) << ','
           << csv_escape(metadata.output_dir) << ','
           << csv_escape(metadata.output_prefix) << ','
           << csv_escape(metadata.innate_opinion_file) << ','
           << csv_escape(metadata.edge_weight_file) << ','
           << csv_escape(metadata.rho_file) << ','
           << csv_escape(metadata.fj_weight_file) << ','
           << csv_escape(metadata.algorithm) << ','
           << csv_escape(metadata.metric) << ','
           << csv_escape(metadata.propagation_model) << ','
           << csv_escape(metadata.conflict_backend) << ','
           << (metadata.conflict_backend_approximate ? "true" : "false") << ','
           << format_decimal(metadata.conflict_beta_bound) << ','
           << format_decimal(metadata.amgcl_tolerance) << ','
           << metadata.amgcl_max_iterations << ','
           << metadata.amgcl_last_iterations << ','
           << format_decimal(metadata.amgcl_last_relative_error) << ','
           << metadata.seed << ','
           << metadata.graph_nodes << ','
           << metadata.graph_directed_edges << ','
           << metadata.graph_skipped_self_loops << ','
           << metadata.graph_skipped_duplicate_edges << ','
           << metadata.fj_undirected_edges << ','
           << metadata.mc_simulations << ','
           << csv_escape(metadata.mc_evaluation_mode) << ','
           << metadata.mc_min_simulations << ','
           << metadata.mc_check_interval << ','
           << format_decimal(metadata.mc_ci_relative_half_width) << ','
           << csv_escape(metadata.final_attack_response) << ','
           << metadata.final_attack_response_rr_samples << ','
           << metadata.k_a << ','
           << metadata.k_b << ','
           << csv_escape(metadata.crd_mode) << ','
           << csv_escape(metadata.crd_variant) << ','
           << metadata.crd_iterations << ','
           << csv_escape(metadata.paper_crd_schedule) << ','
           << csv_escape(metadata.initial_attack_pool_policy) << ','
           << metadata.baseline_iterations << ','
           << metadata.fjc_pair_rr_samples << ','
           << metadata.fjc_linear_rr_samples << ','
           << metadata.hist_rr_samples_floor << ','
           << metadata.baseline_max_adaptive_rounds << ','
           << metadata.attack_rr_samples << ','
           << metadata.defense_rr_samples << ','
           << metadata.max_rr_samples << ','
           << metadata.empirical_samples << ','
           << csv_escape(metadata.mixed_distribution) << ','
           << metadata.mixed_rounds << ','
           << format_decimal(metadata.mixed_learning_rate) << ','
           << format_decimal(metadata.alpha_a) << ','
           << format_decimal(metadata.alpha_b) << ','
           << format_decimal(metadata.rho) << ','
           << format_decimal(metadata.epsilon_cert) << ','
           << format_decimal(metadata.failure_probability) << ','
           << metadata.memory_rss_before_algorithm_kib << ','
           << metadata.memory_rss_after_algorithm_kib << ','
           << metadata.memory_peak_rss_kib;
}

}  // namespace

ResultOutputPaths make_result_output_paths(
    const std::string& output_dir,
    const std::string& output_prefix) {
    if (output_prefix.empty()) {
        throw std::invalid_argument("Output prefix must not be empty");
    }

    ResultOutputPaths paths;
    paths.txt_path = join_path(output_dir, output_prefix + ".txt");
    paths.csv_path = join_path(output_dir, output_prefix + ".csv");
    paths.seeds_path = join_path(output_dir, output_prefix + "_seeds.txt");
    return paths;
}

void write_crd_result_files(
    const CRDResult& result,
    const DirectedGraph& graph,
    const ResultOutputPaths& paths,
    const ExperimentMetadata& metadata) {
    {
        auto output = open_output_file(paths.txt_path);
        output << "CRD result summary\n";
        output << "crd_variant=" << to_string(result.variant) << '\n';
        output << "iterations=" << result.iterations.size() << '\n';
        output << "initial_attack_pool_policy=" << result.initial_attack_pool_policy << '\n';
        output << "initial_attack_pool_size=" << result.initial_attack_pool_size << '\n';
        output << "attack_pool_size=" << result.attack_pool.size() << '\n';
        output << "certificate_status="
               << (result.variant == CRDVariant::LinearApprox
                   ? "not_applicable_linear_approx"
                   : "reported_per_iteration") << '\n';
        output << "best_lower_certificate=" << format_decimal(result.best_lower_certificate) << '\n';
        output << "best_upper_certificate=" << format_decimal(result.best_upper_certificate) << '\n';
        output << "best_certificate_gap=" << format_decimal(result.best_certificate_gap) << '\n';
        output << "final_selection_policy=" << result.final_selection_policy << '\n';
        output << "final_selected_iteration=" << result.final_selected_iteration << '\n';
        output << "final_holdout_samples=" << result.final_holdout_samples << '\n';
        output << "final_holdout_seed=" << result.final_holdout_seed << '\n';
        output << "final_holdout_independent="
               << (result.final_holdout_independent ? "true" : "false") << '\n';
        output << "final_holdout_streaming="
               << (result.final_holdout_streaming ? "true" : "false") << '\n';
        output << "final_selected_holdout_value="
               << format_decimal(result.final_selected_holdout_value) << '\n';
        output << "final_iteration_holdout_values="
               << double_list(result.final_iteration_holdout_values) << '\n';
        output << "algorithm_runtime_sec=" << format_decimal(result.algorithm_runtime_sec) << '\n';
        output << "mc_evaluation_available=" << (result.evaluation.available ? "true" : "false") << '\n';
        if (result.evaluation.available) {
            write_mc_result_lines(
                output, "mc_value_defense_vs_attack", result.evaluation.defense_vs_attack);
            write_mc_result_lines(
                output, "mc_value_no_defense_attack_only", result.evaluation.no_defense_attack_only);
            write_mc_result_lines(
                output, "mc_value_defense_only", result.evaluation.defense_only);
            write_seed_lines(output, graph, "attack_only_seeds", result.evaluation.attack_only_seeds);
        }
        write_seed_lines(output, graph, "defense_seeds", result.defense_seeds);
        write_seed_lines(output, graph, "attack_seeds", result.attack_seeds);

        output << "\nExperiment metadata\n";
        write_metadata_lines(output, metadata);

        output << "\nCRD iteration logs\n";
        for (const auto& log : result.iterations) {
            output << "iteration=" << log.iteration << '\n';
            output << "crd_mode=" << log.crd_mode << '\n';
            output << "crd_variant=" << log.crd_variant << '\n';
            output << "epsilon_A=" << format_decimal(log.epsilon_a) << '\n';
            output << "epsilon_B=" << format_decimal(log.epsilon_b) << '\n';
            output << "delta_A=" << format_decimal(log.delta_a) << '\n';
            output << "delta_B=" << format_decimal(log.delta_b) << '\n';
            output << "theta_A=" << log.theta_a << '\n';
            output << "theta_B=" << log.theta_b << '\n';
            output << "theta_emp=" << log.theta_emp << '\n';
            output << "theta_emp_selection=" << log.theta_emp_selection << '\n';
            output << "theta_emp_validation=" << log.theta_emp_validation << '\n';
            output << "empirical_validation_independent="
                   << (log.empirical_validation_independent ? "true" : "false") << '\n';
            output << "empirical_validation_streaming="
                   << (log.empirical_validation_streaming ? "true" : "false") << '\n';
            output << "attack_pool_size=" << log.attack_pool_size << '\n';
            output << "attack_pool_size_before_defense="
                   << log.attack_pool_size_before_defense << '\n';
            output << "attack_pool_size_after_attack="
                   << log.attack_pool_size_after_attack << '\n';
            output << "attack_pool_update=" << log.attack_pool_update << '\n';
            output << "propagation_model=" << log.propagation_model << '\n';
            output << "defense_status=" << to_string(log.defense_status) << '\n';
            output << "defense_candidate_source=" << to_string(log.defense_candidate_source) << '\n';
            output << "defense_candidate_count=" << log.defense_candidate_count << '\n';
            output << "defense_candidate_sources="
                   << string_list(log.defense_candidate_sources) << '\n';
            output << "defense_candidate_origin_rounds="
                   << string_list(log.defense_candidate_origin_rounds) << '\n';
            output << "defense_candidate_values="
                   << double_list(log.defense_candidate_values) << '\n';
            output << "selected_candidate_mixed_round="
                   << log.selected_candidate_mixed_round << '\n';
            output << "adaptive_candidate_count_before_deduplication="
                   << log.adaptive_candidate_count_before_deduplication << '\n';
            output << "adaptive_candidate_count_after_deduplication="
                   << log.adaptive_candidate_count_after_deduplication << '\n';
            output << "defense_status_message=" << log.defense_status_message << '\n';
            output << "empirical_value_empirical="
                   << format_decimal(log.empirical_value_empirical) << '\n';
            output << "empirical_value_surrogate="
                   << format_decimal(log.empirical_value_surrogate) << '\n';
            output << "empirical_value_selected="
                   << format_decimal(log.empirical_value_selected) << '\n';
            output << "mixed_distribution_type=" << log.mixed_distribution_type << '\n';
            output << "mixed_rounds=" << log.mixed_rounds << '\n';
            output << "selected_mixed_round=" << log.selected_mixed_round << '\n';
            output << "round_lower_certificates="
                   << double_list(log.round_lower_certificates) << '\n';
            output << "round_attack_distributions="
                   << nested_double_list(log.round_attack_distributions) << '\n';
            output << "round_surrogate_defenses="
                   << seed_set_list(log.round_surrogate_defenses) << '\n';
            output << "attack_failure_probability="
                   << format_decimal(log.attack_failure_probability) << '\n';
            output << "defense_failure_probability="
                   << format_decimal(log.defense_failure_probability) << '\n';
            output << "attack_certificate_status=" << log.attack_certificate_status << '\n';
            output << "attack_certificate_formula=" << log.attack_certificate_formula << '\n';
            output << "attack_required_rr_samples=" << log.attack_required_rr_samples << '\n';
            output << "attack_rr_sample_cap=" << log.attack_rr_sample_cap << '\n';
            output << "attack_rr_sample_cap_triggered="
                   << (log.attack_rr_sample_cap_triggered ? "true" : "false") << '\n';
            output << "attack_generated_rr_samples=" << log.attack_generated_rr_samples << '\n';
            output << "defense_certificate_status=" << log.defense_certificate_status << '\n';
            output << "lower_certificate_formula=" << log.lower_certificate_formula << '\n';
            output << "defense_required_rr_samples=" << log.defense_required_rr_samples << '\n';
            output << "defense_rr_sample_cap=" << log.defense_rr_sample_cap << '\n';
            output << "defense_rr_sample_cap_triggered="
                   << (log.defense_rr_sample_cap_triggered ? "true" : "false") << '\n';
            output << "defense_generated_rr_samples=" << log.defense_generated_rr_samples << '\n';
            output << "unclamped_lower_certificate="
                   << format_decimal(log.unclamped_lower_certificate) << '\n';
            output << "raw_lower_certificate=" << format_decimal(log.raw_lower_certificate) << '\n';
            output << "defense_conservative_constant="
                   << format_decimal(log.defense_conservative_constant) << '\n';
            output << "defense_conservative_surrogate="
                   << format_decimal(log.defense_conservative_surrogate) << '\n';
            output << "raw_upper_certificate=" << format_decimal(log.raw_upper_certificate) << '\n';
            output << "attack_conservative_constant="
                   << format_decimal(log.attack_conservative_constant) << '\n';
            output << "attack_conservative_surrogate="
                   << format_decimal(log.attack_conservative_surrogate) << '\n';
            output << "attack_constant_failure_probability="
                   << format_decimal(log.attack_constant_failure_probability) << '\n';
            output << "attack_surrogate_failure_probability="
                   << format_decimal(log.attack_surrogate_failure_probability) << '\n';
            output << "defense_distribution_failure_probability="
                   << format_decimal(log.defense_distribution_failure_probability) << '\n';
            output << "defense_constant_failure_probability="
                   << format_decimal(log.defense_constant_failure_probability) << '\n';
            output << "defense_surrogate_failure_probability="
                   << format_decimal(log.defense_surrogate_failure_probability) << '\n';
            output << "empirical_evaluator=" << log.empirical_evaluator << '\n';
            output << "sample_cap_triggered="
                   << (log.sample_cap_triggered ? "true" : "false") << '\n';
            output << "paper_certificate_valid="
                   << (log.paper_certificate_valid ? "true" : "false") << '\n';
            output << "lower_certificate=" << format_decimal(log.lower_certificate) << '\n';
            output << "upper_certificate=" << format_decimal(log.upper_certificate) << '\n';
            output << "certificate_gap=" << format_decimal(log.certificate_gap) << '\n';
            output << "iteration_runtime_sec=" << format_decimal(log.iteration_runtime_sec) << '\n';
            write_seed_lines(output, graph, "defense_seeds", log.defense_seeds);
            write_seed_lines(output, graph, "attack_seeds", log.attack_seeds);
            output << '\n';
        }
    }

    {
        auto output = open_output_file(paths.csv_path);
        output << "iteration,crd_mode,crd_variant,"
               << "epsilon_A,epsilon_B,delta_A,delta_B,theta_A,theta_B,theta_emp,"
               << "theta_emp_selection,theta_emp_validation,empirical_validation_independent,"
               << "empirical_validation_streaming,"
               << "attack_pool_size,attack_pool_size_before_defense,attack_pool_size_after_attack,attack_pool_update,"
               << "propagation_model,defense_status,defense_candidate_source,"
               << "defense_candidate_count,defense_candidate_sources,defense_candidate_origin_rounds,"
               << "defense_candidate_values,selected_candidate_mixed_round,"
               << "adaptive_candidate_count_before_deduplication,adaptive_candidate_count_after_deduplication,"
               << "defense_status_message,"
               << "empirical_value_empirical,empirical_value_surrogate,empirical_value_selected,"
               << "mixed_distribution_type,mixed_rounds,selected_mixed_round,"
               << "round_lower_certificates,round_attack_distributions,round_surrogate_defenses,"
               << "attack_failure_probability,defense_failure_probability,"
               << "attack_certificate_status,attack_certificate_formula,"
               << "attack_required_rr_samples,attack_rr_sample_cap,attack_rr_sample_cap_triggered,"
               << "attack_generated_rr_samples,"
               << "defense_certificate_status,lower_certificate_formula,"
               << "defense_required_rr_samples,defense_rr_sample_cap,defense_rr_sample_cap_triggered,"
               << "defense_generated_rr_samples,"
               << "unclamped_lower_certificate,raw_lower_certificate,defense_conservative_constant,"
               << "defense_conservative_surrogate,raw_upper_certificate,"
               << "attack_conservative_constant,attack_conservative_surrogate,"
               << "attack_constant_failure_probability,attack_surrogate_failure_probability,"
               << "defense_distribution_failure_probability,defense_constant_failure_probability,"
               << "defense_surrogate_failure_probability,"
               << "empirical_evaluator,sample_cap_triggered,paper_certificate_valid,"
               << "lower_certificate,upper_certificate,certificate_gap,iteration_runtime_sec,"
               << "final_selection_policy,final_selected_iteration,final_holdout_samples,"
               << "final_holdout_seed,final_holdout_independent,final_holdout_streaming,"
               << "final_selected_holdout_value,final_iteration_holdout_values,"
               << "mc_value_defense_vs_attack,mc_value_no_defense_attack_only,mc_value_defense_only,"
               << "mc_value_defense_vs_attack_variance,mc_value_defense_vs_attack_standard_error,"
               << "mc_value_defense_vs_attack_ci95_half_width,mc_value_defense_vs_attack_simulations,mc_value_defense_vs_attack_stop_status,"
               << "mc_value_no_defense_attack_only_variance,mc_value_no_defense_attack_only_standard_error,"
               << "mc_value_no_defense_attack_only_ci95_half_width,mc_value_no_defense_attack_only_simulations,mc_value_no_defense_attack_only_stop_status,"
               << "mc_value_defense_only_variance,mc_value_defense_only_standard_error,"
               << "mc_value_defense_only_ci95_half_width,mc_value_defense_only_simulations,mc_value_defense_only_stop_status,"
               << "defense_seeds_internal,defense_seeds_external,"
               << "attack_seeds_internal,attack_seeds_external,"
               << "attack_only_seeds_internal,attack_only_seeds_external";
        write_metadata_csv_header(output);
        output << '\n';
        for (const auto& log : result.iterations) {
            output << log.iteration << ','
                   << csv_escape(log.crd_mode) << ','
                   << csv_escape(log.crd_variant) << ','
                   << format_decimal(log.epsilon_a) << ','
                   << format_decimal(log.epsilon_b) << ','
                   << format_decimal(log.delta_a) << ','
                   << format_decimal(log.delta_b) << ','
                   << log.theta_a << ','
                   << log.theta_b << ','
                   << log.theta_emp << ','
                   << log.theta_emp_selection << ','
                   << log.theta_emp_validation << ','
                   << (log.empirical_validation_independent ? "true" : "false") << ','
                   << (log.empirical_validation_streaming ? "true" : "false") << ','
                   << log.attack_pool_size << ','
                   << log.attack_pool_size_before_defense << ','
                   << log.attack_pool_size_after_attack << ','
                   << csv_escape(log.attack_pool_update) << ','
                   << csv_escape(log.propagation_model) << ','
                   << csv_escape(to_string(log.defense_status)) << ','
                   << csv_escape(to_string(log.defense_candidate_source)) << ','
                   << log.defense_candidate_count << ','
                   << csv_escape(string_list(log.defense_candidate_sources)) << ','
                   << csv_escape(string_list(log.defense_candidate_origin_rounds)) << ','
                   << csv_escape(double_list(log.defense_candidate_values)) << ','
                   << csv_escape(log.selected_candidate_mixed_round) << ','
                   << log.adaptive_candidate_count_before_deduplication << ','
                   << log.adaptive_candidate_count_after_deduplication << ','
                   << csv_escape(log.defense_status_message) << ','
                   << format_decimal(log.empirical_value_empirical) << ','
                   << format_decimal(log.empirical_value_surrogate) << ','
                   << format_decimal(log.empirical_value_selected) << ','
                   << csv_escape(log.mixed_distribution_type) << ','
                   << log.mixed_rounds << ','
                   << log.selected_mixed_round << ','
                   << csv_escape(double_list(log.round_lower_certificates)) << ','
                   << csv_escape(nested_double_list(log.round_attack_distributions)) << ','
                   << csv_escape(seed_set_list(log.round_surrogate_defenses)) << ','
                   << format_decimal(log.attack_failure_probability) << ','
                   << format_decimal(log.defense_failure_probability) << ','
                   << csv_escape(log.attack_certificate_status) << ','
                   << csv_escape(log.attack_certificate_formula) << ','
                   << log.attack_required_rr_samples << ','
                   << log.attack_rr_sample_cap << ','
                   << (log.attack_rr_sample_cap_triggered ? "true" : "false") << ','
                   << log.attack_generated_rr_samples << ','
                   << csv_escape(log.defense_certificate_status) << ','
                   << csv_escape(log.lower_certificate_formula) << ','
                   << log.defense_required_rr_samples << ','
                   << log.defense_rr_sample_cap << ','
                   << (log.defense_rr_sample_cap_triggered ? "true" : "false") << ','
                   << log.defense_generated_rr_samples << ','
                   << format_decimal(log.unclamped_lower_certificate) << ','
                   << format_decimal(log.raw_lower_certificate) << ','
                   << format_decimal(log.defense_conservative_constant) << ','
                   << format_decimal(log.defense_conservative_surrogate) << ','
                   << format_decimal(log.raw_upper_certificate) << ','
                   << format_decimal(log.attack_conservative_constant) << ','
                   << format_decimal(log.attack_conservative_surrogate) << ','
                   << format_decimal(log.attack_constant_failure_probability) << ','
                   << format_decimal(log.attack_surrogate_failure_probability) << ','
                   << format_decimal(log.defense_distribution_failure_probability) << ','
                   << format_decimal(log.defense_constant_failure_probability) << ','
                   << format_decimal(log.defense_surrogate_failure_probability) << ','
                   << csv_escape(log.empirical_evaluator) << ','
                   << (log.sample_cap_triggered ? "true" : "false") << ','
                   << (log.paper_certificate_valid ? "true" : "false") << ','
                   << format_decimal(log.lower_certificate) << ','
                   << format_decimal(log.upper_certificate) << ','
                   << format_decimal(log.certificate_gap) << ','
                   << format_decimal(log.iteration_runtime_sec) << ','
                   << csv_escape(result.final_selection_policy) << ','
                   << result.final_selected_iteration << ','
                   << result.final_holdout_samples << ','
                   << result.final_holdout_seed << ','
                   << (result.final_holdout_independent ? "true" : "false") << ','
                   << (result.final_holdout_streaming ? "true" : "false") << ','
                   << format_decimal(result.final_selected_holdout_value) << ','
                   << csv_escape(double_list(result.final_iteration_holdout_values)) << ','
                   << optional_mc_mean_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack) << ','
                   << optional_mc_mean_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only) << ','
                   << optional_mc_mean_to_string(
                          result.evaluation.available, result.evaluation.defense_only) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack.variance) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack.standard_error) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack.ci95_half_width) << ','
                   << optional_mc_count_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack.simulations) << ','
                   << csv_escape(optional_mc_status_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack.stop_status)) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only.variance) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only.standard_error) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only.ci95_half_width) << ','
                   << optional_mc_count_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only.simulations) << ','
                   << csv_escape(optional_mc_status_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only.stop_status)) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_only.variance) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_only.standard_error) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_only.ci95_half_width) << ','
                   << optional_mc_count_to_string(
                          result.evaluation.available, result.evaluation.defense_only.simulations) << ','
                   << csv_escape(optional_mc_status_to_string(
                          result.evaluation.available, result.evaluation.defense_only.stop_status)) << ','
                   << csv_escape(format_seed_set_preview(log.defense_seeds)) << ','
                   << csv_escape(external_seed_set(graph, log.defense_seeds)) << ','
                   << csv_escape(format_seed_set_preview(log.attack_seeds)) << ','
                   << csv_escape(external_seed_set(graph, log.attack_seeds)) << ','
                   << csv_escape(format_seed_set_preview(result.evaluation.attack_only_seeds)) << ','
                   << csv_escape(external_seed_set(graph, result.evaluation.attack_only_seeds));
            write_metadata_csv_values(output, metadata);
            output << '\n';
        }
    }

    transpose_csv_file(paths.csv_path, "crd");

    {
        auto output = open_output_file(paths.seeds_path);
        write_seed_lines(output, graph, "final_defense_seeds", result.defense_seeds, true);
        write_seed_lines(output, graph, "final_attack_seeds", result.attack_seeds, true);
        if (result.evaluation.available) {
            write_seed_lines(
                output, graph, "attack_only_seeds", result.evaluation.attack_only_seeds, true);
        }
    }
}

void write_baseline_result_files(
    const BaselineResult& result,
    const DirectedGraph& graph,
    const ResultOutputPaths& paths,
    const ExperimentMetadata& metadata) {
    {
        auto output = open_output_file(paths.txt_path);
        output << "Baseline result summary\n";
        output << "algorithm=" << to_string(result.algorithm) << '\n';
        output << "selection_objective=" << result.selection_objective << '\n';
        output << "uses_attack_pool=" << (result.uses_attack_pool ? "true" : "false") << '\n';
        output << "uses_fj_objective_in_selection="
               << (result.uses_fj_objective_in_selection ? "true" : "false") << '\n';
        output << "uses_certificate=" << (result.uses_certificate ? "true" : "false") << '\n';
        output << "certificate_status=" << result.certificate_status << '\n';
        output << "selection_backend=" << result.selection_backend << '\n';
        output << "selection_context=" << result.selection_context << '\n';
        output << "lower_bound=not_applicable\n";
        output << "upper_bound=not_applicable\n";
        output << "certificate_gap=not_applicable\n";
        output << "baseline_iterations=" << result.iterations.size() << '\n';
        output << "fjc_pair_rr_samples=" << result.fjc_pair_rr_samples << '\n';
        output << "fjc_linear_rr_samples=" << result.fjc_linear_rr_samples << '\n';
        output << "hist_rr_samples_floor=" << result.hist_rr_samples_floor << '\n';
        output << "baseline_sampling_mode=" << to_string(result.sampling_mode) << '\n';
        output << "selection_mc_simulations=" << result.selection_mc_simulations << '\n';
        output << "baseline_max_adaptive_rounds=" << result.max_adaptive_rounds << '\n';
        output << "algorithm_runtime_sec=" << format_decimal(result.algorithm_runtime_sec) << '\n';
        output << "mc_evaluation_available=" << (result.evaluation.available ? "true" : "false") << '\n';
        if (result.evaluation.available) {
            write_mc_result_lines(
                output, "mc_value_defense_vs_attack", result.evaluation.defense_vs_attack);
            write_mc_result_lines(
                output, "mc_value_no_defense_attack_only", result.evaluation.no_defense_attack_only);
            write_mc_result_lines(
                output, "mc_value_defense_only", result.evaluation.defense_only);
            write_seed_lines(output, graph, "attack_only_seeds", result.evaluation.attack_only_seeds);
        }
        write_seed_lines(output, graph, "defense_seeds", result.defense_seeds);
        write_seed_lines(output, graph, "attack_seeds", result.attack_seeds);

        output << "\nExperiment metadata\n";
        write_metadata_lines(output, metadata);

        output << "\nBaseline iteration logs\n";
        for (const auto& log : result.iterations) {
            output << "iteration=" << log.iteration << '\n';
            output << "attack_pool_size=" << log.attack_pool_size << '\n';
            output << "selection_objective_value="
                   << format_decimal(log.selection_objective_value) << '\n';
            output << "sentinel_size=" << log.sentinel_size << '\n';
            output << "selection_rr_samples=" << log.selection_rr_samples << '\n';
            output << "validation_rr_samples=" << log.validation_rr_samples << '\n';
            output << "validation_early_stopped_samples="
                   << log.validation_early_stopped_samples << '\n';
            output << "validation_objective_value="
                   << format_decimal(log.validation_objective_value) << '\n';
            output << "validation_upper_bound="
                   << format_decimal(log.validation_upper_bound) << '\n';
            output << "validation_status=" << log.validation_status << '\n';
            output << "sampling_mode=" << log.sampling_mode << '\n';
            output << "adaptive_rounds_used=" << log.adaptive_rounds_used << '\n';
            output << "target_quality=" << format_decimal(log.target_quality) << '\n';
            output << "required_rr_samples=" << log.required_rr_samples << '\n';
            output << "generated_rr_samples=" << log.generated_rr_samples << '\n';
            output << "verification_samples=" << log.verification_samples << '\n';
            output << "verification_bound=" << format_decimal(log.verification_bound) << '\n';
            output << "verification_accepted="
                   << (log.verification_accepted ? "true" : "false") << '\n';
            output << "rr_sample_cap_triggered="
                   << (log.rr_sample_cap_triggered ? "true" : "false") << '\n';
            output << "attack_adaptive_rounds_used=" << log.attack_adaptive_rounds_used << '\n';
            output << "attack_target_quality=" << format_decimal(log.attack_target_quality) << '\n';
            output << "attack_required_rr_samples=" << log.attack_required_rr_samples << '\n';
            output << "attack_generated_rr_samples=" << log.attack_generated_rr_samples << '\n';
            output << "attack_verification_samples=" << log.attack_verification_samples << '\n';
            output << "attack_validation_early_stopped_samples="
                   << log.attack_validation_early_stopped_samples << '\n';
            output << "attack_verification_lower_bound="
                   << format_decimal(log.attack_verification_lower_bound) << '\n';
            output << "attack_verification_accepted="
                   << (log.attack_verification_accepted ? "true" : "false") << '\n';
            output << "attack_rr_sample_cap_triggered="
                   << (log.attack_rr_sample_cap_triggered ? "true" : "false") << '\n';
            output << "attack_validation_status=" << log.attack_validation_status << '\n';
            output << "iteration_runtime_sec="
                   << format_decimal(log.iteration_runtime_sec) << '\n';
            output << "duplicate_attack=" << (log.duplicate_attack ? "true" : "false") << '\n';
            write_seed_lines(output, graph, "defense_seeds", log.defense_seeds);
            write_seed_lines(output, graph, "attack_seeds", log.attack_seeds);
            output << '\n';
        }
    }

    {
        auto output = open_output_file(paths.csv_path);
        output << "iteration,algorithm,selection_objective,uses_attack_pool,"
               << "uses_fj_objective_in_selection,uses_certificate,certificate_status,"
               << "selection_backend,selection_context,"
               << "lower_bound,upper_bound,certificate_gap,"
               << "attack_pool_size,selection_objective_value,sentinel_size,"
               << "selection_rr_samples,validation_rr_samples,validation_early_stopped_samples,"
               << "validation_objective_value,validation_upper_bound,validation_status,"
               << "sampling_mode,adaptive_rounds_used,target_quality,"
               << "required_rr_samples,generated_rr_samples,verification_samples,"
               << "verification_bound,verification_accepted,rr_sample_cap_triggered,"
               << "attack_adaptive_rounds_used,attack_target_quality,"
               << "attack_required_rr_samples,attack_generated_rr_samples,"
               << "attack_verification_samples,attack_validation_early_stopped_samples,attack_verification_lower_bound,"
               << "attack_verification_accepted,attack_rr_sample_cap_triggered,"
               << "attack_validation_status,"
               << "iteration_runtime_sec,"
               << "duplicate_attack,algorithm_runtime_sec,selection_mc_simulations,"
               << "fjc_linear_rr_samples,"
               << "mc_value_defense_vs_attack,mc_value_no_defense_attack_only,mc_value_defense_only,"
               << "mc_value_defense_vs_attack_variance,mc_value_defense_vs_attack_standard_error,"
               << "mc_value_defense_vs_attack_ci95_half_width,mc_value_defense_vs_attack_simulations,mc_value_defense_vs_attack_stop_status,"
               << "mc_value_no_defense_attack_only_variance,mc_value_no_defense_attack_only_standard_error,"
               << "mc_value_no_defense_attack_only_ci95_half_width,mc_value_no_defense_attack_only_simulations,mc_value_no_defense_attack_only_stop_status,"
               << "mc_value_defense_only_variance,mc_value_defense_only_standard_error,"
               << "mc_value_defense_only_ci95_half_width,mc_value_defense_only_simulations,mc_value_defense_only_stop_status,"
               << "defense_seeds_internal,defense_seeds_external,"
               << "attack_seeds_internal,attack_seeds_external,"
               << "attack_only_seeds_internal,attack_only_seeds_external";
        write_metadata_csv_header(output);
        output << '\n';

        for (const auto& log : result.iterations) {
            output << log.iteration << ','
                   << csv_escape(to_string(result.algorithm)) << ','
                   << csv_escape(result.selection_objective) << ','
                   << (result.uses_attack_pool ? "true" : "false") << ','
                   << (result.uses_fj_objective_in_selection ? "true" : "false") << ','
                   << (result.uses_certificate ? "true" : "false") << ','
                   << csv_escape(result.certificate_status) << ','
                   << csv_escape(result.selection_backend) << ','
                   << csv_escape(result.selection_context) << ','
                   << "not_applicable,not_applicable,not_applicable,"
                   << log.attack_pool_size << ','
                   << format_decimal(log.selection_objective_value) << ','
                   << log.sentinel_size << ','
                   << log.selection_rr_samples << ','
                   << log.validation_rr_samples << ','
                   << log.validation_early_stopped_samples << ','
                   << format_decimal(log.validation_objective_value) << ','
                   << format_decimal(log.validation_upper_bound) << ','
                   << csv_escape(log.validation_status) << ','
                   << csv_escape(log.sampling_mode) << ','
                   << log.adaptive_rounds_used << ','
                   << format_decimal(log.target_quality) << ','
                   << log.required_rr_samples << ','
                   << log.generated_rr_samples << ','
                   << log.verification_samples << ','
                   << format_decimal(log.verification_bound) << ','
                   << (log.verification_accepted ? "true" : "false") << ','
                   << (log.rr_sample_cap_triggered ? "true" : "false") << ','
                   << log.attack_adaptive_rounds_used << ','
                   << format_decimal(log.attack_target_quality) << ','
                   << log.attack_required_rr_samples << ','
                   << log.attack_generated_rr_samples << ','
                   << log.attack_verification_samples << ','
                   << log.attack_validation_early_stopped_samples << ','
                   << format_decimal(log.attack_verification_lower_bound) << ','
                   << (log.attack_verification_accepted ? "true" : "false") << ','
                   << (log.attack_rr_sample_cap_triggered ? "true" : "false") << ','
                   << csv_escape(log.attack_validation_status) << ','
                   << format_decimal(log.iteration_runtime_sec) << ','
                   << (log.duplicate_attack ? "true" : "false") << ','
                   << format_decimal(result.algorithm_runtime_sec) << ','
                   << result.selection_mc_simulations << ','
                   << result.fjc_linear_rr_samples << ','
                   << optional_mc_mean_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack) << ','
                   << optional_mc_mean_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only) << ','
                   << optional_mc_mean_to_string(
                          result.evaluation.available, result.evaluation.defense_only) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack.variance) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack.standard_error) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack.ci95_half_width) << ','
                   << optional_mc_count_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack.simulations) << ','
                   << csv_escape(optional_mc_status_to_string(
                          result.evaluation.available, result.evaluation.defense_vs_attack.stop_status)) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only.variance) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only.standard_error) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only.ci95_half_width) << ','
                   << optional_mc_count_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only.simulations) << ','
                   << csv_escape(optional_mc_status_to_string(
                          result.evaluation.available, result.evaluation.no_defense_attack_only.stop_status)) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_only.variance) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_only.standard_error) << ','
                   << optional_mc_value_to_string(
                          result.evaluation.available, result.evaluation.defense_only.ci95_half_width) << ','
                   << optional_mc_count_to_string(
                          result.evaluation.available, result.evaluation.defense_only.simulations) << ','
                   << csv_escape(optional_mc_status_to_string(
                          result.evaluation.available, result.evaluation.defense_only.stop_status)) << ','
                   << csv_escape(format_seed_set_preview(log.defense_seeds)) << ','
                   << csv_escape(external_seed_set(graph, log.defense_seeds)) << ','
                   << csv_escape(format_seed_set_preview(log.attack_seeds)) << ','
                   << csv_escape(external_seed_set(graph, log.attack_seeds)) << ','
                   << csv_escape(format_seed_set_preview(result.evaluation.attack_only_seeds)) << ','
                   << csv_escape(external_seed_set(graph, result.evaluation.attack_only_seeds));
            write_metadata_csv_values(output, metadata);
            output << '\n';
        }
    }

    transpose_csv_file(paths.csv_path, "baseline");

    {
        auto output = open_output_file(paths.seeds_path);
        write_seed_lines(output, graph, "final_defense_seeds", result.defense_seeds, true);
        write_seed_lines(output, graph, "final_attack_seeds", result.attack_seeds, true);
        if (result.evaluation.available) {
            write_seed_lines(
                output, graph, "attack_only_seeds", result.evaluation.attack_only_seeds, true);
        }
    }
}

}  // namespace rcd
