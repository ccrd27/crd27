#include "rcd/conflict.hpp"

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace rcd {
namespace {

using AmgclBackend = amgcl::backend::builtin<double>;
using AmgclSolver = amgcl::make_solver<
    amgcl::amg<
        AmgclBackend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0>,
    amgcl::solver::cg<AmgclBackend>>;

struct SparseCrs {
    ptrdiff_t rows = 0;
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double> val;
};

void validate_square(const Eigen::MatrixXd& matrix, const char* name) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument(std::string(name) + " must be square");
    }
}

void validate_vector_size(const Eigen::VectorXd& x, Eigen::Index expected) {
    if (x.size() != expected) {
        throw std::invalid_argument("Vector size does not match conflict matrix size");
    }
}

SparseCrs build_laplacian_crs(const UndirectedFJGraph& graph, bool add_identity) {
    const auto n = static_cast<ptrdiff_t>(graph.num_nodes);
    std::vector<std::vector<std::pair<ptrdiff_t, double>>> rows(static_cast<std::size_t>(n));
    std::vector<double> degree(static_cast<std::size_t>(n), 0.0);

    for (const auto& edge : graph.edges) {
        const auto u_node = edge.u;
        const auto v_node = edge.v;
        const double weight = edge.weight;
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument("FJ edge weight must be finite and nonnegative");
        }
        const auto u = static_cast<ptrdiff_t>(u_node);
        const auto v = static_cast<ptrdiff_t>(v_node);
        if (u < 0 || v < 0 || u >= n || v >= n) {
            throw std::out_of_range("FJ edge endpoint is out of range");
        }
        degree[static_cast<std::size_t>(u)] += weight;
        degree[static_cast<std::size_t>(v)] += weight;
        rows[static_cast<std::size_t>(u)].push_back({v, -weight});
        rows[static_cast<std::size_t>(v)].push_back({u, -weight});
    }

    for (ptrdiff_t i = 0; i < n; ++i) {
        double diagonal = degree[static_cast<std::size_t>(i)];
        if (add_identity) {
            diagonal += 1.0;
        }
        rows[static_cast<std::size_t>(i)].push_back({i, diagonal});
    }

    SparseCrs crs;
    crs.rows = n;
    crs.ptr.reserve(static_cast<std::size_t>(n) + 1);
    crs.ptr.push_back(0);

    for (auto& row : rows) {
        std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

        bool has_entry = false;
        ptrdiff_t last_col = -1;
        double accumulated = 0.0;
        for (const auto& [column, value] : row) {
            if (!has_entry || column != last_col) {
                if (has_entry && accumulated != 0.0) {
                    crs.col.push_back(last_col);
                    crs.val.push_back(accumulated);
                }
                has_entry = true;
                last_col = column;
                accumulated = value;
            } else {
                accumulated += value;
            }
        }
        if (has_entry && accumulated != 0.0) {
            crs.col.push_back(last_col);
            crs.val.push_back(accumulated);
        }
        crs.ptr.push_back(static_cast<ptrdiff_t>(crs.col.size()));
    }

    return crs;
}

Eigen::VectorXd crs_multiply(const SparseCrs& matrix, const Eigen::VectorXd& x) {
    validate_vector_size(x, matrix.rows);
    Eigen::VectorXd y = Eigen::VectorXd::Zero(matrix.rows);
    for (ptrdiff_t row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (ptrdiff_t idx = matrix.ptr[static_cast<std::size_t>(row)];
             idx < matrix.ptr[static_cast<std::size_t>(row + 1)];
             ++idx) {
            sum += matrix.val[static_cast<std::size_t>(idx)] *
                   x[static_cast<Eigen::Index>(matrix.col[static_cast<std::size_t>(idx)])];
        }
        y[static_cast<Eigen::Index>(row)] = sum;
    }
    return y;
}

}  // namespace

ConflictMetric parse_conflict_metric(const std::string& metric) {
    if (metric == "polarization") {
        return ConflictMetric::Polarization;
    }
    if (metric == "disagreement") {
        return ConflictMetric::Disagreement;
    }
    throw std::invalid_argument("Unknown conflict metric: " + metric);
}

std::string to_string(ConflictMetric metric) {
    switch (metric) {
        case ConflictMetric::Polarization:
            return "polarization";
        case ConflictMetric::Disagreement:
            return "disagreement";
    }
    return "unknown";
}

DenseConflict::DenseConflict(Eigen::MatrixXd laplacian, Eigen::MatrixXd q, Eigen::MatrixXd conflict)
    : laplacian_(std::move(laplacian)), q_(std::move(q)), conflict_(std::move(conflict)) {
    validate_square(laplacian_, "Laplacian");
    validate_square(q_, "Q");
    validate_square(conflict_, "Conflict matrix");
    if (laplacian_.rows() != q_.rows() || q_.rows() != conflict_.rows()) {
        throw std::invalid_argument("DenseConflict matrices must have the same dimension");
    }
}

DenseConflict DenseConflict::build(const UndirectedFJGraph& graph, ConflictMetric metric) {
    Eigen::MatrixXd laplacian = build_unit_weight_laplacian(graph);
    Eigen::MatrixXd q = invert_i_plus_l(laplacian);
    Eigen::MatrixXd conflict;
    switch (metric) {
        case ConflictMetric::Polarization:
            conflict = build_polarization_matrix(q);
            break;
        case ConflictMetric::Disagreement:
            conflict = build_disagreement_matrix(q, laplacian);
            break;
    }
    DenseConflict result(std::move(laplacian), std::move(q), std::move(conflict));
    result.metric_ = metric;
    return result;
}

std::size_t DenseConflict::num_nodes() const noexcept {
    return static_cast<std::size_t>(conflict_.rows());
}

double DenseConflict::beta_bound() const {
    double max_row_l1 = 0.0;
    for (Eigen::Index row = 0; row < conflict_.rows(); ++row) {
        max_row_l1 = std::max(max_row_l1, conflict_.row(row).cwiseAbs().sum());
    }
    return 0.5 * max_row_l1;
}

double DenseConflict::value(const Eigen::VectorXd& x) const {
    validate_vector_size(x, conflict_.rows());
    return x.dot(conflict_ * x);
}

double DenseConflict::dot_row(NodeId node, const Eigen::VectorXd& x) const {
    validate_vector_size(x, conflict_.rows());
    if (static_cast<Eigen::Index>(node) >= conflict_.rows()) {
        throw std::out_of_range("Node id is out of range");
    }
    return conflict_.row(static_cast<Eigen::Index>(node)).dot(x);
}

Eigen::VectorXd DenseConflict::apply(const Eigen::VectorXd& x) const {
    validate_vector_size(x, conflict_.rows());
    return conflict_ * x;
}

Eigen::VectorXd DenseConflict::expressed_opinion(const Eigen::VectorXd& x) const {
    validate_vector_size(x, q_.rows());
    return q_ * x;
}

struct AmgclConflict::Impl {
    Impl(
        std::size_t node_count,
        ConflictMetric conflict_metric,
        Options solver_options,
        SparseCrs system_matrix,
        SparseCrs laplacian_matrix)
        : num_nodes(node_count),
          metric(conflict_metric),
          options(solver_options),
          system(std::move(system_matrix)),
          laplacian(std::move(laplacian_matrix)) {
        AmgclSolver::params params;
        params.solver.tol = options.tolerance;
        params.solver.maxiter = options.max_iterations;
        solver = std::make_unique<AmgclSolver>(
            std::tie(system.rows, system.ptr, system.col, system.val), params);
    }

    Eigen::VectorXd solve_q(const Eigen::VectorXd& x) const {
        validate_vector_size(x, static_cast<Eigen::Index>(num_nodes));
        std::vector<double> rhs(static_cast<std::size_t>(x.size()));
        for (Eigen::Index i = 0; i < x.size(); ++i) {
            rhs[static_cast<std::size_t>(i)] = x[i];
        }

        std::vector<double> solution(rhs.size(), 0.0);
        const auto result = (*solver)(
            std::tie(system.rows, system.ptr, system.col, system.val), rhs, solution);
        last_iterations = static_cast<int>(std::get<0>(result));
        last_relative_error = std::get<1>(result);

        Eigen::VectorXd y(x.size());
        for (Eigen::Index i = 0; i < x.size(); ++i) {
            y[i] = solution[static_cast<std::size_t>(i)];
        }
        return y;
    }

    std::size_t num_nodes = 0;
    ConflictMetric metric = ConflictMetric::Polarization;
    Options options;
    SparseCrs system;
    SparseCrs laplacian;
    std::unique_ptr<AmgclSolver> solver;
    mutable int last_iterations = 0;
    mutable double last_relative_error = 0.0;
};

AmgclConflict::AmgclConflict(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

AmgclConflict::AmgclConflict(AmgclConflict&&) noexcept = default;
AmgclConflict& AmgclConflict::operator=(AmgclConflict&&) noexcept = default;
AmgclConflict::~AmgclConflict() = default;

AmgclConflict AmgclConflict::build(
    const UndirectedFJGraph& graph,
    ConflictMetric metric) {
    return build(graph, metric, Options{});
}

AmgclConflict AmgclConflict::build(
    const UndirectedFJGraph& graph,
    ConflictMetric metric,
    const Options& options) {
    if (options.tolerance <= 0.0) {
        throw std::invalid_argument("AMGCL tolerance must be positive");
    }
    if (options.max_iterations <= 0) {
        throw std::invalid_argument("AMGCL max iterations must be positive");
    }
    SparseCrs system = build_laplacian_crs(graph, true);
    SparseCrs laplacian = build_laplacian_crs(graph, false);
    return AmgclConflict(std::make_unique<Impl>(
        graph.num_nodes, metric, options, std::move(system), std::move(laplacian)));
}

std::size_t AmgclConflict::num_nodes() const noexcept {
    return impl_->num_nodes;
}

ConflictMetric AmgclConflict::metric() const noexcept {
    return impl_->metric;
}

double AmgclConflict::tolerance() const noexcept {
    return impl_->options.tolerance;
}

int AmgclConflict::max_iterations() const noexcept {
    return impl_->options.max_iterations;
}

double AmgclConflict::last_relative_error() const noexcept {
    return impl_->last_relative_error;
}

int AmgclConflict::last_iterations() const noexcept {
    return impl_->last_iterations;
}

double AmgclConflict::beta_bound() const {
    return 0.5 * std::sqrt(static_cast<double>(impl_->num_nodes));
}

double AmgclConflict::value(const Eigen::VectorXd& x) const {
    validate_vector_size(x, static_cast<Eigen::Index>(impl_->num_nodes));
    const Eigen::VectorXd qx = expressed_opinion(x);
    // Evaluate the quadratic form in expressed opinions to avoid a second solve.
    switch (impl_->metric) {
        case ConflictMetric::Polarization: {
            if (qx.size() == 0) {
                return 0.0;
            }
            return (qx.array() - qx.mean()).matrix().squaredNorm();
        }
        case ConflictMetric::Disagreement:
            return qx.dot(crs_multiply(impl_->laplacian, qx));
    }
    throw std::logic_error("Unknown conflict metric");
}

double AmgclConflict::dot_row(NodeId node, const Eigen::VectorXd& x) const {
    if (node >= impl_->num_nodes) {
        throw std::out_of_range("Node id is out of range");
    }
    return apply(x)[static_cast<Eigen::Index>(node)];
}

Eigen::VectorXd AmgclConflict::apply(const Eigen::VectorXd& x) const {
    validate_vector_size(x, static_cast<Eigen::Index>(impl_->num_nodes));
    const Eigen::VectorXd qx = expressed_opinion(x);

    switch (impl_->metric) {
        case ConflictMetric::Polarization: {
            Eigen::VectorXd centered = qx;
            centered.array() -= qx.mean();
            return impl_->solve_q(centered);
        }
        case ConflictMetric::Disagreement: {
            const Eigen::VectorXd lqx = crs_multiply(impl_->laplacian, qx);
            return impl_->solve_q(lqx);
        }
    }
    throw std::logic_error("Unknown conflict metric");
}

Eigen::VectorXd AmgclConflict::expressed_opinion(const Eigen::VectorXd& x) const {
    validate_vector_size(x, static_cast<Eigen::Index>(impl_->num_nodes));
    return impl_->solve_q(x);
}

Eigen::MatrixXd build_unit_weight_laplacian(const UndirectedFJGraph& graph) {
    Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(graph.num_nodes),
        static_cast<Eigen::Index>(graph.num_nodes));

    for (const auto& edge : graph.edges) {
        if (!std::isfinite(edge.weight) || edge.weight < 0.0) {
            throw std::invalid_argument("FJ edge weight must be finite and nonnegative");
        }
        const Eigen::Index ui = static_cast<Eigen::Index>(edge.u);
        const Eigen::Index vi = static_cast<Eigen::Index>(edge.v);
        laplacian(ui, ui) += edge.weight;
        laplacian(vi, vi) += edge.weight;
        laplacian(ui, vi) -= edge.weight;
        laplacian(vi, ui) -= edge.weight;
    }

    return laplacian;
}

Eigen::MatrixXd invert_i_plus_l(const Eigen::MatrixXd& laplacian) {
    validate_square(laplacian, "Laplacian");
    const Eigen::Index n = laplacian.rows();
    const Eigen::MatrixXd system = Eigen::MatrixXd::Identity(n, n) + laplacian;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(system);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize I + L");
    }
    return ldlt.solve(Eigen::MatrixXd::Identity(n, n));
}

Eigen::MatrixXd build_polarization_matrix(const Eigen::MatrixXd& q) {
    validate_square(q, "Q");
    const Eigen::Index n = q.rows();
    const Eigen::MatrixXd centered =
        Eigen::MatrixXd::Identity(n, n) -
        Eigen::MatrixXd::Constant(n, n, 1.0 / static_cast<double>(n));
    Eigen::MatrixXd matrix = q * centered * q;
    matrix = 0.5 * (matrix + matrix.transpose());
    return matrix;
}

Eigen::MatrixXd build_disagreement_matrix(const Eigen::MatrixXd& q, const Eigen::MatrixXd& laplacian) {
    validate_square(q, "Q");
    validate_square(laplacian, "Laplacian");
    if (q.rows() != laplacian.rows()) {
        throw std::invalid_argument("Q and Laplacian dimensions do not match");
    }
    Eigen::MatrixXd matrix = q * laplacian * q;
    matrix = 0.5 * (matrix + matrix.transpose());
    return matrix;
}

}  // namespace rcd
