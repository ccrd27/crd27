#pragma once

#include "rcd/graph.hpp"
#include "rcd/types.hpp"

#include <Eigen/Dense>

#include <memory>
#include <string>

namespace rcd {

enum class ConflictMetric {
    Polarization,
    Disagreement
};

ConflictMetric parse_conflict_metric(const std::string& metric);
std::string to_string(ConflictMetric metric);

class ConflictBackend {
public:
    virtual ~ConflictBackend() = default;

    virtual std::size_t num_nodes() const noexcept = 0;
    virtual ConflictMetric metric() const noexcept = 0;
    virtual const char* backend_name() const noexcept = 0;
    virtual bool is_approximate() const noexcept = 0;
    virtual double beta_bound() const = 0;
    virtual double value(const Eigen::VectorXd& x) const = 0;
    virtual double dot_row(NodeId node, const Eigen::VectorXd& x) const = 0;
    virtual Eigen::VectorXd apply(const Eigen::VectorXd& x) const = 0;
    virtual Eigen::VectorXd expressed_opinion(const Eigen::VectorXd& x) const = 0;
};

class DenseConflict final : public ConflictBackend {
public:
    DenseConflict(Eigen::MatrixXd laplacian, Eigen::MatrixXd q, Eigen::MatrixXd conflict);

    static DenseConflict build(const UndirectedFJGraph& graph, ConflictMetric metric);

    std::size_t num_nodes() const noexcept override;
    ConflictMetric metric() const noexcept override { return metric_; }
    const char* backend_name() const noexcept override { return "dense"; }
    bool is_approximate() const noexcept override { return false; }
    double beta_bound() const override;
    double value(const Eigen::VectorXd& x) const override;
    double dot_row(NodeId node, const Eigen::VectorXd& x) const override;
    Eigen::VectorXd apply(const Eigen::VectorXd& x) const override;
    Eigen::VectorXd expressed_opinion(const Eigen::VectorXd& x) const override;

    const Eigen::MatrixXd& laplacian() const noexcept { return laplacian_; }
    const Eigen::MatrixXd& q_matrix() const noexcept { return q_; }
    const Eigen::MatrixXd& conflict_matrix() const noexcept { return conflict_; }

private:
    ConflictMetric metric_ = ConflictMetric::Polarization;
    Eigen::MatrixXd laplacian_;
    Eigen::MatrixXd q_;
    Eigen::MatrixXd conflict_;
};

class AmgclConflict final : public ConflictBackend {
public:
    struct Options {
        double tolerance = 1e-8;
        int max_iterations = 200;
    };

    AmgclConflict(AmgclConflict&&) noexcept;
    AmgclConflict& operator=(AmgclConflict&&) noexcept;
    AmgclConflict(const AmgclConflict&) = delete;
    AmgclConflict& operator=(const AmgclConflict&) = delete;
    ~AmgclConflict();

    static AmgclConflict build(
        const UndirectedFJGraph& graph,
        ConflictMetric metric);
    static AmgclConflict build(
        const UndirectedFJGraph& graph,
        ConflictMetric metric,
        const Options& options);

    std::size_t num_nodes() const noexcept override;
    ConflictMetric metric() const noexcept override;
    const char* backend_name() const noexcept override { return "amgcl"; }
    bool is_approximate() const noexcept override { return true; }
    double beta_bound() const override;
    double tolerance() const noexcept;
    int max_iterations() const noexcept;
    double last_relative_error() const noexcept;
    int last_iterations() const noexcept;

    double value(const Eigen::VectorXd& x) const override;
    double dot_row(NodeId node, const Eigen::VectorXd& x) const override;
    Eigen::VectorXd apply(const Eigen::VectorXd& x) const override;
    Eigen::VectorXd expressed_opinion(const Eigen::VectorXd& x) const override;

private:
    struct Impl;
    explicit AmgclConflict(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

Eigen::MatrixXd build_unit_weight_laplacian(const UndirectedFJGraph& graph);
Eigen::MatrixXd invert_i_plus_l(const Eigen::MatrixXd& laplacian);
Eigen::MatrixXd build_polarization_matrix(const Eigen::MatrixXd& q);
Eigen::MatrixXd build_disagreement_matrix(const Eigen::MatrixXd& q, const Eigen::MatrixXd& laplacian);

}  // namespace rcd
