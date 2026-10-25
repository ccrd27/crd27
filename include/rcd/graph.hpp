#pragma once

#include "rcd/types.hpp"

#include <Eigen/Dense>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rcd {

struct DirectedEdge {
    NodeId from = 0;
    NodeId to = 0;
    EdgeIndex id = 0;
    Prob probability = 0.0;
};

struct AdjacentEdge {
    NodeId node = 0;
    EdgeIndex edge_id = 0;
    Prob probability = 0.0;
};

struct DirectedGraph {
    std::vector<std::uint64_t> external_ids;
    std::unordered_map<std::uint64_t, NodeId> internal_id_of;
    std::vector<DirectedEdge> edges;
    std::vector<std::vector<AdjacentEdge>> out_adj;
    std::vector<std::vector<AdjacentEdge>> in_adj;
    std::size_t skipped_self_loops = 0;
    std::size_t skipped_duplicate_edges = 0;

    std::size_t num_nodes() const noexcept { return external_ids.size(); }
    std::size_t num_edges() const noexcept { return edges.size(); }
};

struct UndirectedFJGraph {
    struct Edge {
        NodeId u = 0;
        NodeId v = 0;
        double weight = 1.0;
    };

    std::size_t num_nodes = 0;
    std::vector<Edge> edges;
    std::vector<std::vector<NodeId>> adj;

    std::size_t num_edges() const noexcept { return edges.size(); }
};

struct EdgeListReadOptions {
    bool use_header_node_count = true;
    bool weighted_cascade_probabilities = true;
    bool remove_self_loops = true;
    bool remove_duplicate_directed_edges = true;
};

DirectedGraph read_directed_edge_list(
    const std::string& path,
    const EdgeListReadOptions& options = EdgeListReadOptions{});

void apply_edge_probability_file(DirectedGraph& graph, const std::string& path);

Eigen::VectorXd read_node_value_file(
    const DirectedGraph& graph,
    const std::string& path,
    const std::string& value_name,
    double min_value,
    double max_value);

UndirectedFJGraph build_fj_graph(const DirectedGraph& graph);
void apply_fj_weight_file(UndirectedFJGraph& fj, const DirectedGraph& graph, const std::string& path);

UndirectedFJGraph build_unit_weight_fj_graph(const DirectedGraph& graph);

}  // namespace rcd
