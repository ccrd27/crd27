#include "rcd/graph.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rcd {
namespace {

struct RawDirectedEdge {
    std::uint64_t from = 0;
    std::uint64_t to = 0;
};

struct RawWeightedEdge {
    std::uint64_t from = 0;
    std::uint64_t to = 0;
    double weight = 0.0;
};

std::string trim_comment_marker(std::string line) {
    if (!line.empty() && line.front() == '#') {
        line.erase(line.begin());
    }
    return line;
}

bool try_parse_pair(const std::string& line, std::uint64_t& first, std::uint64_t& second) {
    std::istringstream input(line);
    input >> first >> second;
    return !input.fail();
}

bool try_parse_weighted_edge(
    const std::string& line,
    std::uint64_t& first,
    std::uint64_t& second,
    double& weight) {
    std::istringstream input(line);
    input >> first >> second >> weight;
    return !input.fail();
}

bool try_parse_node_value(const std::string& line, std::uint64_t& node, double& value) {
    std::istringstream input(line);
    input >> node >> value;
    return !input.fail();
}

NodeId checked_node_id(std::size_t value) {
    if (value > static_cast<std::size_t>(std::numeric_limits<NodeId>::max())) {
        throw std::runtime_error("The graph has too many nodes for NodeId");
    }
    return static_cast<NodeId>(value);
}

void add_external_id(DirectedGraph& graph, std::uint64_t external_id) {
    if (graph.internal_id_of.find(external_id) != graph.internal_id_of.end()) {
        return;
    }
    const auto internal_id = checked_node_id(graph.external_ids.size());
    graph.internal_id_of.emplace(external_id, internal_id);
    graph.external_ids.push_back(external_id);
}

std::uint64_t undirected_key(NodeId a, NodeId b, std::size_t n) {
    if (a > b) {
        std::swap(a, b);
    }
    return static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(n) + b;
}

std::uint64_t directed_key(NodeId from, NodeId to, std::size_t n) {
    return static_cast<std::uint64_t>(from) * static_cast<std::uint64_t>(n) + to;
}

void validate_finite_range(double value, double min_value, double max_value, const std::string& name) {
    if (!std::isfinite(value) || value < min_value || value > max_value) {
        throw std::invalid_argument(name + " must be finite and in the configured range");
    }
}

std::vector<RawWeightedEdge> read_weighted_edges(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open weighted edge file: " + path);
    }

    std::vector<RawWeightedEdge> edges;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::uint64_t from = 0;
        std::uint64_t to = 0;
        double weight = 0.0;
        if (!try_parse_weighted_edge(line, from, to, weight)) {
            continue;
        }
        edges.push_back(RawWeightedEdge{from, to, weight});
    }
    return edges;
}

}  // namespace

DirectedGraph read_directed_edge_list(const std::string& path, const EdgeListReadOptions& options) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open dataset: " + path);
    }

    std::vector<RawDirectedEdge> raw_edges;
    std::size_t header_node_count = 0;
    bool has_header_node_count = false;
    std::size_t skipped_self_loops = 0;

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        const bool is_comment = !line.empty() && line.front() == '#';
        std::uint64_t first = 0;
        std::uint64_t second = 0;
        const std::string parse_line = is_comment ? trim_comment_marker(line) : line;

        if (!try_parse_pair(parse_line, first, second)) {
            continue;
        }

        if (is_comment) {
            if (options.use_header_node_count && !has_header_node_count) {
                header_node_count = static_cast<std::size_t>(first);
                has_header_node_count = true;
            }
            continue;
        }

        if (options.remove_self_loops && first == second) {
            ++skipped_self_loops;
            continue;
        }

        raw_edges.push_back(RawDirectedEdge{first, second});
    }

    std::sort(raw_edges.begin(), raw_edges.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.from != rhs.from) {
            return lhs.from < rhs.from;
        }
        return lhs.to < rhs.to;
    });

    std::size_t skipped_duplicates = 0;
    if (options.remove_duplicate_directed_edges) {
        auto new_end = std::unique(raw_edges.begin(), raw_edges.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.from == rhs.from && lhs.to == rhs.to;
        });
        skipped_duplicates = static_cast<std::size_t>(std::distance(new_end, raw_edges.end()));
        raw_edges.erase(new_end, raw_edges.end());
    }

    DirectedGraph graph;
    graph.skipped_self_loops = skipped_self_loops;
    graph.skipped_duplicate_edges = skipped_duplicates;

    if (options.use_header_node_count && has_header_node_count) {
        for (std::size_t id = 0; id < header_node_count; ++id) {
            add_external_id(graph, static_cast<std::uint64_t>(id));
        }
    }

    for (const auto& edge : raw_edges) {
        add_external_id(graph, edge.from);
        add_external_id(graph, edge.to);
    }

    const std::size_t n = graph.num_nodes();
    graph.out_adj.assign(n, {});
    graph.in_adj.assign(n, {});
    graph.edges.reserve(raw_edges.size());

    std::vector<std::size_t> in_degree(n, 0);
    for (const auto& edge : raw_edges) {
        const NodeId to = graph.internal_id_of.at(edge.to);
        ++in_degree[to];
    }

    for (const auto& raw : raw_edges) {
        const NodeId from = graph.internal_id_of.at(raw.from);
        const NodeId to = graph.internal_id_of.at(raw.to);
        const EdgeIndex edge_id = graph.edges.size();
        const Prob probability =
            options.weighted_cascade_probabilities ? 1.0 / static_cast<Prob>(in_degree[to]) : 0.0;

        graph.edges.push_back(DirectedEdge{from, to, edge_id, probability});
        graph.out_adj[from].push_back(AdjacentEdge{to, edge_id, probability});
        graph.in_adj[to].push_back(AdjacentEdge{from, edge_id, probability});
    }

    return graph;
}

void apply_edge_probability_file(DirectedGraph& graph, const std::string& path) {
    std::unordered_map<std::uint64_t, EdgeIndex> edge_by_key;
    edge_by_key.reserve(graph.num_edges());
    for (const auto& edge : graph.edges) {
        edge_by_key.emplace(directed_key(edge.from, edge.to, graph.num_nodes()), edge.id);
    }

    std::vector<std::uint8_t> seen(graph.num_edges(), 0);
    for (const auto& raw : read_weighted_edges(path)) {
        const auto from_it = graph.internal_id_of.find(raw.from);
        const auto to_it = graph.internal_id_of.find(raw.to);
        if (from_it == graph.internal_id_of.end() || to_it == graph.internal_id_of.end()) {
            throw std::invalid_argument("Edge probability file contains an unknown node id");
        }
        validate_finite_range(raw.weight, 0.0, 1.0, "Edge probability");
        const auto key = directed_key(from_it->second, to_it->second, graph.num_nodes());
        const auto edge_it = edge_by_key.find(key);
        if (edge_it == edge_by_key.end()) {
            throw std::invalid_argument("Edge probability file contains an edge not present in the graph");
        }
        const EdgeIndex edge_id = edge_it->second;
        if (seen[edge_id] != 0) {
            throw std::invalid_argument("Edge probability file contains a duplicate directed edge");
        }
        seen[edge_id] = 1;
        graph.edges[edge_id].probability = raw.weight;
    }

    for (std::size_t i = 0; i < seen.size(); ++i) {
        if (seen[i] == 0) {
            throw std::invalid_argument("Edge probability file is missing at least one graph edge");
        }
    }

    for (auto& adj : graph.out_adj) {
        for (auto& edge : adj) {
            edge.probability = graph.edges[edge.edge_id].probability;
        }
    }
    for (auto& adj : graph.in_adj) {
        for (auto& edge : adj) {
            edge.probability = graph.edges[edge.edge_id].probability;
        }
    }
}

Eigen::VectorXd read_node_value_file(
    const DirectedGraph& graph,
    const std::string& path,
    const std::string& value_name,
    double min_value,
    double max_value) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open " + value_name + " file: " + path);
    }

    Eigen::VectorXd values(graph.num_nodes());
    std::vector<std::uint8_t> seen(graph.num_nodes(), 0);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::uint64_t external_id = 0;
        double value = 0.0;
        if (!try_parse_node_value(line, external_id, value)) {
            continue;
        }
        const auto it = graph.internal_id_of.find(external_id);
        if (it == graph.internal_id_of.end()) {
            throw std::invalid_argument(value_name + " file contains an unknown node id");
        }
        validate_finite_range(value, min_value, max_value, value_name);
        const auto node = static_cast<std::size_t>(it->second);
        if (seen[node] != 0) {
            throw std::invalid_argument(value_name + " file contains a duplicate node id");
        }
        seen[node] = 1;
        values[static_cast<Eigen::Index>(node)] = value;
    }

    for (std::size_t i = 0; i < seen.size(); ++i) {
        if (seen[i] == 0) {
            throw std::invalid_argument(value_name + " file is missing at least one graph node");
        }
    }
    return values;
}

UndirectedFJGraph build_fj_graph(const DirectedGraph& graph) {
    UndirectedFJGraph fj;
    fj.num_nodes = graph.num_nodes();
    fj.adj.assign(fj.num_nodes, {});

    std::vector<std::uint64_t> keys;
    keys.reserve(graph.num_edges());

    for (const auto& edge : graph.edges) {
        if (edge.from == edge.to) {
            continue;
        }
        keys.push_back(undirected_key(edge.from, edge.to, graph.num_nodes()));
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    fj.edges.reserve(keys.size());
    for (const auto key : keys) {
        const NodeId u = static_cast<NodeId>(key / graph.num_nodes());
        const NodeId v = static_cast<NodeId>(key % graph.num_nodes());
        fj.edges.push_back(UndirectedFJGraph::Edge{u, v, 1.0});
        fj.adj[u].push_back(v);
        fj.adj[v].push_back(u);
    }

    return fj;
}

void apply_fj_weight_file(UndirectedFJGraph& fj, const DirectedGraph& graph, const std::string& path) {
    std::unordered_map<std::uint64_t, std::size_t> edge_by_key;
    edge_by_key.reserve(fj.num_edges());
    for (std::size_t i = 0; i < fj.edges.size(); ++i) {
        edge_by_key.emplace(undirected_key(fj.edges[i].u, fj.edges[i].v, fj.num_nodes), i);
    }

    std::vector<std::uint8_t> seen(fj.num_edges(), 0);
    for (const auto& raw : read_weighted_edges(path)) {
        const auto from_it = graph.internal_id_of.find(raw.from);
        const auto to_it = graph.internal_id_of.find(raw.to);
        if (from_it == graph.internal_id_of.end() || to_it == graph.internal_id_of.end()) {
            throw std::invalid_argument("FJ weight file contains an unknown node id");
        }
        if (from_it->second == to_it->second) {
            throw std::invalid_argument("FJ weight file must not contain self loops");
        }
        validate_finite_range(raw.weight, 0.0, std::numeric_limits<double>::max(), "FJ edge weight");
        const auto key = undirected_key(from_it->second, to_it->second, fj.num_nodes);
        const auto edge_it = edge_by_key.find(key);
        if (edge_it == edge_by_key.end()) {
            throw std::invalid_argument("FJ weight file contains an edge not present in the FJ graph");
        }
        if (seen[edge_it->second] != 0) {
            throw std::invalid_argument("FJ weight file contains a duplicate undirected edge");
        }
        seen[edge_it->second] = 1;
        fj.edges[edge_it->second].weight = raw.weight;
    }

    for (std::size_t i = 0; i < seen.size(); ++i) {
        if (seen[i] == 0) {
            throw std::invalid_argument("FJ weight file is missing at least one FJ edge");
        }
    }
}

UndirectedFJGraph build_unit_weight_fj_graph(const DirectedGraph& graph) {
    return build_fj_graph(graph);
}

}  // namespace rcd
