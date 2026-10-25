#include "rcd/propagation.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace rcd {
namespace {

constexpr std::int8_t kUnknown = 0;
constexpr std::int8_t kLive = 1;
constexpr std::int8_t kDead = -1;
constexpr std::int64_t kUnknownParent = -2;
constexpr std::int64_t kNoParent = -1;

void validate_node(NodeId node, std::size_t num_nodes) {
    if (static_cast<std::size_t>(node) >= num_nodes) {
        throw std::out_of_range("Node id is out of range");
    }
}

void validate_probability(Prob probability) {
    if (probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument("Edge probability must be in [0, 1]");
    }
}

}  // namespace

std::string to_string(PropagationModel model) {
    switch (model) {
        case PropagationModel::IC:
            return "ic";
        case PropagationModel::LT:
            return "lt";
    }
    return "unknown";
}

PropagationModel parse_propagation_model(const std::string& value) {
    if (value == "ic") {
        return PropagationModel::IC;
    }
    if (value == "lt") {
        return PropagationModel::LT;
    }
    throw std::invalid_argument("Propagation model must be either ic or lt");
}

LazyLiveEdgeOracle::LazyLiveEdgeOracle(std::size_t num_edges, std::uint64_t seed)
    : edge_state_(num_edges, kUnknown), rng_(seed), uniform_(0.0, 1.0) {
    touched_edges_.reserve(std::min<std::size_t>(num_edges, 1024));
}

bool LazyLiveEdgeOracle::is_live(EdgeIndex edge_id, Prob probability) {
    if (edge_id >= edge_state_.size()) {
        throw std::out_of_range("Edge id is out of range");
    }
    validate_probability(probability);

    auto& state = edge_state_[edge_id];
    if (state == kUnknown) {
        state = (uniform_(rng_) < probability) ? kLive : kDead;
        touched_edges_.push_back(edge_id);
    }
    return state == kLive;
}

void LazyLiveEdgeOracle::reset() {
    for (const auto edge_id : touched_edges_) {
        edge_state_[edge_id] = kUnknown;
    }
    touched_edges_.clear();
}

LazyTriggeringEdgeOracle::LazyTriggeringEdgeOracle(std::size_t num_nodes, std::uint64_t seed)
    : parent_state_(num_nodes, kUnknownParent), rng_(seed), uniform_(0.0, 1.0) {
    touched_nodes_.reserve(std::min<std::size_t>(num_nodes, 1024));
}

NodeId LazyTriggeringEdgeOracle::triggering_parent(const DirectedGraph& graph, NodeId node) {
    validate_node(node, graph.num_nodes());
    if (parent_state_.size() != graph.num_nodes()) {
        throw std::invalid_argument("Triggering oracle node count does not match graph size");
    }

    auto& state = parent_state_[node];
    if (state == kUnknownParent) {
        double cumulative = 0.0;
        const double draw = uniform_(rng_);
        state = kNoParent;
        for (const auto& edge : graph.in_adj[node]) {
            validate_probability(edge.probability);
            cumulative += edge.probability;
            if (cumulative > 1.0 + 1e-12) {
                throw std::invalid_argument("LT incoming edge probabilities must sum to at most 1");
            }
            if (draw < cumulative) {
                state = static_cast<std::int64_t>(edge.node);
                break;
            }
        }
        touched_nodes_.push_back(node);
    }

    if (state == kNoParent) {
        return static_cast<NodeId>(graph.num_nodes());
    }
    return static_cast<NodeId>(state);
}

bool LazyTriggeringEdgeOracle::has_triggering_parent(
    const DirectedGraph& graph,
    NodeId node,
    NodeId parent) {
    return triggering_parent(graph, node) == parent;
}

void LazyTriggeringEdgeOracle::reset() {
    for (const auto node : touched_nodes_) {
        parent_state_[node] = kUnknownParent;
    }
    touched_nodes_.clear();
}

TraversalWorkspace::TraversalWorkspace(std::size_t num_nodes)
    : visit_stamp_(num_nodes, 0) {
    nodes_.reserve(num_nodes);
}

void TraversalWorkspace::reset(std::size_t num_nodes) {
    if (visit_stamp_.size() != num_nodes) {
        visit_stamp_.assign(num_nodes, 0);
        current_stamp_ = 1;
        nodes_.clear();
        nodes_.reserve(num_nodes);
        return;
    }

    nodes_.clear();
    if (current_stamp_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(visit_stamp_.begin(), visit_stamp_.end(), 0);
        current_stamp_ = 1;
    } else {
        ++current_stamp_;
    }
}

bool TraversalWorkspace::mark(NodeId node) {
    validate_node(node, visit_stamp_.size());
    if (visit_stamp_[node] == current_stamp_) {
        return false;
    }
    visit_stamp_[node] = current_stamp_;
    nodes_.push_back(node);
    return true;
}

bool TraversalWorkspace::marked(NodeId node) const {
    validate_node(node, visit_stamp_.size());
    return visit_stamp_[node] == current_stamp_;
}

std::vector<NodeId> forward_reachable_ic(
    const DirectedGraph& graph,
    const std::vector<NodeId>& seeds,
    LazyLiveEdgeOracle& oracle,
    TraversalWorkspace& workspace) {
    workspace.reset(graph.num_nodes());

    for (const auto seed : seeds) {
        workspace.mark(seed);
    }

    std::size_t head = 0;
    auto& queue = workspace.nodes();
    while (head < queue.size()) {
        const NodeId current = queue[head++];
        for (const auto& edge : graph.out_adj[current]) {
            if (workspace.marked(edge.node)) {
                continue;
            }
            if (oracle.is_live(edge.edge_id, edge.probability)) {
                workspace.mark(edge.node);
            }
        }
    }

    return queue;
}

std::vector<NodeId> reverse_rr_set_ic(
    const DirectedGraph& graph,
    NodeId root,
    LazyLiveEdgeOracle& oracle,
    TraversalWorkspace& workspace) {
    validate_node(root, graph.num_nodes());
    workspace.reset(graph.num_nodes());
    workspace.mark(root);

    std::size_t head = 0;
    auto& queue = workspace.nodes();
    while (head < queue.size()) {
        const NodeId current = queue[head++];
        for (const auto& edge : graph.in_adj[current]) {
            if (workspace.marked(edge.node)) {
                continue;
            }
            if (oracle.is_live(edge.edge_id, edge.probability)) {
                workspace.mark(edge.node);
            }
        }
    }

    return queue;
}

std::vector<NodeId> forward_reachable_lt(
    const DirectedGraph& graph,
    const std::vector<NodeId>& seeds,
    LazyTriggeringEdgeOracle& oracle,
    TraversalWorkspace& workspace) {
    workspace.reset(graph.num_nodes());

    for (const auto seed : seeds) {
        workspace.mark(seed);
    }

    std::size_t head = 0;
    auto& queue = workspace.nodes();
    while (head < queue.size()) {
        const NodeId current = queue[head++];
        for (const auto& edge : graph.out_adj[current]) {
            if (workspace.marked(edge.node)) {
                continue;
            }
            if (oracle.has_triggering_parent(graph, edge.node, current)) {
                workspace.mark(edge.node);
            }
        }
    }

    return queue;
}

std::vector<NodeId> reverse_rr_set_lt(
    const DirectedGraph& graph,
    NodeId root,
    LazyTriggeringEdgeOracle& oracle,
    TraversalWorkspace& workspace) {
    validate_node(root, graph.num_nodes());
    workspace.reset(graph.num_nodes());
    workspace.mark(root);

    std::size_t head = 0;
    auto& queue = workspace.nodes();
    while (head < queue.size()) {
        const NodeId current = queue[head++];
        const NodeId parent = oracle.triggering_parent(graph, current);
        if (static_cast<std::size_t>(parent) < graph.num_nodes() && !workspace.marked(parent)) {
            workspace.mark(parent);
        }
    }

    return queue;
}

std::vector<NodeId> forward_reachable(
    PropagationModel model,
    const DirectedGraph& graph,
    const std::vector<NodeId>& seeds,
    LazyLiveEdgeOracle& live_oracle,
    LazyTriggeringEdgeOracle& triggering_oracle,
    TraversalWorkspace& workspace) {
    if (model == PropagationModel::LT) {
        return forward_reachable_lt(graph, seeds, triggering_oracle, workspace);
    }
    return forward_reachable_ic(graph, seeds, live_oracle, workspace);
}

std::vector<NodeId> reverse_rr_set(
    PropagationModel model,
    const DirectedGraph& graph,
    NodeId root,
    LazyLiveEdgeOracle& live_oracle,
    LazyTriggeringEdgeOracle& triggering_oracle,
    TraversalWorkspace& workspace) {
    if (model == PropagationModel::LT) {
        return reverse_rr_set_lt(graph, root, triggering_oracle, workspace);
    }
    return reverse_rr_set_ic(graph, root, live_oracle, workspace);
}

ReverseRREarlyStopResult reverse_rr_set_until_hit(
    PropagationModel model,
    const DirectedGraph& graph,
    NodeId root,
    const std::vector<std::uint8_t>& stop_mask,
    LazyLiveEdgeOracle& live_oracle,
    LazyTriggeringEdgeOracle& triggering_oracle,
    TraversalWorkspace& workspace) {
    validate_node(root, graph.num_nodes());
    if (stop_mask.size() != graph.num_nodes()) {
        throw std::invalid_argument("Early-stop mask size does not match graph size");
    }

    workspace.reset(graph.num_nodes());
    workspace.mark(root);
    if (stop_mask[root]) {
        return {workspace.nodes(), true};
    }

    std::size_t head = 0;
    auto& queue = workspace.nodes();
    while (head < queue.size()) {
        const NodeId current = queue[head++];
        if (model == PropagationModel::LT) {
            const NodeId parent = triggering_oracle.triggering_parent(graph, current);
            if (static_cast<std::size_t>(parent) < graph.num_nodes() &&
                !workspace.marked(parent)) {
                workspace.mark(parent);
                if (stop_mask[parent]) {
                    return {queue, true};
                }
            }
            continue;
        }

        for (const auto& edge : graph.in_adj[current]) {
            if (workspace.marked(edge.node)) {
                continue;
            }
            if (live_oracle.is_live(edge.edge_id, edge.probability)) {
                workspace.mark(edge.node);
                if (stop_mask[edge.node]) {
                    return {queue, true};
                }
            }
        }
    }

    return {queue, false};
}

}  // namespace rcd
