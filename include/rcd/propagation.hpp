#pragma once

#include "rcd/graph.hpp"
#include "rcd/types.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace rcd {

enum class PropagationModel {
    IC,
    LT
};

std::string to_string(PropagationModel model);
PropagationModel parse_propagation_model(const std::string& value);

class LazyLiveEdgeOracle {
public:
    explicit LazyLiveEdgeOracle(std::size_t num_edges, std::uint64_t seed);

    bool is_live(EdgeIndex edge_id, Prob probability);
    void reset();

    std::size_t num_edges() const noexcept { return edge_state_.size(); }
    std::size_t touched_edges() const noexcept { return touched_edges_.size(); }

private:
    std::vector<std::int8_t> edge_state_;
    std::vector<EdgeIndex> touched_edges_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_;
};

class LazyTriggeringEdgeOracle {
public:
    explicit LazyTriggeringEdgeOracle(std::size_t num_nodes, std::uint64_t seed);

    NodeId triggering_parent(const DirectedGraph& graph, NodeId node);
    bool has_triggering_parent(const DirectedGraph& graph, NodeId node, NodeId parent);
    void reset();

    std::size_t num_nodes() const noexcept { return parent_state_.size(); }
    std::size_t touched_nodes() const noexcept { return touched_nodes_.size(); }

private:
    std::vector<std::int64_t> parent_state_;
    std::vector<NodeId> touched_nodes_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_;
};

class TraversalWorkspace {
public:
    explicit TraversalWorkspace(std::size_t num_nodes = 0);

    void reset(std::size_t num_nodes);
    bool mark(NodeId node);
    bool marked(NodeId node) const;

    std::vector<NodeId>& nodes() noexcept { return nodes_; }
    const std::vector<NodeId>& nodes() const noexcept { return nodes_; }

private:
    std::vector<std::uint32_t> visit_stamp_;
    std::uint32_t current_stamp_ = 1;
    std::vector<NodeId> nodes_;
};

struct ReverseRREarlyStopResult {
    std::vector<NodeId> rr_prefix;
    bool stopped_early = false;
};

std::vector<NodeId> forward_reachable_ic(
    const DirectedGraph& graph,
    const std::vector<NodeId>& seeds,
    LazyLiveEdgeOracle& oracle,
    TraversalWorkspace& workspace);

std::vector<NodeId> reverse_rr_set_ic(
    const DirectedGraph& graph,
    NodeId root,
    LazyLiveEdgeOracle& oracle,
    TraversalWorkspace& workspace);

std::vector<NodeId> forward_reachable_lt(
    const DirectedGraph& graph,
    const std::vector<NodeId>& seeds,
    LazyTriggeringEdgeOracle& oracle,
    TraversalWorkspace& workspace);

std::vector<NodeId> reverse_rr_set_lt(
    const DirectedGraph& graph,
    NodeId root,
    LazyTriggeringEdgeOracle& oracle,
    TraversalWorkspace& workspace);

std::vector<NodeId> forward_reachable(
    PropagationModel model,
    const DirectedGraph& graph,
    const std::vector<NodeId>& seeds,
    LazyLiveEdgeOracle& live_oracle,
    LazyTriggeringEdgeOracle& triggering_oracle,
    TraversalWorkspace& workspace);

std::vector<NodeId> reverse_rr_set(
    PropagationModel model,
    const DirectedGraph& graph,
    NodeId root,
    LazyLiveEdgeOracle& live_oracle,
    LazyTriggeringEdgeOracle& triggering_oracle,
    TraversalWorkspace& workspace);

// Stops as soon as the reverse traversal reaches any node enabled in stop_mask.
ReverseRREarlyStopResult reverse_rr_set_until_hit(
    PropagationModel model,
    const DirectedGraph& graph,
    NodeId root,
    const std::vector<std::uint8_t>& stop_mask,
    LazyLiveEdgeOracle& live_oracle,
    LazyTriggeringEdgeOracle& triggering_oracle,
    TraversalWorkspace& workspace);

}  // namespace rcd
