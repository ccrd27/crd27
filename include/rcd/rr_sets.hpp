#pragma once

#include "rcd/graph.hpp"
#include "rcd/propagation.hpp"
#include "rcd/progress.hpp"
#include "rcd/types.hpp"

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace rcd {

struct WeightedRRSample {
    std::vector<NodeId> rr_set;
    NodeId root = 0;
    Weight weight = 0.0;
};

using RootWeightFunction = std::function<Weight(NodeId root, const std::vector<NodeId>& rr_set)>;

// Samples ordinary weighted RR sets under the requested propagation model. The
// supplied root-weight function returns the unscaled root weight; each stored
// sample is multiplied by n for uniform-root RIS estimation.
std::vector<WeightedRRSample> generate_weighted_rr_samples(
    PropagationModel model,
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const RootWeightFunction& root_weight,
    const ProgressCallback& progress_callback = {});

std::vector<WeightedRRSample> generate_weighted_rr_samples_from_node_weights(
    PropagationModel model,
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const std::vector<Weight>& node_weights,
    const ProgressCallback& progress_callback = {});

// Legacy IC-specific wrappers retained for existing callers.
std::vector<WeightedRRSample> generate_weighted_rr_samples_ic(
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const RootWeightFunction& root_weight);

std::vector<WeightedRRSample> generate_weighted_rr_samples_from_node_weights_ic(
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const std::vector<Weight>& node_weights);

Score weighted_coverage_value(
    const std::vector<WeightedRRSample>& samples,
    const std::vector<NodeId>& seeds);

std::vector<NodeId> weighted_coverage_greedy(
    std::size_t num_nodes,
    const std::vector<WeightedRRSample>& samples,
    std::size_t budget,
    const std::vector<std::uint8_t>& candidate_mask = {},
    const ProgressCallback& progress_callback = {});

}  // namespace rcd
