#include "rcd/rr_sets.hpp"

#include <algorithm>
#include <limits>
#include <queue>
#include <string>
#include <stdexcept>

namespace rcd {
namespace {

void validate_node_weights(std::size_t num_nodes, const std::vector<Weight>& node_weights) {
    if (node_weights.size() != num_nodes) {
        throw std::invalid_argument("Node weight vector size does not match graph size");
    }
}

std::vector<std::uint8_t> make_all_candidates(std::size_t num_nodes) {
    return std::vector<std::uint8_t>(num_nodes, 1);
}

void validate_candidate_mask(std::size_t num_nodes, const std::vector<std::uint8_t>& candidate_mask) {
    if (!candidate_mask.empty() && candidate_mask.size() != num_nodes) {
        throw std::invalid_argument("Candidate mask size does not match node count");
    }
}

}  // namespace

std::vector<WeightedRRSample> generate_weighted_rr_samples(
    PropagationModel model,
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const RootWeightFunction& root_weight,
    const ProgressCallback& progress_callback) {
    if (!root_weight) {
        throw std::invalid_argument("Root weight function must be provided");
    }
    if (graph.num_nodes() == 0) {
        return {};
    }

    std::mt19937_64 root_rng(seed);
    std::uniform_int_distribution<std::size_t> root_dist(0, graph.num_nodes() - 1);
    LazyLiveEdgeOracle live_oracle(graph.num_edges(), seed ^ 0x9e3779b97f4a7c15ULL);
    LazyTriggeringEdgeOracle triggering_oracle(
        graph.num_nodes(), seed ^ 0xbf58476d1ce4e5b9ULL);
    TraversalWorkspace workspace(graph.num_nodes());
    const std::size_t progress_interval = std::max<std::size_t>(1, sample_count / 10);

    std::vector<WeightedRRSample> samples;
    samples.reserve(sample_count);
    if (progress_callback) {
        progress_callback(
            "Weighted RR sampling started (total_samples=" + std::to_string(sample_count) + ")",
            0.0);
    }

    for (std::size_t i = 0; i < sample_count; ++i) {
        const NodeId root = static_cast<NodeId>(root_dist(root_rng));
        std::vector<NodeId> rr_set = reverse_rr_set(
            model, graph, root, live_oracle, triggering_oracle, workspace);
        std::sort(rr_set.begin(), rr_set.end());
        const Weight sample_weight = static_cast<Weight>(graph.num_nodes()) * root_weight(root, rr_set);
        samples.push_back(WeightedRRSample{std::move(rr_set), root, sample_weight});
        live_oracle.reset();
        triggering_oracle.reset();
        if (progress_callback &&
            ((i + 1) % progress_interval == 0 || i + 1 == sample_count)) {
            progress_callback(
                "Weighted RR sampling",
                100.0 * static_cast<double>(i + 1) /
                    static_cast<double>(std::max<std::size_t>(sample_count, 1)));
        }
    }

    return samples;
}

std::vector<WeightedRRSample> generate_weighted_rr_samples_from_node_weights(
    PropagationModel model,
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const std::vector<Weight>& node_weights,
    const ProgressCallback& progress_callback) {
    validate_node_weights(graph.num_nodes(), node_weights);
    return generate_weighted_rr_samples(
        model,
        graph,
        sample_count,
        seed,
        [&node_weights](NodeId root, const std::vector<NodeId>&) {
            return node_weights[root];
        },
        progress_callback);
}

std::vector<WeightedRRSample> generate_weighted_rr_samples_ic(
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const RootWeightFunction& root_weight) {
    return generate_weighted_rr_samples(
        PropagationModel::IC, graph, sample_count, seed, root_weight);
}

std::vector<WeightedRRSample> generate_weighted_rr_samples_from_node_weights_ic(
    const DirectedGraph& graph,
    std::size_t sample_count,
    std::uint64_t seed,
    const std::vector<Weight>& node_weights) {
    return generate_weighted_rr_samples_from_node_weights(
        PropagationModel::IC, graph, sample_count, seed, node_weights);
}

Score weighted_coverage_value(
    const std::vector<WeightedRRSample>& samples,
    const std::vector<NodeId>& seeds) {
    NodeId max_node = 0;
    for (const auto seed : seeds) {
        max_node = std::max(max_node, seed);
    }
    for (const auto& sample : samples) {
        for (const auto node : sample.rr_set) {
            max_node = std::max(max_node, node);
        }
    }

    std::vector<std::uint8_t> selected(static_cast<std::size_t>(max_node) + 1, 0);
    for (const auto seed : seeds) {
        selected[seed] = 1;
    }

    Score value = 0.0;
    for (const auto& sample : samples) {
        for (const auto node : sample.rr_set) {
            if (selected[node]) {
                value += sample.weight;
                break;
            }
        }
    }
    return value / static_cast<Score>(std::max<std::size_t>(samples.size(), 1));
}

std::vector<NodeId> weighted_coverage_greedy(
    std::size_t num_nodes,
    const std::vector<WeightedRRSample>& samples,
    std::size_t budget,
    const std::vector<std::uint8_t>& candidate_mask,
    const ProgressCallback& progress_callback) {
    validate_candidate_mask(num_nodes, candidate_mask);
    const std::vector<std::uint8_t> all_candidates =
        candidate_mask.empty() ? make_all_candidates(num_nodes) : candidate_mask;

    std::vector<std::vector<RRIndex>> inverted_index(num_nodes);
    for (RRIndex sample_id = 0; sample_id < samples.size(); ++sample_id) {
        for (const auto node : samples[sample_id].rr_set) {
            if (static_cast<std::size_t>(node) >= num_nodes) {
                throw std::out_of_range("RR-set contains node outside candidate universe");
            }
            if (all_candidates[node]) {
                inverted_index[node].push_back(sample_id);
            }
        }
    }

    std::vector<std::uint8_t> covered(samples.size(), 0);
    std::vector<std::uint8_t> selected(num_nodes, 0);

    struct GainEntry {
        Score gain;
        NodeId node;
    };
    struct GainEntryLess {
        bool operator()(const GainEntry& lhs, const GainEntry& rhs) const {
            if (lhs.gain != rhs.gain) {
                return lhs.gain < rhs.gain;
            }
            return lhs.node > rhs.node;
        }
    };

    // Lazy greedy keeps one upper-bound gain per candidate. Since coverage only
    // grows, a cached gain never increases; stale entries are recomputed only
    // when they reach the top of the priority queue.
    std::priority_queue<GainEntry, std::vector<GainEntry>, GainEntryLess> gain_queue;
    for (std::size_t node_index = 0; node_index < num_nodes; ++node_index) {
        const NodeId node = static_cast<NodeId>(node_index);
        if (!all_candidates[node]) {
            continue;
        }
        Score gain = 0.0;
        for (const auto sample_id : inverted_index[node]) {
            gain += samples[sample_id].weight;
        }
        gain_queue.push(GainEntry{gain, node});
    }

    std::vector<NodeId> result;
    result.reserve(std::min<std::size_t>(budget, num_nodes));
    const std::size_t progress_interval = std::max<std::size_t>(1, budget / 10);

    for (std::size_t round = 0; round < budget; ++round) {
        bool selected_node = false;
        while (!gain_queue.empty()) {
            const GainEntry entry = gain_queue.top();
            gain_queue.pop();
            if (selected[entry.node]) {
                continue;
            }

            Score current_gain = 0.0;
            for (const auto sample_id : inverted_index[entry.node]) {
                if (!covered[sample_id]) {
                    current_gain += samples[sample_id].weight;
                }
            }

            // The cached value is an upper bound. Reinsert a refreshed value
            // when coverage invalidated the cached marginal gain.
            if (current_gain < entry.gain) {
                gain_queue.push(GainEntry{current_gain, entry.node});
                continue;
            }
            if (current_gain <= 0.0) {
                gain_queue = {};
                break;
            }

            selected[entry.node] = 1;
            result.push_back(entry.node);
            for (const auto sample_id : inverted_index[entry.node]) {
                covered[sample_id] = 1;
            }
            selected_node = true;
            if (progress_callback &&
                (result.size() % progress_interval == 0 || result.size() == budget)) {
                progress_callback(
                    "Greedy selection",
                    100.0 * static_cast<double>(result.size()) /
                        static_cast<double>(std::max<std::size_t>(budget, 1)));
            }
            break;
        }

        if (!selected_node) {
            break;
        }
    }

    return result;
}

}  // namespace rcd
