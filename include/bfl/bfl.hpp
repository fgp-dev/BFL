// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace bfl {
using Bit = std::uint8_t;
using NodeId = std::uint32_t;
enum class NodeKind : std::uint8_t { Input, Constant, Mux };
enum class SelectorKind : std::uint8_t { Signal, Adaptive };

// Count plus the first route_prefix_bits effective events in graph order.
// A zero prefix length gives the original count-only route.
struct RouteKey {
    std::size_t count = 0;
    std::uint64_t prefix = 0;
    bool operator<(const RouteKey& other) const noexcept {
        return count < other.count || (count == other.count && prefix < other.prefix);
    }
    bool operator==(const RouteKey& other) const noexcept {
        return count == other.count && prefix == other.prefix;
    }
};

// Adaptive MUXes read a direct binary selector state for their route.
// Signal MUXes read d and have no trainable state or effective change event.
struct Node {
    NodeKind kind = NodeKind::Input;
    SelectorKind selector_kind = SelectorKind::Signal;
    NodeId a = 0, b = 0, d = 0;
    std::uint32_t input_index = 0;
    Bit bit = 0; // Constant value or initial adaptive state.
    std::map<RouteKey, Bit> states; // Materialized route-specific selector states.
};

struct Execution {
    std::vector<Bit> values;
    std::vector<Bit> selected;
    std::vector<RouteKey> routes;
};

// Encode a nonnegative event count as the usual base-two digits, without leading zeroes.
std::vector<Bit> binary_route(std::size_t count);
// One direct selector-state update: S' = S XOR (CHANGE AND USED).
Bit update_state(Bit state, Bit change, Bit used);

class Graph {
public:
    explicit Graph(std::size_t route_prefix_bits = 0);
    NodeId add_input();
    NodeId add_constant(Bit value);
    NodeId add_signal_mux(NodeId a, NodeId b, NodeId selector);
    NodeId add_adaptive_mux(NodeId a, NodeId b, Bit initial_state = 0);
    void add_output(NodeId node);

    // Each bit is the effective CHANGE AND USED event from the previous comparison.
    // The route of MUX j uses only events at earlier MUX IDs.
    void set_routing_events(const std::vector<Bit>& events);
    const std::vector<Bit>& routing_events() const noexcept { return routing_events_; }
    std::size_t route_prefix_bits() const noexcept { return route_prefix_bits_; }
    Bit selector_state(NodeId id, RouteKey route) const;
    void apply_change(NodeId id, RouteKey route, Bit change, Bit used);

    Execution execute(const std::vector<Bit>& input) const;
    std::vector<Bit> predict(const std::vector<Bit>& input) const;
    std::vector<std::vector<Bit>> predict_batch(const std::vector<std::vector<Bit>>& inputs) const;
    // USED traces the selected data branch and, for signal MUXes, selector source.
    std::vector<Bit> used(std::size_t output_index, const Execution& execution) const;

    std::size_t input_count() const noexcept { return input_count_; }
    std::size_t node_count() const noexcept { return nodes_.size(); }
    std::size_t output_count() const noexcept { return outputs_.size(); }
    std::size_t mux_count() const noexcept;
    std::size_t state_count() const noexcept;
    std::size_t memory_bytes() const noexcept;
    const Node& node(NodeId id) const;
    const std::vector<NodeId>& outputs() const noexcept { return outputs_; }
private:
    std::vector<Node> nodes_;
    std::vector<NodeId> outputs_;
    std::vector<Bit> routing_events_;
    std::size_t input_count_ = 0;
    std::size_t route_prefix_bits_ = 0;
};

struct Example { std::vector<Bit> input, target; };
using Dataset = std::vector<Example>;
struct TrainConfig { std::size_t epochs = 1; };
struct TrainReport {
    std::size_t initial_loss = 0, final_loss = 0;
    std::size_t comparisons = 0, state_updates = 0;
    std::size_t distinct_routes = 0, distinct_adaptive_routes = 0;
    std::size_t adaptive_addresses = 0, used_adaptive_addresses = 0, route_visits = 0;
    double mean_examples_per_route = 0;
};

std::size_t hamming_loss(const Graph& graph, const Dataset& data);
// Online, deterministic comparison order: example, then output, with a fresh pass each time.
// CHANGE is OUT XOR TARGET. Every adaptive USED state receives S' = S XOR (C AND U).
// The resulting event vector becomes the routing context for the next comparison.
TrainReport train(Graph& graph, const Dataset& data, const TrainConfig& config = {});
} // namespace bfl
