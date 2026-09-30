// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#include "bfl/bfl.hpp"
#include <limits>
#include <set>
#include <stdexcept>

namespace bfl {
namespace {
void require_bit(Bit bit) { if (bit > 1) throw std::invalid_argument("Expected a binary value"); }
void validate_dataset(const Graph& graph, const Dataset& data) {
    if (!graph.output_count()) throw std::invalid_argument("Graph needs an output");
    for (const auto& example : data) {
        if (example.input.size() != graph.input_count() || example.target.size() != graph.output_count())
            throw std::invalid_argument("Dataset shape does not match graph");
        for (Bit bit : example.input) require_bit(bit);
        for (Bit bit : example.target) require_bit(bit);
    }
}
} // namespace

std::vector<Bit> binary_route(std::size_t count) {
    if (!count) return {0};
    std::vector<Bit> bits;
    for (; count; count >>= 1) bits.push_back(static_cast<Bit>(count & 1));
    return {bits.rbegin(), bits.rend()};
}
Bit update_state(Bit state, Bit change, Bit used) {
    require_bit(state); require_bit(change); require_bit(used);
    return static_cast<Bit>(state ^ (change & used));
}
Graph::Graph(std::size_t route_prefix_bits) : route_prefix_bits_(route_prefix_bits) {
    if (route_prefix_bits > 64) throw std::invalid_argument("Route prefix cannot exceed 64 bits");
}
NodeId Graph::add_input() {
    if (nodes_.size() >= std::numeric_limits<NodeId>::max() || input_count_ >= std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("Too many nodes or inputs");
    Node node;
    node.kind = NodeKind::Input;
    node.input_index = static_cast<std::uint32_t>(input_count_++);
    nodes_.push_back(node);
    routing_events_.push_back(0);
    return static_cast<NodeId>(nodes_.size() - 1);
}
NodeId Graph::add_constant(Bit value) {
    require_bit(value);
    if (nodes_.size() >= std::numeric_limits<NodeId>::max()) throw std::length_error("Too many nodes");
    Node node;
    node.kind = NodeKind::Constant; node.bit = value;
    nodes_.push_back(node); routing_events_.push_back(0);
    return static_cast<NodeId>(nodes_.size() - 1);
}
NodeId Graph::add_signal_mux(NodeId a, NodeId b, NodeId selector) {
    if (nodes_.size() >= std::numeric_limits<NodeId>::max()) throw std::length_error("Too many nodes");
    const auto id = static_cast<NodeId>(nodes_.size());
    if (a >= id || b >= id || selector >= id) throw std::invalid_argument("MUX predecessors must have lower node IDs");
    Node node;
    node.kind = NodeKind::Mux; node.selector_kind = SelectorKind::Signal;
    node.a = a; node.b = b; node.d = selector;
    nodes_.push_back(node); routing_events_.push_back(0);
    return id;
}
NodeId Graph::add_adaptive_mux(NodeId a, NodeId b, Bit initial_state) {
    require_bit(initial_state);
    if (nodes_.size() >= std::numeric_limits<NodeId>::max()) throw std::length_error("Too many nodes");
    const auto id = static_cast<NodeId>(nodes_.size());
    if (a >= id || b >= id) throw std::invalid_argument("MUX predecessors must have lower node IDs");
    Node node;
    node.kind = NodeKind::Mux; node.selector_kind = SelectorKind::Adaptive;
    node.a = a; node.b = b; node.bit = initial_state;
    nodes_.push_back(node); routing_events_.push_back(0);
    return id;
}
void Graph::add_output(NodeId id) {
    if (id >= nodes_.size()) throw std::out_of_range("Unknown output node");
    outputs_.push_back(id);
}
const Node& Graph::node(NodeId id) const { return nodes_.at(id); }
void Graph::set_routing_events(const std::vector<Bit>& events) {
    if (events.size() != nodes_.size()) throw std::invalid_argument("Routing event shape does not match graph");
    for (std::size_t i = 0; i < events.size(); ++i) {
        require_bit(events[i]);
        if (events[i] && (nodes_[i].kind != NodeKind::Mux || nodes_[i].selector_kind != SelectorKind::Adaptive))
            throw std::invalid_argument("Only adaptive MUXes can produce effective state changes");
    }
    routing_events_ = events;
}
Bit Graph::selector_state(NodeId id, RouteKey route) const {
    const Node& n = nodes_.at(id);
    if (n.kind != NodeKind::Mux || n.selector_kind != SelectorKind::Adaptive)
        throw std::invalid_argument("Node has no adaptive selector state");
    const auto entry = n.states.find(route);
    return entry == n.states.end() ? n.bit : entry->second;
}
void Graph::apply_change(NodeId id, RouteKey route, Bit change, Bit used) {
    const Bit next = update_state(selector_state(id, route), change, used);
    if (change && used) nodes_[id].states[route] = next;
}
Execution Graph::execute(const std::vector<Bit>& input) const {
    if (input.size() != input_count_) throw std::invalid_argument("Input shape does not match graph");
    for (Bit bit : input) require_bit(bit);
    Execution run{std::vector<Bit>(nodes_.size()), std::vector<Bit>(nodes_.size()),
                  std::vector<RouteKey>(nodes_.size())};
    std::size_t route_count = 0;
    std::uint64_t route_prefix = 0;
    std::size_t mux_index = 0;
    for (std::size_t id = 0; id < nodes_.size(); ++id) {
        const Node& n = nodes_[id];
        if (n.kind == NodeKind::Input) run.values[id] = input[n.input_index];
        else if (n.kind == NodeKind::Constant) run.values[id] = n.bit;
        else {
            const RouteKey route{route_count, route_prefix};
            run.routes[id] = route;
            run.selected[id] = n.selector_kind == SelectorKind::Adaptive
                ? selector_state(static_cast<NodeId>(id), route) : run.values[n.d];
            run.values[id] = run.values[run.selected[id] ? n.b : n.a];
            // Add this event after assigning its route, so routes use only prior events.
            if (mux_index < route_prefix_bits_)
                route_prefix = (route_prefix << 1) | routing_events_[id];
            ++mux_index;
            route_count += routing_events_[id];
        }
    }
    return run;
}
std::vector<Bit> Graph::predict(const std::vector<Bit>& input) const {
    const auto run = execute(input);
    std::vector<Bit> result;
    result.reserve(outputs_.size());
    for (NodeId id : outputs_) result.push_back(run.values[id]);
    return result;
}
std::vector<std::vector<Bit>> Graph::predict_batch(const std::vector<std::vector<Bit>>& inputs) const {
    std::vector<std::vector<Bit>> result;
    result.reserve(inputs.size());
    for (const auto& input : inputs) result.push_back(predict(input));
    return result;
}
std::vector<Bit> Graph::used(std::size_t output_index, const Execution& run) const {
    if (output_index >= outputs_.size()) throw std::out_of_range("Unknown output index");
    if (run.values.size() != nodes_.size() || run.selected.size() != nodes_.size() || run.routes.size() != nodes_.size())
        throw std::invalid_argument("Execution shape does not match graph");
    std::vector<Bit> result(nodes_.size());
    result[outputs_[output_index]] = 1;
    for (std::size_t id = nodes_.size(); id-- > 0;) {
        if (!result[id] || nodes_[id].kind != NodeKind::Mux) continue;
        const Node& n = nodes_[id];
        result[run.selected[id] ? n.b : n.a] = 1;
        if (n.selector_kind == SelectorKind::Signal) result[n.d] = 1;
    }
    return result;
}
std::size_t Graph::mux_count() const noexcept {
    std::size_t count = 0;
    for (const auto& n : nodes_) count += n.kind == NodeKind::Mux;
    return count;
}
std::size_t Graph::state_count() const noexcept {
    std::size_t count = 0;
    for (const auto& n : nodes_) if (n.kind == NodeKind::Mux && n.selector_kind == SelectorKind::Adaptive)
        count += 1 + n.states.size();
    return count;
}
std::size_t Graph::memory_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) + nodes_.capacity() * sizeof(Node) +
        outputs_.capacity() * sizeof(NodeId) + routing_events_.capacity() * sizeof(Bit);
    for (const auto& n : nodes_) bytes += n.states.size() * (sizeof(std::pair<const RouteKey, Bit>) + 3 * sizeof(void*));
    return bytes;
}
std::size_t hamming_loss(const Graph& graph, const Dataset& data) {
    validate_dataset(graph, data);
    std::size_t loss = 0;
    for (const auto& example : data) {
        const auto output = graph.predict(example.input);
        for (std::size_t j = 0; j < output.size(); ++j) loss += output[j] ^ example.target[j];
    }
    return loss;
}
TrainReport train(Graph& graph, const Dataset& data, const TrainConfig& config) {
    validate_dataset(graph, data);
    TrainReport report;
    report.initial_loss = hamming_loss(graph, data);
    std::set<RouteKey> observed_routes;
    std::set<RouteKey> adaptive_routes;
    std::set<std::pair<NodeId, RouteKey>> adaptive_addresses;
    std::set<std::pair<NodeId, RouteKey>> used_adaptive_addresses;
    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        for (const auto& example : data) {
            std::set<RouteKey> example_routes;
            for (std::size_t j = 0; j < graph.output_count(); ++j) {
                const auto run = graph.execute(example.input);
                const Bit change = static_cast<Bit>(run.values[graph.outputs()[j]] ^ example.target[j]);
                const auto used = graph.used(j, run);
                std::vector<Bit> events(graph.node_count());
                for (NodeId id = 0; id < graph.node_count(); ++id) {
                    const Node& n = graph.node(id);
                    if (n.kind != NodeKind::Mux) continue;
                    observed_routes.insert(run.routes[id]);
                    example_routes.insert(run.routes[id]);
                    if (n.selector_kind == SelectorKind::Adaptive) {
                        adaptive_routes.insert(run.routes[id]);
                        adaptive_addresses.insert({id, run.routes[id]});
                        if (used[id]) used_adaptive_addresses.insert({id, run.routes[id]});
                        events[id] = static_cast<Bit>(change & used[id]);
                        graph.apply_change(id, run.routes[id], change, used[id]);
                        report.state_updates += events[id];
                    }
                }
                graph.set_routing_events(events);
                ++report.comparisons;
            }
            report.route_visits += example_routes.size();
        }
    }
    report.final_loss = hamming_loss(graph, data);
    report.distinct_routes = observed_routes.size();
    report.distinct_adaptive_routes = adaptive_routes.size();
    report.adaptive_addresses = adaptive_addresses.size();
    report.used_adaptive_addresses = used_adaptive_addresses.size();
    report.mean_examples_per_route = report.distinct_routes
        ? static_cast<double>(report.route_visits) / report.distinct_routes : 0;
    return report;
}
} // namespace bfl
