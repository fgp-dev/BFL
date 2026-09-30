// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#include "bfl/bfl.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
using namespace bfl;

template<class F> void throws(F action) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    assert(caught);
}

void test_state_law() {
    for (Bit state = 0; state <= 1; ++state) {
        assert(update_state(state, 0, 0) == state);
        assert(update_state(state, 0, 1) == state);
        assert(update_state(state, 1, 0) == state);
        assert(update_state(state, 1, 1) == (state ^ 1));
    }
    throws([] { update_state(2, 1, 1); });
}

Graph routing_graph() {
    Graph g;
    const auto zero = g.add_constant(0), one = g.add_constant(1);
    for (int i = 0; i < 6; ++i) g.add_adaptive_mux(zero, one);
    g.add_output(static_cast<NodeId>(g.node_count() - 1));
    return g;
}
void test_route(const std::vector<Bit>& pattern, std::size_t expected) {
    auto g = routing_graph();
    std::vector<Bit> events(g.node_count());
    for (std::size_t i = 0; i < pattern.size(); ++i) events[2 + i] = pattern[i];
    g.set_routing_events(events);
    const auto run = g.execute({});
    assert(run.routes[2 + pattern.size()] == (RouteKey{expected, 0}));
    assert(binary_route(expected) == binary_route(run.routes[2 + pattern.size()].count));
}
void test_routes_and_sharing() {
    test_route({0,0,0,0}, 0);
    test_route({0,1,0,0}, 1);
    test_route({1,0,1,0}, 2);
    test_route({1,1,0,1}, 3);
    test_route({0,1,0,1}, 2);
    assert((binary_route(0) == std::vector<Bit>{0}));
    assert((binary_route(1) == std::vector<Bit>{1}));
    assert((binary_route(2) == std::vector<Bit>{1,0}));
    assert((binary_route(3) == std::vector<Bit>{1,1}));
    auto g = routing_graph();
    std::vector<Bit> first(g.node_count()), second(g.node_count());
    first[2]=first[4]=1;
    second[3]=second[5]=1;
    g.set_routing_events(first);
    const auto a = g.execute({});
    g.apply_change(6, a.routes[6], 1, 1);
    g.set_routing_events(second);
    const auto b = g.execute({});
    assert(a.routes[6] == (RouteKey{2, 0}) && b.routes[6] == (RouteKey{2, 0}));
    assert(b.selected[6] == 1); // The same node and count reuse one selector state.
    throws([&] { g.set_routing_events({1}); });
    auto invalid = second; invalid[0] = 1;
    throws([&] { g.set_routing_events(invalid); });
}
void test_partial_prefix_routes() {
    Graph g(2);
    const auto zero = g.add_constant(0), one = g.add_constant(1);
    for (int i = 0; i < 5; ++i) g.add_adaptive_mux(zero, one);
    std::vector<Bit> events(g.node_count());
    events[2] = events[4] = 1; // 1010: count 2, prefix 10.
    g.set_routing_events(events);
    const auto first = g.execute({}).routes[6];
    assert(first == (RouteKey{2, 2}));
    g.apply_change(6, first, 1, 1);
    events[2] = events[4] = 0;
    events[3] = events[5] = 1; // 0101: same count, prefix 01.
    g.set_routing_events(events);
    const auto second = g.execute({}).routes[6];
    assert(second == (RouteKey{2, 1}));
    assert(g.selector_state(6, second) == 0);
    events[3] = events[5] = 0;
    events[2] = events[5] = 1; // 1001: same count and prefix 10.
    g.set_routing_events(events);
    assert(g.execute({}).routes[6] == first);
    assert(g.selector_state(6, first) == 1);
    throws([] { Graph invalid(65); });
}
void test_mux_used_and_online_training() {
    Graph g;
    const auto x = g.add_input();
    const auto z = g.add_constant(0), o = g.add_constant(1);
    const auto left = g.add_adaptive_mux(z, o);
    const auto unused = g.add_adaptive_mux(z, o);
    const auto root = g.add_signal_mux(left, unused, x);
    g.add_output(root);
    const auto run = g.execute({0});
    const auto used = g.used(0, run);
    assert(used[root] && used[left] && used[x] && !used[unused]);
    const auto report = train(g, {{{0}, {1}}});
    assert(report.initial_loss == 1 && report.final_loss == 0);
    assert(report.comparisons == 1 && report.state_updates == 1);
    assert(g.selector_state(left, {0, 0}) == 1 && g.selector_state(unused, {0, 0}) == 0);
    assert(g.routing_events()[left] == 1 && g.routing_events()[unused] == 0);
    assert(g.execute({0}).routes[root] == (RouteKey{1, 0}));
    assert(g.predict({0})[0] == 1);
    const auto stable = train(g, {{{0}, {1}}});
    assert(stable.state_updates == 0 && g.selector_state(left, {0, 0}) == 1);
    assert(g.routing_events()[left] == 0);
    assert(g.predict_batch({{0},{1}})[0] == g.predict({0}));
    throws([&] { g.predict({2}); });
    throws([&] { train(g, {{{0}, {2}}}); });
}
int main() {
    test_state_law();
    test_routes_and_sharing();
    test_partial_prefix_routes();
    test_mux_used_and_online_training();
    std::cout << "All BFL tests passed\n";
}
