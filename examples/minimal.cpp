// Copyright 2026 Facundo Gomez Prates
// SPDX-License-Identifier: Apache-2.0
#include "bfl/bfl.hpp"
#include <iostream>
using namespace bfl;
int main() {
    Graph graph;
    const auto zero = graph.add_constant(0), one = graph.add_constant(1);
    graph.add_output(graph.add_adaptive_mux(zero, one));
    const auto report = train(graph, {{{}, {1}}});
    std::cout << "initial_loss=" << report.initial_loss << " final_loss=" << report.final_loss
              << " state_updates=" << report.state_updates << " route=";
    for (Bit bit : binary_route(graph.execute({}).routes[2].count)) std::cout << int(bit);
    std::cout << '\n';
}
