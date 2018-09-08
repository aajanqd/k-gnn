import torch

import graph_cpu
from glocal_gnn import local_three_graph, malkin_three_graph


def test_three_graph_idx():
    assert graph_cpu.three_graph_idx(0, 1, 2, 5) == 0
    assert graph_cpu.three_graph_idx(0, 1, 3, 5) == 1
    assert graph_cpu.three_graph_idx(0, 1, 4, 5) == 2
    assert graph_cpu.three_graph_idx(0, 2, 3, 5) == 3
    assert graph_cpu.three_graph_idx(0, 2, 4, 5) == 4
    assert graph_cpu.three_graph_idx(0, 3, 4, 5) == 5
    assert graph_cpu.three_graph_idx(1, 2, 3, 5) == 6
    assert graph_cpu.three_graph_idx(1, 2, 4, 5) == 7
    assert graph_cpu.three_graph_idx(1, 3, 4, 5) == 8
    assert graph_cpu.three_graph_idx(2, 3, 4, 5) == 9

    assert graph_cpu.three_graph_idx(2, 1, 0, 5) == 0
    assert graph_cpu.three_graph_idx(3, 0, 1, 5) == 1
    assert graph_cpu.three_graph_idx(4, 1, 0, 5) == 2
    assert graph_cpu.three_graph_idx(3, 0, 2, 5) == 3
    assert graph_cpu.three_graph_idx(4, 2, 0, 5) == 4
    assert graph_cpu.three_graph_idx(4, 0, 3, 5) == 5
    assert graph_cpu.three_graph_idx(3, 2, 1, 5) == 6
    assert graph_cpu.three_graph_idx(4, 1, 2, 5) == 7
    assert graph_cpu.three_graph_idx(4, 3, 1, 5) == 8
    assert graph_cpu.three_graph_idx(4, 2, 3, 5) == 9


def test_local_three_graph():
    # Star graph with 4 nodes.
    row = torch.tensor([0, 0, 0, 1, 2, 3])
    col = torch.tensor([1, 2, 3, 0, 0, 0])
    edge_index = torch.stack([row, col], dim=0)

    edge_index, assignment = local_three_graph(edge_index)
    assert edge_index.tolist() == [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]
    assert assignment.tolist() == [
        [0, 0, 0, 1, 1, 2, 2, 3, 3],
        [0, 1, 2, 0, 1, 0, 2, 1, 2],
    ]

    # Line graph with 4 nodes.
    row = torch.tensor([0, 1, 1, 2, 2, 3])
    col = torch.tensor([1, 0, 2, 1, 3, 2])
    edge_index = torch.stack([row, col], dim=0)

    edge_index, assignment = local_three_graph(edge_index)
    assert edge_index.tolist() == [[0, 1], [1, 0]]
    assert assignment.tolist() == [[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]]


def test_malkin_three_graph():
    # Star graph with 4 nodes.
    row = torch.tensor([0, 0, 0, 1, 2, 3])
    col = torch.tensor([1, 2, 3, 0, 0, 0])
    edge_index = torch.stack([row, col], dim=0)

    edge_index, assignment = malkin_three_graph(edge_index)
    assert edge_index.tolist() == []
    assert assignment.tolist() == []

    # Line graph with 4 nodes.
    row = torch.tensor([0, 1, 1, 2, 2, 3])
    col = torch.tensor([1, 0, 2, 1, 3, 2])
    edge_index = torch.stack([row, col], dim=0)

    edge_index, assignment = malkin_three_graph(edge_index)
    assert edge_index.tolist() == []
    assert assignment.tolist() == []

    # Star graph with an additional edge.
    row = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3])
    col = torch.tensor([1, 2, 3, 0, 2, 0, 1, 0])
    edge_index = torch.stack([row, col], dim=0)

    edge_index, assignment = malkin_three_graph(edge_index)
    assert edge_index.tolist() == [[0, 1], [1, 0]]
    assert assignment.tolist() == [[0, 0, 1, 2, 3, 3], [0, 1, 0, 1, 0, 1]]
