"""Category-constrained cardinality-first assignment in pixel coordinates."""
from __future__ import annotations

from dataclasses import dataclass

from src.data.geometry import iou_xyxy


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int
    kind: str = ""
    gt_index: int = -1
    pred_index: int = -1


def global_matches(
    gt: list[tuple[str, tuple[float, float, float, float]]],
    pred: list[tuple[str, tuple[float, float, float, float]]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Return deterministic cardinality-first, maximum-IoU matches.

    A tiny integer-cost min-cost-flow implementation avoids making an optional
    SciPy dependency part of this offline script.  Successive shortest paths
    augment until no path remains, giving maximum cardinality first; negative
    IoU costs then maximize total overlap for that cardinality.
    """
    source = 0
    gt_start = 1
    pred_start = gt_start + len(gt)
    sink = pred_start + len(pred)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]

    def add_edge(left: int, right: int, cost: int, *, kind: str = "", gt_index: int = -1, pred_index: int = -1) -> None:
        forward = _FlowEdge(right, len(graph[right]), 1, cost, kind, gt_index, pred_index)
        reverse = _FlowEdge(left, len(graph[left]), 0, -cost)
        graph[left].append(forward)
        graph[right].append(reverse)

    for gt_index in range(len(gt)):
        add_edge(source, gt_start + gt_index, 0)
    for pred_index in range(len(pred)):
        add_edge(pred_start + pred_index, sink, 0)
    for gt_index, (gt_category, gt_box) in enumerate(gt):
        for pred_index, (pred_category, pred_box) in enumerate(pred):
            if gt_category != pred_category:
                continue
            overlap = iou_xyxy(gt_box, pred_box)
            if overlap >= threshold:
                # Scale enough to preserve ordinary float IoU ordering while
                # keeping costs integral and reproducible.
                add_edge(gt_start + gt_index, pred_start + pred_index, -round(overlap * 1_000_000_000),
                         kind="match", gt_index=gt_index, pred_index=pred_index)

    node_count = len(graph)
    while True:
        distances: list[int | None] = [None] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0
        for _ in range(node_count - 1):
            changed = False
            for left, edges in enumerate(graph):
                if distances[left] is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate = distances[left] + edge.cost
                    if distances[edge.to] is None or candidate < distances[edge.to]:
                        distances[edge.to] = candidate
                        previous[edge.to] = (left, edge_index)
                        changed = True
            if not changed:
                break
        if distances[sink] is None:
            break
        node = sink
        while node != source:
            left, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[left][edge_index]
            edge.capacity = 0
            graph[node][edge.reverse].capacity = 1
            node = left

    matches: list[tuple[int, int, float]] = []
    for gt_index in range(len(gt)):
        for edge in graph[gt_start + gt_index]:
            if edge.kind == "match" and edge.capacity == 0:
                matches.append((gt_index, edge.pred_index, iou_xyxy(gt[gt_index][1], pred[edge.pred_index][1])))
    return sorted(matches, key=lambda item: (item[0], item[1]))
