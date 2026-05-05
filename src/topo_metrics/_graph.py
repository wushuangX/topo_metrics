"""Core graph data structure and utilities for topological metrics."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import KDTree


@dataclass
class TopoGraph:
    """A graph structure for topological metric computation.

    Nodes store 2D coordinates (absolute, offset already applied).
    Edges store pairs of node indices.
    Spacing converts pixel distances to physical distances.

    Attributes:
        nodes: (N, 2) array of node coordinates.
        edges: (E, 2) array of node index pairs.
        spacing: (sy, sx) physical size per pixel unit. Default (1.0, 1.0).
    """

    nodes: np.ndarray
    edges: np.ndarray
    spacing: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.edges = np.asarray(self.edges, dtype=np.int64)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError(f"nodes must be (N, 2), got {self.nodes.shape}")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError(f"edges must be (E, 2), got {self.edges.shape}")
        if self.nodes.shape[0] == 0 and self.edges.shape[0] > 0:
            raise ValueError("edges reference nodes but nodes is empty")
        if (
            self.edges.shape[0] > 0
            and (np.min(self.edges) < 0 or np.max(self.edges) >= self.nodes.shape[0])
        ):
            raise ValueError("edge indices out of bounds")

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return self.nodes.shape[0]

    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return self.edges.shape[0]

    def physical_distance(self, i: int, j: int) -> float:
        """Compute physical distance between two nodes.

        Args:
            i: Index of the first node.
            j: Index of the second node.

        Returns:
            Physical distance accounting for spacing.
        """
        d = (self.nodes[i] - self.nodes[j]) * np.array(self.spacing)
        return float(np.linalg.norm(d))

    def physical_distance_array(self, points: np.ndarray) -> np.ndarray:
        """Compute pairwise physical distances for an array of point pairs.

        Args:
            points: (M, 2, 2) array where points[k, 0] and points[k, 1]
                are two node coordinates (not indices).

        Returns:
            (M,) array of physical distances.
        """
        diff = (points[:, 0] - points[:, 1]) * np.array(self.spacing)
        return np.linalg.norm(diff, axis=1)

    def _build_adjacency(self) -> dict[int, list[int]]:
        """Build adjacency list from edges.

        Returns:
            Dictionary mapping node index to list of neighbor indices.
        """
        adj: dict[int, list[int]] = {i: [] for i in range(self.num_nodes)}
        for e0, e1 in self.edges:
            adj[int(e0)].append(int(e1))
            adj[int(e1)].append(int(e0))
        return adj

    def _build_sparse_matrix(self) -> csr_matrix:
        """Build sparse adjacency matrix with physical edge weights.

        Returns:
            Sparse (N, N) matrix of physical distances.
        """
        n = self.num_nodes
        if n == 0 or self.num_edges == 0:
            return csr_matrix((n, n))

        rows = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        cols = np.concatenate([self.edges[:, 1], self.edges[:, 0]])

        diff = (self.nodes[self.edges[:, 0]] - self.nodes[self.edges[:, 1]]) * np.array(
            self.spacing
        )
        weights = np.linalg.norm(diff, axis=1)
        data = np.concatenate([weights, weights])

        return csr_matrix((data, (rows, cols)), shape=(n, n))

    def shortest_paths(self, source: int, targets: list[int]) -> dict[int, float]:
        """Compute shortest physical distances from source to targets using Dijkstra.

        Args:
            source: Source node index.
            targets: Target node indices.

        Returns:
            Dictionary mapping target index to shortest distance.
            Unreachable targets have distance -1.0.
        """
        if self.num_nodes == 0:
            return {t: -1.0 for t in targets}

        sparse = self._build_sparse_matrix()
        dist_matrix = shortest_path(sparse, indices=source, directed=False)

        result: dict[int, float] = {}
        for t in targets:
            d = dist_matrix[t]
            result[t] = float(d) if np.isfinite(d) else -1.0
        return result

    def densify(
        self, interval: float = 2.0
    ) -> TopoGraph:
        """Insert intermediate nodes on long edges for uniform sampling.

        Args:
            interval: Target physical distance between inserted nodes.

        Returns:
            New TopoGraph with densified edges.
        """
        if self.num_edges == 0:
            return TopoGraph(
                nodes=self.nodes.copy(),
                edges=self.edges.copy(),
                spacing=self.spacing,
            )

        new_nodes = list(self.nodes)
        new_edges: list[tuple[int, int]] = []
        next_id = self.num_nodes

        spacing_arr = np.array(self.spacing)

        for e0, e1 in self.edges:
            p0 = self.nodes[e0]
            p1 = self.nodes[e1]
            phys_dist = float(np.linalg.norm((p1 - p0) * spacing_arr))

            if phys_dist <= interval:
                new_edges.append((int(e0), int(e1)))
                continue

            n_segments = int(np.ceil(phys_dist / interval))
            prev_id = int(e0)

            for k in range(1, n_segments):
                alpha = k / n_segments
                interp = p0 * (1 - alpha) + p1 * alpha
                new_nodes.append(interp)
                new_edges.append((prev_id, next_id))
                prev_id = next_id
                next_id += 1

            new_edges.append((prev_id, int(e1)))

        return TopoGraph(
            nodes=np.array(new_nodes),
            edges=(
                np.array(new_edges, dtype=np.int64)
                if new_edges
                else np.empty((0, 2), dtype=np.int64)
            ),
            spacing=self.spacing,
        )

    def get_chains(self) -> list[list[int]]:
        """Extract chains (paths between junction/endpoint nodes).

        A chain is a sequence of degree-2 nodes connecting two non-degree-2
        nodes (junctions or endpoints).

        Returns:
            List of chains, each chain is a list of node indices.
        """
        if self.num_nodes == 0:
            return []

        adj = self._build_adjacency()
        degree = {i: len(neighbors) for i, neighbors in adj.items()}
        visited_edges: set[tuple[int, int]] = set()
        chains: list[list[int]] = []

        for node_id in range(self.num_nodes):
            if degree[node_id] == 2:
                continue

            for neighbor in adj[node_id]:
                edge_key = (min(node_id, neighbor), max(node_id, neighbor))
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                chain = [node_id, neighbor]
                visited_edges.add(
                    (min(node_id, neighbor), max(node_id, neighbor))
                )

                current = neighbor
                prev = node_id

                while degree.get(current, 0) == 2:
                    neighbors = adj[current]
                    next_node = neighbors[0] if neighbors[1] == prev else neighbors[1]
                    edge_key = (min(current, next_node), max(current, next_node))
                    visited_edges.add(edge_key)
                    chain.append(next_node)
                    prev = current
                    current = next_node

                chains.append(chain)

        return chains

    def to_networkx(self) -> nx.Graph:
        """Convert to a NetworkX graph with node positions.

        Returns:
            NetworkX Graph with 'pos' node attributes.
        """
        g = nx.Graph()
        for i, pos in enumerate(self.nodes):
            g.add_node(i, pos=tuple(pos))
        for e0, e1 in self.edges:
            g.add_edge(int(e0), int(e1))
        return g

    @classmethod
    def from_networkx(
        cls,
        g: nx.Graph,
        pos_attr: str = "pos",
        spacing: tuple[float, float] = (1.0, 1.0),
    ) -> TopoGraph:
        """Create a TopoGraph from a NetworkX graph.

        Args:
            g: NetworkX graph with node positions.
            pos_attr: Attribute name for node positions. Default "pos".
            spacing: Physical size per pixel unit. Default (1.0, 1.0).

        Returns:
            TopoGraph instance.

        Raises:
            ValueError: If any node is missing the position attribute.
        """
        nodes_list = list(g.nodes())
        if not nodes_list:
            return cls(
                nodes=np.empty((0, 2)),
                edges=np.empty((0, 2), dtype=np.int64),
                spacing=spacing,
            )

        node_id_map: dict[object, int] = {}
        positions: list[np.ndarray] = []
        for idx, node in enumerate(nodes_list):
            node_id_map[node] = idx
            data = g.nodes[node]
            if pos_attr not in data:
                raise ValueError(
                    f"Node {node} missing position attribute '{pos_attr}'"
                )
            pos = data[pos_attr]
            positions.append(np.array(pos, dtype=np.float64))

        edges_list: list[tuple[int, int]] = []
        for u, v in g.edges():
            edges_list.append((node_id_map[u], node_id_map[v]))

        return cls(
            nodes=np.array(positions),
            edges=(
                np.array(edges_list, dtype=np.int64)
                if edges_list
                else np.empty((0, 2), dtype=np.int64)
            ),
            spacing=spacing,
        )


def snap_nodes(
    source_nodes: np.ndarray,
    target_graph: TopoGraph,
    tolerance: float,
    spacing: tuple[float, float] = (1.0, 1.0),
    one_to_one: bool = True,
    exclude_radius: float = 0.0,
) -> np.ndarray:
    """Snap source node positions to nearest target graph nodes.

    Args:
        source_nodes: (M, 2) array of source node coordinates.
        target_graph: Target TopoGraph to snap to.
        tolerance: Maximum physical distance for a valid snap.
        spacing: Physical size per pixel unit.
        one_to_one: If True, each target node matches at most one source.
        exclude_radius: Physical distance around matched targets to exclude
            from further matching (for one-to-one enforcement).

    Returns:
        (M,) array of matched target node indices. -1 means no match found.
    """
    if source_nodes.shape[0] == 0 or target_graph.num_nodes == 0:
        return np.full(source_nodes.shape[0], -1, dtype=np.int64)

    spacing_arr = np.array(spacing)

    source_phys = source_nodes * spacing_arr
    target_phys = target_graph.nodes * spacing_arr

    tree = KDTree(target_phys)
    matches = np.full(source_nodes.shape[0], -1, dtype=np.int64)
    used_targets: set[int] = set()

    distances, indices = tree.query(source_phys)

    for i in range(source_nodes.shape[0]):
        if distances[i] > tolerance:
            continue

        candidate = int(indices[i])

        if one_to_one:
            if candidate in used_targets:
                continue
            matches[i] = candidate
            used_targets.add(candidate)
        else:
            matches[i] = candidate

    return matches
