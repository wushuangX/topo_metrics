"""Core graph data structure and utilities for topological metrics."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import KDTree
from skimage.feature import corner_harris, corner_peaks, peak_local_max
from skimage.morphology import skeletonize


@dataclass
class TopoGraph:
    """A graph structure for topological metric computation.

    Nodes store 2D coordinates (absolute, offset already applied).
    Edges store pairs of node indices.
    Spacing converts pixel distances to physical distances.

    Attributes:
        nodes: (N, 2) array of node coordinates.
        edges: (E, 2) array of node index pairs.
        spacing: (sx, sy) physical size per pixel unit. Default (1.0, 1.0).
    """

    nodes: np.ndarray
    edges: np.ndarray
    spacing: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        """Validate and coerce node/edge arrays after initialization."""
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
) -> np.ndarray:
    """Snap source node positions to nearest target graph nodes.

    Args:
        source_nodes: (M, 2) array of source node coordinates.
        target_graph: Target TopoGraph to snap to.
        tolerance: Maximum physical distance for a valid snap.
        spacing: Physical size per pixel unit.
        one_to_one: If True, each target node matches at most one source.

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

def _find_1d_segments(indices: np.ndarray) -> list[list[int]]:
    """Find continuous segments in a sorted 1D array of indices.

    Args:
        indices: Sorted 1D array of integer indices.

    Returns:
        List of lists, each containing a continuous segment of indices.
    """
    if len(indices) == 0:
        return []

    segments = []
    current_segment = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > 1:
            segments.append(current_segment)
            current_segment = [indices[i]]
        else:
            current_segment.append(indices[i])

    segments.append(current_segment)
    return segments


def _extract_edge_peaks(
    mask: np.ndarray,
) -> np.ndarray:
    """Extract peak nodes from tile edges.

    For each of the 4 image edges (1 pixel wide strip):
    1. Find road pixels on the edge.
    2. Locate the center point(s) along each road segment on the edge.
    3. Handle corner points only once (a corner is the intersection of
       two edges).

    Args:
        mask: Binary mask, 1=foreground, 0=background. Shape (H, W).

    Returns:
        (N, 2) array of (row, col) coordinates for edge peaks.
    """
    h, w = mask.shape

    edge_coords_list: list[tuple[int, int]] = []
    added_points: set[tuple[int, int]] = set()

    def _add_point(y: int, x: int) -> None:
        if (y, x) not in added_points:
            edge_coords_list.append((y, x))
            added_points.add((y, x))

    # Top edge (y = 0): include left corner, exclude right corner
    top_road = np.where(mask[0, :] == 1)[0]
    if len(top_road) > 0:
        segments = _find_1d_segments(top_road)
        for seg in segments:
            center_x = (seg[0] + seg[-1]) // 2
            if not (center_x == w - 1):
                _add_point(0, int(center_x))

    # Bottom edge (y = h-1): exclude left corner, include right corner
    bottom_road = np.where(mask[h - 1, :] == 1)[0]
    if len(bottom_road) > 0:
        segments = _find_1d_segments(bottom_road)
        for seg in segments:
            center_x = (seg[0] + seg[-1]) // 2
            if not (center_x == 0):
                _add_point(h - 1, int(center_x))

    # Left edge (x = 0): include bottom corner (already handled top-left), exclude top corner
    left_road = np.where(mask[:, 0] == 1)[0]
    if len(left_road) > 0:
        segments = _find_1d_segments(left_road)
        for seg in segments:
            center_y = (seg[0] + seg[-1]) // 2
            if not (center_y == 0):
                _add_point(int(center_y), 0)

    # Right edge (x = w-1): exclude top corner, include bottom corner
    right_road = np.where(mask[:, w - 1] == 1)[0]
    if len(right_road) > 0:
        segments = _find_1d_segments(right_road)
        for seg in segments:
            center_y = (seg[0] + seg[-1]) // 2
            if not (center_y == 0):
                _add_point(int(center_y), w - 1)

    if not edge_coords_list:
        return np.empty((0, 2), dtype=np.int64)

    return np.array(edge_coords_list, dtype=np.int64)


def _sknw_neighbors(shape: tuple[int, ...]) -> np.ndarray:
    """Compute 8-neighbor flat index offsets for a 2D array with 1-pixel buffer.

    The offsets account for the 1-pixel zero border added during buffering.

    Args:
        shape: Full shape of the buffered array (H+2, W+2).

    Returns:
        (8,) array of flat index offsets for the 8 neighbors.
    """
    block = np.ones((3, 3), dtype=np.uint8)
    block[1, 1] = 0
    idx = np.argwhere(block > 0) - 1  # offsets relative to center
    acc = np.array([shape[1], 1], dtype=np.int64)
    return idx @ acc


def _sknw_mark(img: np.ndarray, nbs: np.ndarray) -> None:
    """Classify skeleton pixels into edge (1) and node (2) pixels.

    Pixels with exactly 2 skeleton neighbors are classified as edge pixels
    (value 1). All other skeleton pixels (endpoints, junctions) are
    classified as node pixels (value 2). Operates on a flat view,
    modifying img in-place.

    Args:
        img: 2D uint16 array of the buffered skeleton. Modified in-place.
        nbs: 8-neighbor flat offsets from _sknw_neighbors.
    """
    flat = img.ravel()
    n = len(flat)
    for p in range(n):
        if flat[p] == 0:
            continue
        s = 0
        for dp in nbs:
            if flat[p + dp] != 0:
                s += 1
        if s == 2:
            flat[p] = 1
        else:
            flat[p] = 2


def _sknw_idx2rc(idx: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Convert flat indices to (row, col) coordinates with buffer offset removal.

    Args:
        idx: (N,) array of flat indices in the buffered array.
        acc: Accumulator array [width, 1] from shape.

    Returns:
        (N, 2) int16 array of (row, col) coordinates in the original
        (un-buffered) coordinate system.
    """
    rst = np.zeros((len(idx), 2), dtype=np.int16)
    remaining = idx.copy()
    for j in range(2):
        rst[:, j] = remaining // acc[j]
        remaining -= rst[:, j] * acc[j]
    rst -= 1  # offset for the 1-pixel buffer
    return rst


def _sknw_fill(
    img: np.ndarray,
    p: int,
    num: int,
    nbs: np.ndarray,
    acc: np.ndarray,
) -> np.ndarray:
    """Flood-fill a connected cluster of node pixels into a single node.

    Starting from pixel *p*, BFS-collects all connected pixels with the
    same value, assigns them a common node marker (*num*), and returns
    their coordinates.

    Args:
        img: 2D uint16 array (flat view used internally). Modified in-place.
        p: Flat index of the starting pixel.
        num: Unique node marker value to assign.
        nbs: 8-neighbor flat offsets.
        acc: Accumulator array for coordinate conversion.

    Returns:
        (M, 2) array of (row, col) coordinates for all pixels in the
        node cluster.
    """
    flat = img.ravel()
    back = flat[p]
    flat[p] = num
    buf = [p]
    cur = 0
    while cur < len(buf):
        p = buf[cur]
        for dp in nbs:
            cp = p + dp
            if flat[cp] == back:
                flat[cp] = num
                buf.append(cp)
        cur += 1
    return _sknw_idx2rc(np.array(buf, dtype=np.int64), acc)


def _sknw_trace(
    img: np.ndarray,
    p: int,
    nbs: np.ndarray,
    acc: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    """Trace an edge along skeleton pixels between two node markers.

    Walks along edge pixels (value 1) from *p* until both endpoint node
    markers (value >= 10) are found. Edge pixels are consumed (set to 0)
    as they are traversed.

    Args:
        img: 2D uint16 array. Modified in-place (edge pixels cleared).
        p: Flat index of the starting edge pixel adjacent to a node.
        nbs: 8-neighbor flat offsets.
        acc: Accumulator array for coordinate conversion.

    Returns:
        Tuple of (node_id_1, node_id_2, edge_pixel_coords) where
        node_id_1 and node_id_2 are the node indices (0-based) and
        edge_pixel_coords is an (M, 2) array of (row, col) coordinates.
    """
    flat = img.ravel()
    c1 = 0
    c2 = 0
    newp = 0
    path_indices: list[int] = []

    while True:
        path_indices.append(p)
        flat[p] = 0  # mark as visited
        for dp in nbs:
            cp = p + dp
            if flat[cp] >= 10:  # node marker
                if c1 == 0:
                    c1 = flat[cp]
                else:
                    c2 = flat[cp]
            if flat[cp] == 1:  # edge pixel
                newp = cp
        p = newp
        if c2 != 0:
            break

    pts = _sknw_idx2rc(np.array(path_indices, dtype=np.int64), acc)
    return c1 - 10, c2 - 10, pts


def _sknw_parse_with_keypoints(
    buf: np.ndarray,
    keypoint_rc: np.ndarray,
    snap_radius: int = 5,
) -> tuple[list[np.ndarray], list[tuple[int, int, np.ndarray]], list[str]]:
    """Parse a skeleton into nodes and edges, forcing keypoints as nodes.

    After standard skeleton pixel classification, additional keypoint
    positions are forced to node status. Keypoints that fall off the
    skeleton are snapped to the nearest skeleton pixel within a search
    radius.

    Args:
        buf: Skeleton image with 1-pixel zero border (uint8 or bool).
            Shape (H+2, W+2).
        keypoint_rc: (N, 2) array of (row, col) keypoint positions in
            the original (un-buffered) coordinate system.
        snap_radius: Maximum city-block search radius for snapping a
            keypoint to the nearest skeleton pixel when it falls off the
            skeleton.

    Returns:
        Tuple of (nodes, edges, node_sources) where:
        - nodes: List of (M_i, 2) arrays of pixel coordinates per node.
        - edges: List of (node_id_1, node_id_2, pixel_coords (P, 2)) tuples.
        - node_sources: List of source labels ("keypoint", "natural",
          or "both") per node.
    """
    img = buf.astype(np.uint16).copy()
    shape = img.shape
    nbs = _sknw_neighbors(shape)
    acc = np.array([shape[1], 1], dtype=np.int64)
    width = shape[1]

    # Step 1: standard mark
    _sknw_mark(img, nbs)

    flat = img.ravel()

    # Step 2: build set of keypoint flat indices (in buffered coordinates)
    # Keypoints that land on skeleton=0 are snapped to the nearest skeleton
    # pixel within the search radius.
    kp_set: set[int] = set()
    if len(keypoint_rc) > 0:
        for row, col in keypoint_rc:
            r_buf = row + 1
            c_buf = col + 1
            idx = r_buf * width + c_buf
            if flat[idx] != 0:
                # Directly on skeleton
                if flat[idx] == 1:
                    flat[idx] = 2
                kp_set.add(idx)
            else:
                # Snap to nearest skeleton pixel
                found = False
                for dr in range(-snap_radius, snap_radius + 1):
                    for dc in range(-snap_radius, snap_radius + 1):
                        nr = r_buf + dr
                        nc = c_buf + dc
                        if 1 <= nr < shape[0] - 1 and 1 <= nc < shape[1] - 1:
                            nidx = nr * width + nc
                            if flat[nidx] != 0:
                                if flat[nidx] == 1:
                                    flat[nidx] = 2
                                kp_set.add(nidx)
                                found = True
                                break
                    if found:
                        break

    # Step 3: fill -- merge connected node pixels into graph nodes
    node_markers = np.where(flat == 2)[0]

    num = 10
    nodes: list[np.ndarray] = []
    node_has_keypoint: list[bool] = []
    for p in node_markers:
        if flat[p] == 2:
            # Collect cluster
            cluster_indices: list[int] = [p]
            is_keypoint = p in kp_set

            # BFS fill
            flat[p] = num
            cur = 0
            while cur < len(cluster_indices):
                pp = cluster_indices[cur]
                for dp in nbs:
                    cp = pp + dp
                    if flat[cp] == 2:
                        flat[cp] = num
                        cluster_indices.append(cp)
                        if cp in kp_set:
                            is_keypoint = True
                cur += 1

            nds = _sknw_idx2rc(np.array(cluster_indices, dtype=np.int64), acc)
            nodes.append(nds)
            node_has_keypoint.append(is_keypoint)
            num += 1

    # Step 4: trace edges
    edges: list[tuple[int, int, np.ndarray]] = []
    for p in node_markers:
        for dp in nbs:
            if flat[p + dp] == 1:
                n1, n2, pts = _sknw_trace(img, p + dp, nbs, acc)
                edges.append((n1, n2, pts))

    # Build source labels
    node_sources: list[str] = []
    for i in range(len(nodes)):
        has_kp = node_has_keypoint[i]
        deg = sum(1 for n1, n2, _ in edges if n1 == i or n2 == i)
        if has_kp and deg >= 3:
            node_sources.append("both")
        elif has_kp:
            node_sources.append("keypoint")
        else:
            node_sources.append("natural")

    return nodes, edges, node_sources


def _build_graph_hybrid(
    skeleton: np.ndarray,
    all_keypoint_coords: np.ndarray,
    border_natural_margin: int = 2,
    keypoint_snap_radius: int = 5,
) -> tuple[nx.Graph, dict[int, tuple[int, int]]]:
    """Build a graph using keypoints as forced nodes and skeleton-traced edges.

    Combines keypoint extraction with sknw edge tracing. Keypoints are
    injected as mandatory nodes on the skeleton; natural skeleton endpoints
    and junctions that are not near any keypoint are also kept. Edges are
    traced along actual skeleton pixels.

    Args:
        skeleton: 2D binary/uint8 skeleton array. Shape (H, W).
        all_keypoint_coords: (N, 2) array of (row, col) keypoint positions.
        border_natural_margin: Remove natural (non-keypoint) skeleton endpoint
            nodes within this many pixels of the image border.
        keypoint_snap_radius: Maximum search radius for snapping a
            keypoint to the nearest skeleton pixel when it falls outside
            the skeleton.

    Returns:
        Tuple of (G, node_positions) where:
        - G: networkx.Graph with node attributes including 'pos', 'pts',
          'source', and 'type'.
        - node_positions: dict mapping node_id to (x, y) coordinates
          where x=column, y=row.
    """
    if not np.any(skeleton) and len(all_keypoint_coords) == 0:
        return nx.Graph(), {}

    ske = skeleton.astype(np.uint8)
    buf = np.zeros((ske.shape[0] + 2, ske.shape[1] + 2), dtype=np.uint8)
    buf[1:-1, 1:-1] = ske

    if len(all_keypoint_coords) == 0:
        all_keypoint_coords = np.empty((0, 2), dtype=np.int64)

    nodes, edges, node_sources = _sknw_parse_with_keypoints(buf, all_keypoint_coords, snap_radius=keypoint_snap_radius)

    if len(nodes) == 0:
        return nx.Graph(), {}

    G = nx.Graph()
    node_positions: dict[int, tuple[int, int]] = {}

    for i, pts in enumerate(nodes):
        centroid = pts.mean(axis=0)  # (row, col)
        row, col = int(round(centroid[0])), int(round(centroid[1]))
        G.add_node(i, pts=pts, source=node_sources[i])
        node_positions[i] = (col, row)  # (x, y) convention

    for n1, n2, pts in edges:
        if n1 < len(nodes) and n2 < len(nodes):
            weight = float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())
            if not G.has_edge(n1, n2):
                G.add_edge(n1, n2, weight=weight, pts=pts)

    # Assign node types based on degree
    for nid in G.nodes:
        deg = G.degree(nid)
        if deg == 1:
            G.nodes[nid]["type"] = "endpoint"
        elif deg >= 3:
            G.nodes[nid]["type"] = "junction"
        else:
            G.nodes[nid]["type"] = "path"

    # Remove natural endpoint nodes near image border.
    # These are unstable skeleton endpoints whose coverage is already
    # handled by edge_peak keypoints from _extract_edge_peaks().
    ske_h, ske_w = skeleton.shape
    nodes_to_remove: list[int] = []
    for nid in list(G.nodes):
        x, y = node_positions[nid]  # (col, row)
        is_border = (
            y < border_natural_margin
            or y >= ske_h - border_natural_margin
            or x < border_natural_margin
            or x >= ske_w - border_natural_margin
        )
        if (
            is_border
            and G.nodes[nid].get("source") == "natural"
            and G.degree(nid) == 1
        ):
            nodes_to_remove.append(nid)
    for nid in nodes_to_remove:
        G.remove_node(nid)
        del node_positions[nid]

    return G, node_positions


def mask_to_nx_graph(
    mask: np.ndarray,
    min_distance: int = 8,
    drop_border_corners: int = 5,
    border_natural_margin: int = 2,
    keypoint_snap_radius: int = 5,
) -> tuple[nx.Graph, dict[int, tuple[int, int]]]:
    """Convert a binary mask to a graph using the hybrid method.

    Combines keypoint extraction (distance transform peaks, Harris corners,
    edge peaks) with skeleton pixel tracing (sknw-style) for edge construction.

    Pipeline:
        1. Skeletonize the mask.
        2. Extract keypoints: distance transform peaks, Harris corners
           (filtered), edge peaks.
        3. Classify skeleton pixels as edge or node pixels.
        4. Force keypoint positions onto skeleton as additional nodes.
        5. Trace edges along skeleton pixels between nodes.
        6. Remove unstable natural endpoint nodes near image border.

    Args:
        mask: 2D binary array where 1=foreground, 0=background.
            Shape (H, W). uint8 or boolean.
        min_distance: Minimum distance between distance transform peaks.
            Controls node density -- smaller values yield more nodes.
        drop_border_corners: Drop Harris corners within this many pixels
            of the image border to reduce noise.
        border_natural_margin: Remove natural (non-keypoint) skeleton
            endpoint nodes within this many pixels of the image border.
        keypoint_snap_radius: Maximum search radius for snapping a
            keypoint to the nearest skeleton pixel when it falls outside
            the skeleton.

    Returns:
        Tuple of (G, node_positions) where:
        - G: networkx.Graph with nodes and edges, plus 'pos' node
          attribute as (x, y) = (col, row).
        - node_positions: dict mapping node_id to (x, y) coordinates
          where x=column, y=row (suitable for imshow overlay).
    """
    skeleton = skeletonize(mask)
    dist_transform = distance_transform_edt(mask) * skeleton

    peak_coords = peak_local_max(
        dist_transform, min_distance=min_distance, exclude_border=False
    )
    corner_coords = corner_peaks(
        corner_harris(skeleton, k=0.04),
        min_distance=5,
        threshold_rel=0.05,
        exclude_border=False,
    )
    if len(corner_coords) > 0:
        h, w = mask.shape
        corner_keep = (
            (corner_coords[:, 0] >= drop_border_corners)
            & (corner_coords[:, 0] < h - drop_border_corners)
            & (corner_coords[:, 1] >= drop_border_corners)
            & (corner_coords[:, 1] < w - drop_border_corners)
        )
        corner_coords = corner_coords[corner_keep]

    edge_peak_coords = _extract_edge_peaks(mask)

    if len(peak_coords) + len(corner_coords) + len(edge_peak_coords) == 0:
        return nx.Graph(), {}

    all_keypoint_coords = np.concatenate(
        [peak_coords, corner_coords, edge_peak_coords], axis=0
    )

    G, node_positions = _build_graph_hybrid(
        skeleton,
        all_keypoint_coords,
        border_natural_margin=border_natural_margin,
        keypoint_snap_radius=keypoint_snap_radius,
    )

    # Add pos attribute for TopoGraph.from_networkx compatibility
    for nid in G.nodes:
        G.nodes[nid]["pos"] = node_positions[nid]

    return G, node_positions


def mask_to_graph(
    mask: np.ndarray,
    min_distance: int = 8,
    drop_border_corners: int = 5,
    border_natural_margin: int = 2,
    keypoint_snap_radius: int = 5,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> TopoGraph:
    """Convert a binary mask to a TopoGraph using the hybrid method.

    This is the primary entry point for mask-to-graph conversion in
    topo-metrics. See mask_to_nx_graph for algorithm details.

    Args:
        mask: 2D binary array, 1=foreground, 0=background. Shape (H, W).
        min_distance: Minimum distance between distance transform peaks.
        drop_border_corners: Drop Harris corners near image border.
        border_natural_margin: Remove natural endpoints near border.
        keypoint_snap_radius: Snap radius for keypoints off-skeleton.
        spacing: Physical size per pixel unit for TopoGraph.
            Default (1.0, 1.0).

    Returns:
        TopoGraph with nodes at (x, y) float coordinates and edges
        connecting them along the skeleton topology.
    """
    g, positions = mask_to_nx_graph(
        mask,
        min_distance=min_distance,
        drop_border_corners=drop_border_corners,
        border_natural_margin=border_natural_margin,
        keypoint_snap_radius=keypoint_snap_radius,
    )
    return TopoGraph.from_networkx(g, pos_attr="pos", spacing=spacing)
