from typing import List, Tuple, Dict, Any
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict


class Mesh:
    def __init__(self, vertices: List[Tuple[float, float, float]] = None, faces: List[List[int]] = None) -> None:
        self.vertices: List[Tuple[float, float, float]] = vertices if vertices is not None else []
        self.faces: List[List[int]] = faces if faces is not None else []
        self.graph: nx.Graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()

        # Add vertices as nodes
        for i, pos in enumerate(self.vertices):
            G.add_node(i, pos=pos)

        # Add edges from faces
        for face in self.faces:
            for i, v in enumerate(face):
                G.add_edge(v, face[(i + 1) % len(face)])

        return G

    def detect_non_manifold_edges(self) -> List[Tuple[int, int]]:
        """Find edges that are shared by more than two faces."""
        edge_face_count: Dict[Tuple[int, int], List[int]] = {}

        for i, face in enumerate(self.faces):
            for j, _ in enumerate(face):
                # Create a sorted edge tuple to ensure consistent keys (a,b) == (b,a)
                edge = tuple(sorted([face[j], face[(j + 1) % len(face)]]))
                if edge in edge_face_count:
                    edge_face_count[edge].append(i)
                else:
                    edge_face_count[edge] = [i]

        non_manifold_edges: List[Tuple[int, int]] = [edge for edge, faces in edge_face_count.items() if len(faces) > 2]
        return non_manifold_edges

    def can_order_edges(self, edges: set) -> bool:
        """Try to order edges into a single chain or loop."""
        if not edges:
            return True
        edges_list = list(edges)
        chain = [edges_list[0]]
        used = set([edges_list[0]])
        while len(chain) < len(edges_list):
            last = chain[-1]
            found = False
            for e in edges_list:
                if e in used:
                    continue
                if last[1] in e:
                    e_oriented = e if last[1] == e[0] else (e[1], e[0])
                    chain.append(e_oriented)
                    used.add(e)
                    found = True
                    break
            if not found:
                return False
        return chain[0][0] == chain[-1][1]

    def detect_non_manifold_vertices(self) -> List[int]:
        """Detect vertices that have non-manifold connectivity."""
        vertex_edges: Dict[int, set] = defaultdict(set)

        # For each face, collect edges connected to each vertex
        for face in self.faces:
            n = len(face)
            for i, v1 in enumerate(face):
                v2 = face[(i + 1) % n]
                edge = tuple(sorted((v1, v2)))
                vertex_edges[v1].add(edge)
                vertex_edges[v2].add(edge)

        non_manifold_vertices: List[int] = []

        # Check topological connectivity around each vertex
        for v, edges in vertex_edges.items():
            if len(edges) < 3:
                continue
            if not self.can_order_edges(edges):
                non_manifold_vertices.append(v)

        return non_manifold_vertices

    def count_disconnected_components(self) -> int:
        return len(list(nx.connected_components(self.graph)))

    def apply_action(self, action_type: str, vertex_index: int = None, params: Any = None) -> bool:
        """Apply a topology or position modification action."""
        if action_type == "merge_vertices":
            return self._merge_vertex(vertex_index)
        elif action_type == "move_vertex":
            return self._move_vertex(vertex_index, params)
        elif action_type == "edge_split":
            return self._split_edge(params)
        return False

    def _merge_vertex(self, vertex_index: int) -> bool:
        if vertex_index is None or len(self.vertices) <= 1:
            return False
        dists = [
            (np.linalg.norm(np.array(self.vertices[vertex_index]) - np.array(v)), i)
            for i, v in enumerate(self.vertices)
            if i != vertex_index
        ]
        if not dists:
            return False
        nearest_vertex = min(dists, key=lambda x: x[0])[1]
        for face in self.faces:
            for j, v in enumerate(face):
                if v == vertex_index:
                    face[j] = nearest_vertex
        new_faces = []
        for face in self.faces:
            unique_face = []
            for v in face:
                if v not in unique_face:
                    unique_face.append(v)
            if len(unique_face) >= 3:
                new_faces.append(unique_face)
        self.faces = new_faces
        self.graph = self._build_graph()
        return True

    def _move_vertex(self, vertex_index: int, params: Any) -> bool:
        if vertex_index is None or params is None:
            return False
        dx, dy, dz = params
        self.vertices[vertex_index] = (
            self.vertices[vertex_index][0] + dx,
            self.vertices[vertex_index][1] + dy,
            self.vertices[vertex_index][2] + dz,
        )
        nx.set_node_attributes(self.graph, {vertex_index: self.vertices[vertex_index]}, "pos")
        return True

    def _split_edge(self, params: Any) -> bool:
        if params is None:
            return False
        v1, v2 = params
        new_vertex_pos = (
            (self.vertices[v1][0] + self.vertices[v2][0]) / 2,
            (self.vertices[v1][1] + self.vertices[v2][1]) / 2,
            (self.vertices[v1][2] + self.vertices[v2][2]) / 2,
        )
        new_vertex_idx = len(self.vertices)
        self.vertices.append(new_vertex_pos)
        for face in self.faces:
            for j, v in enumerate(face):
                if (v == v1 and face[(j + 1) % len(face)] == v2) or (v == v2 and face[(j + 1) % len(face)] == v1):
                    face.insert((j + 1) % len(face), new_vertex_idx)
                    break
        self.graph = self._build_graph()
        return True

    def to_pytorch_geometric(self) -> Data:
        """Convert mesh to PyTorch Geometric Data object."""
        # Create edge index
        edges = list(self.graph.edges())
        if not edges:
            # Create a dummy edge if none exist
            edge_index = torch.zeros((2, 1), dtype=torch.long)
        else:
            edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

        # Create node features (positions)
        x = torch.tensor(self.vertices, dtype=torch.float)

        # Add additional features for each vertex
        non_manifold_vertices = self.detect_non_manifold_vertices()
        nm_vertex_feature = torch.zeros(len(self.vertices), 1)
        for idx in non_manifold_vertices:
            nm_vertex_feature[idx] = 1.0

        # Add non-manifold edge features
        non_manifold_edges = self.detect_non_manifold_edges()
        nm_edge_feature = torch.zeros(len(self.vertices), 1)
        for edge in non_manifold_edges:
            nm_edge_feature[edge[0]] += 1.0
            nm_edge_feature[edge[1]] += 1.0

        # Combine features
        features = torch.cat([x, nm_vertex_feature, nm_edge_feature], dim=1)

        return Data(x=features, edge_index=edge_index)
