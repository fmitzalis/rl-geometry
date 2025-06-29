import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


class Mesh:
    def __init__(self, vertices=None, faces=None):
        self.vertices = vertices if vertices is not None else []
        self.faces = faces if faces is not None else []
        self.graph = self._build_graph()

    def _build_graph(self):
        G = nx.Graph()

        # Add vertices
        for i, pos in enumerate(self.vertices):
            G.add_node(i, pos=pos)

        # Add edges from faces
        for face in self.faces:
            for i in range(len(face)):
                G.add_edge(face[i], face[(i + 1) % len(face)])

        return G

    def detect_non_manifold_edges(self):
        """Find edges that are shared by more than two faces."""
        edge_face_count = {}

        for i, face in enumerate(self.faces):
            for j in range(len(face)):
                # Create a sorted edge tuple to ensure consistent keys
                edge = tuple(sorted([face[j], face[(j + 1) % len(face)]]))
                if edge in edge_face_count:
                    edge_face_count[edge].append(i)
                else:
                    edge_face_count[edge] = [i]

        non_manifold_edges = [edge for edge, faces in edge_face_count.items() if len(faces) > 2]
        return non_manifold_edges

    def detect_non_manifold_vertices(self):
        """Detect vertices that have non-manifold connectivity."""
        vertex_edges = {i: [] for i in range(len(self.vertices))}

        # For each face, collect edges connected to each vertex
        for face in self.faces:
            for i in range(len(face)):
                v1, v2 = face[i], face[(i + 1) % len(face)]
                vertex_edges[v1].append((v1, v2))
                vertex_edges[v2].append((v1, v2))

        non_manifold_vertices = []

        # Check topological connectivity around each vertex
        for vertex, edges in vertex_edges.items():
            if len(edges) < 3:
                continue

            # Try to find a consistent ordering of edges
            ordered_edges = [edges[0]]
            tried_edges = set([edges[0]])

            while len(ordered_edges) < len(edges):
                current_edge = ordered_edges[-1]
                next_found = False

                for edge in edges:
                    if edge in tried_edges:
                        continue

                    # Check if this edge connects to the last one
                    if edge[0] == current_edge[1] or edge[1] == current_edge[1]:
                        ordered_edges.append(edge)
                        tried_edges.add(edge)
                        next_found = True
                        break

                if not next_found:
                    # If we can't complete the circuit, vertex is non-manifold
                    non_manifold_vertices.append(vertex)
                    break

                # Check if we've formed a complete cycle
                if len(ordered_edges) == len(edges) and (
                    ordered_edges[0][0] != ordered_edges[-1][1] and ordered_edges[0][1] != ordered_edges[-1][1]
                ):
                    non_manifold_vertices.append(vertex)

        return non_manifold_vertices

    def count_disconnected_components(self):
        """Count number of disconnected components in the mesh."""
        return len(list(nx.connected_components(self.graph)))

    def apply_action(self, action_type, vertex_index=None, params=None):
        """Apply a topology or position modification action."""
        if action_type == "merge_vertices":
            # Merge a non-manifold vertex with its nearest neighbor
            if vertex_index is None or len(self.vertices) <= 1:
                return False

            # Find nearest vertex
            dists = []
            for i, v in enumerate(self.vertices):
                if i != vertex_index:
                    dist = np.linalg.norm(np.array(self.vertices[vertex_index]) - np.array(v))
                    dists.append((dist, i))

            if not dists:
                return False

            nearest_vertex = min(dists, key=lambda x: x[0])[1]

            # Update faces to use nearest vertex instead of target vertex
            for i in range(len(self.faces)):
                for j in range(len(self.faces[i])):
                    if self.faces[i][j] == vertex_index:
                        self.faces[i][j] = nearest_vertex

            # Remove redundant vertices from faces and any degenerate faces
            new_faces = []
            for face in self.faces:
                # Remove duplicate vertices in same face
                unique_face = []
                for v in face:
                    if v not in unique_face:
                        unique_face.append(v)

                # Only keep faces with at least 3 vertices
                if len(unique_face) >= 3:
                    new_faces.append(unique_face)

            self.faces = new_faces

            # Rebuild the graph
            self.graph = self._build_graph()
            return True

        elif action_type == "move_vertex":
            # Move a vertex in a direction to reduce non-manifold structure
            if vertex_index is None or params is None:
                return False

            # params is the movement vector (dx, dy, dz)
            dx, dy, dz = params
            self.vertices[vertex_index] = (
                self.vertices[vertex_index][0] + dx,
                self.vertices[vertex_index][1] + dy,
                self.vertices[vertex_index][2] + dz,
            )

            # Update the graph node position
            nx.set_node_attributes(self.graph, {vertex_index: self.vertices[vertex_index]}, "pos")
            return True

        elif action_type == "edge_split":
            # Split an edge by adding a new vertex
            if params is None:  # params should be (v1, v2) edge
                return False

            v1, v2 = params
            # Create a new vertex in the middle of the edge
            new_vertex_pos = (
                (self.vertices[v1][0] + self.vertices[v2][0]) / 2,
                (self.vertices[v1][1] + self.vertices[v2][1]) / 2,
                (self.vertices[v1][2] + self.vertices[v2][2]) / 2,
            )

            new_vertex_idx = len(self.vertices)
            self.vertices.append(new_vertex_pos)

            # Update faces to include new vertex
            for i in range(len(self.faces)):
                for j in range(len(self.faces[i])):
                    if (self.faces[i][j] == v1 and self.faces[i][(j + 1) % len(self.faces[i])] == v2) or (
                        self.faces[i][j] == v2 and self.faces[i][(j + 1) % len(self.faces[i])] == v1
                    ):
                        # Insert new vertex between v1 and v2
                        self.faces[i].insert((j + 1) % len(self.faces[i]), new_vertex_idx)
                        break

            # Rebuild graph
            self.graph = self._build_graph()
            return True

        return False

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
