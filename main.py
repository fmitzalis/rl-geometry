import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random


# Define Policy Network with GNN
class PolicyGNN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(PolicyGNN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_features)

    def forward(self, x, edge_index):
        # Use multiple GCN layers for better graph understanding
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return torch.softmax(x, dim=-1)  # Output action probabilities


# Mesh class with proper manifold checking
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

    def to_pytorch_geometric(self):
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


# Reward function
def compute_reward(mesh):
    """Calculate reward based on non-manifold structures."""
    non_manifold_edges = mesh.detect_non_manifold_edges()
    non_manifold_vertices = mesh.detect_non_manifold_vertices()
    num_disconnected = mesh.count_disconnected_components()

    # Weighted reward function - prioritize fixing non-manifold issues
    reward = -(2.0 * len(non_manifold_edges) + 1.5 * len(non_manifold_vertices) + 0.5 * max(0, num_disconnected - 1))

    # Add a small negative reward for extreme actions to prevent oscillations
    return torch.tensor(reward, dtype=torch.float)


# Function to visualize mesh in 3D
def visualize_mesh_3d(mesh, title="Mesh Visualization", highlight_non_manifold=True):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw edges
    for i, j in mesh.graph.edges():
        x = [mesh.vertices[i][0], mesh.vertices[j][0]]
        y = [mesh.vertices[i][1], mesh.vertices[j][1]]
        z = [mesh.vertices[i][2], mesh.vertices[j][2]]
        ax.plot(x, y, z, color="gray", linewidth=1)

    # Draw vertices
    x_vals = [v[0] for v in mesh.vertices]
    y_vals = [v[1] for v in mesh.vertices]
    z_vals = [v[2] for v in mesh.vertices]
    ax.scatter(x_vals, y_vals, z_vals, color="blue", s=20)

    # Highlight non-manifold structures
    if highlight_non_manifold:
        nm_edges = mesh.detect_non_manifold_edges()
        for edge in nm_edges:
            x = [mesh.vertices[edge[0]][0], mesh.vertices[edge[1]][0]]
            y = [mesh.vertices[edge[0]][1], mesh.vertices[edge[1]][1]]
            z = [mesh.vertices[edge[0]][2], mesh.vertices[edge[1]][2]]
            ax.plot(x, y, z, color="red", linewidth=2)

        nm_vertices = mesh.detect_non_manifold_vertices()
        for v_idx in nm_vertices:
            ax.scatter(mesh.vertices[v_idx][0], mesh.vertices[v_idx][1], mesh.vertices[v_idx][2], color="red", s=50)

    ax.set_title(title)
    plt.tight_layout()
    return fig


# Generate a test mesh with non-manifold features
def generate_test_mesh():
    """Generate a simple test mesh with non-manifold features."""
    # Create a cube with an additional face sharing an edge (non-manifold edge)
    vertices = [
        (0, 0, 0),  # 0
        (1, 0, 0),  # 1
        (1, 1, 0),  # 2
        (0, 1, 0),  # 3
        (0, 0, 1),  # 4
        (1, 0, 1),  # 5
        (1, 1, 1),  # 6
        (0, 1, 1),  # 7
        (0.5, 0.5, -0.5),  # 8 - additional vertex for non-manifold issues
    ]

    # Cube faces
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [1, 2, 6, 5],  # right
        [2, 3, 7, 6],  # back
        [3, 0, 4, 7],  # left
        [0, 1, 8],  # additional face creating non-manifold edge (0,1)
        [1, 2, 8],  # additional face creating non-manifold vertex at 1
    ]

    return Mesh(vertices=vertices, faces=faces)


# Select action based on policy network output
def select_action(state, model):
    """Select action based on policy network output."""
    with torch.no_grad():
        # Make sure we're passing tensors with right dimensions
        if state.x.dim() == 1:
            state.x = state.x.unsqueeze(0)

        # Get action probabilities from model
        action_probs = model(state.x, state.edge_index)

        # Ensure action_probs is properly shaped (batch_size, num_actions)
        if action_probs.dim() == 1:
            action_probs = action_probs.unsqueeze(0)

        # Create action distribution
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)

    return action, log_prob


# Apply selected action to the mesh
def apply_action(mesh, action_idx):
    """Apply the selected action to the mesh."""
    num_vertices = len(mesh.vertices)

    # Define action space:
    # - First num_vertices actions: move vertex in random direction
    # - Next num_vertices actions: merge this vertex if it's non-manifold
    # - Final actions: split a non-manifold edge if any exist

    if action_idx < num_vertices:
        # Move vertex action
        # Generate small random movement
        delta = np.random.uniform(-0.1, 0.1, 3)
        return mesh.apply_action("move_vertex", vertex_index=action_idx, params=delta)

    elif action_idx < 2 * num_vertices:
        # Merge vertex action
        vertex_idx = action_idx - num_vertices
        # Only apply merge if it's a non-manifold vertex
        if vertex_idx in mesh.detect_non_manifold_vertices():
            return mesh.apply_action("merge_vertices", vertex_index=vertex_idx)
        return False

    else:
        # Split edge action
        nm_edges = mesh.detect_non_manifold_edges()
        if nm_edges:
            # Select a random non-manifold edge to split
            edge = random.choice(nm_edges)
            return mesh.apply_action("edge_split", params=edge)
        return False


# Load STL to Mesh object
def load_stl_to_mesh(stl_file_path):
    """Load an STL file into our Mesh structure."""
    try:
        from stl import mesh as stl_mesh

        stl_data = stl_mesh.Mesh.from_file(stl_file_path)

        # Extract vertices and faces from STL
        vertices = []
        faces = []
        vertex_dict = {}  # Used to identify unique vertices

        for i, face in enumerate(stl_data.vectors):
            face_indices = []
            for vertex in face:
                # Convert to tuple for hashability
                v_tuple = tuple(vertex)
                # Check if we've seen this vertex before
                if v_tuple not in vertex_dict:
                    vertex_dict[v_tuple] = len(vertices)
                    vertices.append(v_tuple)
                # Add vertex index to face
                face_indices.append(vertex_dict[v_tuple])
            faces.append(face_indices)

        return Mesh(vertices=vertices, faces=faces)
    except ImportError:
        print("numpy-stl package required for STL loading. Install with: pip install numpy-stl")
        return None
    except Exception as e:
        print(f"Error loading STL file: {e}")
        return None


# Main RL training loop
def train_rl_agent(mesh=None, num_episodes=100, steps_per_episode=50, visualize=True):
    """Train an RL agent to repair mesh non-manifold issues."""
    if mesh is None:
        mesh = generate_test_mesh()

    # Define model
    in_features = 5  # x,y,z coordinates + 2 non-manifold indicators
    hidden_dim = 64
    # Action space is 2*num_vertices (move or merge) + 1 (edge split)
    out_features = 2 * len(mesh.vertices) + 1

    model = PolicyGNN(in_features=in_features, hidden_dim=hidden_dim, out_features=out_features)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Experience buffer for more stable training
    buffer_states = []
    buffer_actions = []
    buffer_log_probs = []
    buffer_rewards = []

    # Initial mesh state
    if visualize:
        fig = visualize_mesh_3d(mesh, title="Initial Mesh State")
        plt.pause(1)

    # Statistics tracking
    best_reward = float("-inf")
    best_mesh = None
    rewards_history = []
    non_manifold_history = []

    print(
        f"Initial mesh has {len(mesh.detect_non_manifold_edges())} non-manifold edges and "
        f"{len(mesh.detect_non_manifold_vertices())} non-manifold vertices"
    )

    for episode in range(num_episodes):
        episode_rewards = 0

        for step in range(steps_per_episode):
            # Convert mesh to PyTorch Geometric data
            state = mesh.to_pytorch_geometric()

            # Select action
            action, log_prob = select_action(state, model)

            # Fix: Safely extract action index
            if action.dim() > 0:
                action_idx = action[0].item() if action.size(0) > 0 else 0
            else:
                action_idx = action.item()

            # Apply action
            action_success = apply_action(mesh, action_idx)

            # Compute reward
            reward = compute_reward(mesh)

            # Store experience (with proper tensor handling)
            try:
                # Convert to scalar if needed
                if hasattr(reward, "shape") and reward.numel() > 1:
                    buffer_reward = reward.clone().detach().mean()  # Average if multi-dimensional
                else:
                    buffer_reward = reward.clone().detach()

                if hasattr(log_prob, "shape") and log_prob.numel() > 1:
                    buffer_log_prob = log_prob.clone().detach().mean()  # Average if multi-dimensional
                else:
                    buffer_log_prob = log_prob.clone().detach()

                buffer_states.append(state)
                buffer_actions.append(action.clone().detach())
                buffer_log_probs.append(buffer_log_prob)
                buffer_rewards.append(buffer_reward)

                episode_rewards += reward.item() if hasattr(reward, "item") else float(reward)
            except Exception as e:
                print(f"Error storing experience: {e}")

            # Visualize every few steps if requested
            if visualize and (step % 10 == 0 or step == steps_per_episode - 1):
                fig = visualize_mesh_3d(
                    mesh, title=f"Episode {episode+1}, Step {step+1}\n" f"Reward: {reward.item():.2f}"
                )
                plt.pause(0.5)
                plt.close(fig)

            # Check if we've fixed all non-manifold issues
            nm_edges = mesh.detect_non_manifold_edges()
            nm_vertices = mesh.detect_non_manifold_vertices()
            if len(nm_edges) == 0 and len(nm_vertices) == 0:
                print(f"All non-manifold issues fixed at episode {episode+1}, step {step+1}!")
                if visualize:
                    visualize_mesh_3d(mesh, title="Fixed Mesh - All Issues Resolved")
                    plt.show()
                return mesh, model

        # Update statistics
        rewards_history.append(episode_rewards)
        non_manifold_count = len(mesh.detect_non_manifold_edges()) + len(mesh.detect_non_manifold_vertices())
        non_manifold_history.append(non_manifold_count)

        # Save best mesh so far
        if episode_rewards > best_reward:
            best_reward = episode_rewards
            # Create a deep copy of the current best mesh
            best_mesh = Mesh(vertices=mesh.vertices.copy(), faces=[face.copy() for face in mesh.faces])

        # Update policy with collected experience
        if (episode + 1) % 5 == 0 or episode == num_episodes - 1:
            update_policy(model, optimizer, buffer_states, buffer_actions, buffer_log_probs, buffer_rewards)
            # Clear experience buffer
            buffer_states = []
            buffer_actions = []
            buffer_log_probs = []
            buffer_rewards = []

        print(
            f"Episode {episode+1}/{num_episodes}, Reward: {episode_rewards:.2f}, "
            f"Non-manifold elements: {non_manifold_count}"
        )

    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    plt.plot(non_manifold_history)
    plt.title("Non-manifold Elements")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    if visualize:
        # Show final mesh state
        fig = visualize_mesh_3d(mesh, title="Final Mesh State")
        plt.show()

        # Show best mesh found
        if best_mesh:
            fig = visualize_mesh_3d(best_mesh, title="Best Mesh Found")
            plt.show()

    return best_mesh, model


# Policy update function
def update_policy(model, optimizer, states, actions, log_probs, rewards):
    """Update policy using policy gradient."""
    # Convert rewards to tensor if they're not already
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float)

    # Normalize rewards for training stability
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Calculate policy loss
    policy_loss = 0
    for i, (log_prob, reward) in enumerate(zip(log_probs, rewards)):
        # Debug info
        if i >= len(rewards):
            print(f"Warning: Index {i} exceeds reward tensor length {len(rewards)}")
            continue

        try:
            # Make sure we're dealing with scalar values
            if hasattr(log_prob, "shape") and log_prob.shape:
                if log_prob.numel() > 1:
                    log_prob = log_prob.mean()  # Average if multi-dimensional
                else:
                    log_prob = log_prob.reshape(())  # Convert to scalar

            if hasattr(reward, "shape") and reward.shape:
                if reward.numel() > 1:
                    reward = reward.mean()  # Average if multi-dimensional
                else:
                    reward = reward.reshape(())  # Convert to scalar

            # Add to policy loss
            policy_loss += -log_prob * reward
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            print(f"log_prob shape: {log_prob.shape if hasattr(log_prob, 'shape') else 'N/A'}")
            print(f"reward shape: {reward.shape if hasattr(reward, 'shape') else 'N/A'}")
            continue

    # Check if we have a valid loss
    if isinstance(policy_loss, int) and policy_loss == 0:
        print("Warning: Policy loss is zero. Skipping update.")
        return

    # Update model
    optimizer.zero_grad()

    # Handle scalar vs tensor loss
    if isinstance(policy_loss, torch.Tensor):
        policy_loss.backward()
    else:
        # If somehow we got a non-tensor, create a tensor
        torch.tensor(policy_loss, requires_grad=True).backward()

    optimizer.step()


# Save mesh to STL file
def save_mesh_to_stl(mesh, output_file):
    """Save mesh to an STL file."""
    try:
        from stl import mesh as stl_mesh
        import numpy as np

        # Count triangular faces (we need to triangulate any quads)
        triangle_count = 0
        for face in mesh.faces:
            if len(face) == 3:
                triangle_count += 1
            elif len(face) == 4:
                triangle_count += 2  # Quad will be split into 2 triangles
            else:
                # Simple triangulation: connect first vertex to all others
                triangle_count += len(face) - 2

        # Create STL mesh
        stl_data = stl_mesh.Mesh(np.zeros(triangle_count, dtype=stl_mesh.Mesh.dtype))

        # Add faces to STL mesh
        i = 0
        for face in mesh.faces:
            if len(face) == 3:
                # Triangle: directly add
                stl_data.vectors[i] = np.array([mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]])
                i += 1
            elif len(face) == 4:
                # Quad: split into 2 triangles
                stl_data.vectors[i] = np.array([mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]])
                i += 1
                stl_data.vectors[i] = np.array([mesh.vertices[face[0]], mesh.vertices[face[2]], mesh.vertices[face[3]]])
                i += 1
            else:
                # N-gon: fan triangulation from first vertex
                for j in range(1, len(face) - 1):
                    stl_data.vectors[i] = np.array(
                        [mesh.vertices[face[0]], mesh.vertices[face[j]], mesh.vertices[face[j + 1]]]
                    )
                    i += 1

        # Save the mesh to file
        stl_data.save(output_file)
        print(f"Mesh saved successfully to {output_file}")
        return True
    except ImportError:
        print("numpy-stl package required for STL saving. Install with: pip install numpy-stl")
        return False
    except Exception as e:
        print(f"Error saving STL file: {e}")
        return False


# Main execution
if __name__ == "__main__":
    # Either generate a test mesh with known issues or load from STL
    use_test_mesh = True
    input_stl = None  # "path/to/your/input.stl"

    if use_test_mesh:
        mesh = generate_test_mesh()
        print("Using test mesh with known non-manifold issues")
    elif input_stl:
        mesh = load_stl_to_mesh(input_stl)
        if mesh is None:
            print("Falling back to test mesh")
            mesh = generate_test_mesh()
    else:
        mesh = generate_test_mesh()

    # Train the RL agent
    repaired_mesh, trained_model = train_rl_agent(mesh=mesh, num_episodes=50, steps_per_episode=30, visualize=True)

    # Save the result if desired
    if repaired_mesh:
        save_mesh_to_stl(repaired_mesh, "repaired_mesh.stl")

    print("Process completed!")

# Non manifold types - https://www.sculpteo.com/en/3d-learning-hub/create-3d-file/fix-non-manifold-geometry/
# 1. three or more faces meet at an edge
# 2. several surfaces meet at a vertex
# 3. open object
# 4. internal faces
# 5. opposite normals between adjacent faces


"""
Steps to do:
1. Understand how mesh is captured in the form of graph
2. Understand the reward function and how it's linked to the model outputs
3. Understand the action selection process
4. Understand the training loop
5. Understand training of the policy and how to generate realistic shapes
6. is 2D useful or straight to 3d
"""
