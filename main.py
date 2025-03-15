import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import random

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import random


# Define GNN Policy Network
class MeshGNN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(MeshGNN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_features)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  # Output is not necessarily probabilities, so we can treat this as raw values


# Example reward function
def compute_reward(mesh):
    num_non_manifold_edges = count_non_manifold_edges(mesh)
    num_non_manifold_vertices = count_non_manifold_vertices(mesh)
    num_disconnected_components = count_disconnected_components(mesh)

    reward = -(num_non_manifold_edges + num_non_manifold_vertices + num_disconnected_components)
    return reward


# Function to visualize mesh
def visualize_mesh(mesh, title="Mesh Visualization"):
    plt.figure(figsize=(6, 6))
    nx.draw(mesh, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.title(title)
    plt.show()


def train_rl_agent():
    model = MeshGNN(in_features=3, hidden_dim=16, out_features=4)  # Output 4 possible actions
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    mesh = generate_random_mesh()
    visualize_mesh(mesh, title="Initial Mesh")
    state = convert_mesh_to_graph(mesh)

    for episode in range(1000):
        for step in range(50):
            action = select_action(state, model)  # Select action from model
            mesh = apply_action(mesh, action)  # Apply selected action to the mesh
            reward = compute_reward(mesh)  # Compute the reward based on the mesh state

            if step % 10 == 0:
                visualize_mesh(mesh, title=f"Mesh at Step {step} in Episode {episode}")

            optimizer.zero_grad()

            # Get model output (this could be logits or raw values, e.g., Q-values)
            model_output = model(
                state.x, state.edge_index
            )  # Output is size [num_nodes, 4] (e.g., Q-values or action logits)

            # Assuming model_output contains action values (e.g., Q-values for 4 possible actions)
            action_probabilities = torch.softmax(model_output, dim=-1)  # Softmax to get probabilities
            log_prob = torch.log(action_probabilities[0, action])  # Log of the probability of the chosen action

            # Compute loss using the log-probability of the action and reward
            loss = -log_prob * reward  # Policy gradient loss: maximize reward

            # Ensure loss is a scalar (in case it's not)
            loss = loss.mean()  # This guarantees loss is a scalar

            loss.backward()
            optimizer.step()

            if reward > -0.1:  # Threshold for a well-formed mesh
                break

    visualize_mesh(mesh, title="Final Mesh")
    print("Training complete!")


# Placeholder functions
def count_non_manifold_edges(mesh):
    return random.randint(0, 10)  # Simulating a varying count


def count_non_manifold_vertices(mesh):
    return random.randint(0, 5)  # Simulating a varying count


def count_disconnected_components(mesh):
    return random.randint(0, 3)  # Simulating a varying count


def generate_random_mesh():
    G = nx.erdos_renyi_graph(10, 0.3)  # Generating a random graph
    return G


def convert_mesh_to_graph(mesh):
    edge_index = torch.tensor(list(mesh.edges), dtype=torch.long).t()
    x = torch.rand(len(mesh.nodes), 3)  # Random node features
    return Data(x=x, edge_index=edge_index)


def select_action(state, model):
    # Select the action with the highest Q-value (policy gradient) for each node
    with torch.no_grad():
        model_output = model(state.x, state.edge_index)
        action_probabilities = torch.softmax(model_output, dim=-1)  # Assuming we want action probabilities
        actions = torch.argmax(action_probabilities, dim=-1)  # Select actions with highest probabilities for each node
    return actions  # This will return a tensor with one action per node


def apply_action(mesh, action):
    if random.random() > 0.5:  # Simulating an action effect
        new_edge = (random.randint(0, 9), random.randint(0, 9))
        mesh.add_edge(*new_edge)
    return mesh  # Returning modified mesh


# Run test example
if __name__ == "__main__":
    train_rl_agent()
