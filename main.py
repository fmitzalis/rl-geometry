import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random

from Mesh import Mesh, load_stl_to_mesh, save_mesh_to_stl
from PolicyGNN import PolicyGNN


# Mesh class with proper manifold checking
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


# Select action based on policy network output
def select_action(state, model):
    """Select action based on policy network output."""
    if state.x.dim() == 1:
        state.x = state.x.unsqueeze(0)

    action_probs = model(state.x, state.edge_index)

    if action_probs.dim() == 1:
        action_probs = action_probs.unsqueeze(0)

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
# Main RL training loop
def train_rl_agent(mesh: Mesh, num_episodes=100, steps_per_episode=50, visualize=True):
    """Train an RL agent to repair mesh non-manifold issues."""

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
                buffer_states.append(state)
                buffer_actions.append(action)
                buffer_log_probs.append(log_prob)  # Do NOT detach here
                buffer_rewards.append(reward)  # Do NOT detach here

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
if __name__ == "__main__":
    INPUT_STL = "non_manifold.stl"
    mesh = load_stl_to_mesh(INPUT_STL)
    repaired_mesh, trained_model = train_rl_agent(mesh=mesh, num_episodes=50, steps_per_episode=30, visualize=True)
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
1. Understand how mesh is captured in the form of graph - Done
2. Understand the reward function and how it's linked to the model outputs
3. Understand the action selection process
4. Understand the training loop
5. Understand training of the policy and how to generate realistic shapes
6. is 2D useful or straight to 3d
"""
