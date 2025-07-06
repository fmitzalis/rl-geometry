import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
from typing import Any, Tuple

from Mesh import Mesh, load_stl_to_mesh, save_mesh_to_stl, visualize_mesh_3d
from PolicyGNN import PolicyGNN


class RLAgent:
    def __init__(self, mesh: Mesh, in_features: int = 5, hidden_dim: int = 64):
        self.mesh = mesh
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = 2 * len(mesh.vertices) + 1
        self.model = PolicyGNN(in_features=self.in_features, hidden_dim=self.hidden_dim, out_features=self.out_features)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_log_probs = []
        self.buffer_rewards = []

    def compute_reward(self) -> torch.Tensor:
        non_manifold_edges = self.mesh.detect_non_manifold_edges()
        non_manifold_vertices = self.mesh.detect_non_manifold_vertices()
        num_disconnected = self.mesh.count_disconnected_components()
        reward = -(
            2.0 * len(non_manifold_edges) + 1.5 * len(non_manifold_vertices) + 0.5 * max(0, num_disconnected - 1)
        )
        return torch.tensor(reward, dtype=torch.float)

    def select_action(self, state: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.x.dim() == 1:
            state.x = state.x.unsqueeze(0)
        action_probs = self.model(state.x, state.edge_index)
        if action_probs.dim() == 1:
            action_probs = action_probs.unsqueeze(0)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        return action, log_prob

    def apply_action(self, action_idx: int) -> bool:
        num_vertices = len(self.mesh.vertices)
        if action_idx < num_vertices:
            delta = np.random.uniform(-0.1, 0.1, 3)
            return self.mesh.apply_action("move_vertex", vertex_index=action_idx, params=delta)
        elif action_idx < 2 * num_vertices:
            vertex_idx = action_idx - num_vertices
            if vertex_idx in self.mesh.detect_non_manifold_vertices():
                return self.mesh.apply_action("merge_vertices", vertex_index=vertex_idx)
            return False
        else:
            nm_edges = self.mesh.detect_non_manifold_edges()
            if nm_edges:
                edge = random.choice(nm_edges)
                return self.mesh.apply_action("edge_split", params=edge)
            return False

    def update_policy(self) -> None:
        rewards = self.buffer_rewards
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        policy_loss = 0
        for i, (log_prob, reward) in enumerate(zip(self.buffer_log_probs, rewards)):
            if i >= len(rewards):
                print(f"Warning: Index {i} exceeds reward tensor length {len(rewards)}")
                continue
            try:
                if hasattr(log_prob, "shape") and log_prob.shape:
                    if log_prob.numel() > 1:
                        log_prob = log_prob.mean()
                    else:
                        log_prob = log_prob.reshape(())
                if hasattr(reward, "shape") and reward.shape:
                    if reward.numel() > 1:
                        reward = reward.mean()
                    else:
                        reward = reward.reshape(())
                policy_loss += -log_prob * reward
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                print(f"log_prob shape: {log_prob.shape if hasattr(log_prob, 'shape') else 'N/A'}")
                print(f"reward shape: {reward.shape if hasattr(reward, 'shape') else 'N/A'}")
                continue
        if isinstance(policy_loss, int) and policy_loss == 0:
            print("Warning: Policy loss is zero. Skipping update.")
            return
        self.optimizer.zero_grad()
        if isinstance(policy_loss, torch.Tensor):
            policy_loss.backward()
        else:
            torch.tensor(policy_loss, requires_grad=True).backward()
        self.optimizer.step()

    def train(
        self, num_episodes: int = 100, steps_per_episode: int = 50, visualize: bool = True
    ) -> Tuple[Mesh, PolicyGNN]:
        if visualize:
            fig = visualize_mesh_3d(self.mesh, title="Initial Mesh State")
            plt.pause(1)
        best_reward = float("-inf")
        best_mesh = None
        rewards_history = []
        non_manifold_history = []
        print(
            f"Initial mesh has {len(self.mesh.detect_non_manifold_edges())} non-manifold edges and "
            f"{len(self.mesh.detect_non_manifold_vertices())} non-manifold vertices"
        )
        for episode in range(num_episodes):
            episode_rewards = 0
            for step in range(steps_per_episode):
                state = self.mesh.to_pytorch_geometric()
                action, log_prob = self.select_action(state)
                if action.dim() > 0:
                    action_idx = action[0].item() if action.size(0) > 0 else 0
                else:
                    action_idx = action.item()
                action_success = self.apply_action(action_idx)
                reward = self.compute_reward()
                try:
                    self.buffer_states.append(state)
                    self.buffer_actions.append(action)
                    self.buffer_log_probs.append(log_prob)
                    self.buffer_rewards.append(reward)
                    episode_rewards += reward.item() if hasattr(reward, "item") else float(reward)
                except Exception as e:
                    print(f"Error storing experience: {e}")
                if visualize and (step % 10 == 0 or step == steps_per_episode - 1):
                    fig = visualize_mesh_3d(
                        self.mesh, title=f"Episode {episode+1}, Step {step+1}\n" f"Reward: {reward.item():.2f}"
                    )
                    plt.pause(0.5)
                    plt.close(fig)
                nm_edges = self.mesh.detect_non_manifold_edges()
                nm_vertices = self.mesh.detect_non_manifold_vertices()
                if len(nm_edges) == 0 and len(nm_vertices) == 0:
                    print(f"All non-manifold issues fixed at episode {episode+1}, step {step+1}!")
                    if visualize:
                        visualize_mesh_3d(self.mesh, title="Fixed Mesh - All Issues Resolved")
                        plt.show()
                    return self.mesh, self.model
            rewards_history.append(episode_rewards)
            non_manifold_count = len(self.mesh.detect_non_manifold_edges()) + len(
                self.mesh.detect_non_manifold_vertices()
            )
            non_manifold_history.append(non_manifold_count)
            if episode_rewards > best_reward:
                best_reward = episode_rewards
                best_mesh = Mesh(vertices=self.mesh.vertices.copy(), faces=[face.copy() for face in self.mesh.faces])
            if (episode + 1) % 5 == 0 or episode == num_episodes - 1:
                self.update_policy()
                self.buffer_states = []
                self.buffer_actions = []
                self.buffer_log_probs = []
                self.buffer_rewards = []
            print(
                f"Episode {episode+1}/{num_episodes}, Reward: {episode_rewards:.2f}, "
                f"Non-manifold elements: {non_manifold_count}"
            )
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
            fig = visualize_mesh_3d(self.mesh, title="Final Mesh State")
            plt.show()
            if best_mesh:
                fig = visualize_mesh_3d(best_mesh, title="Best Mesh Found")
                plt.show()
        return best_mesh, self.model


# Usage example in main
if __name__ == "__main__":
    INPUT_STL: str = "non_manifold.stl"
    mesh: Mesh = load_stl_to_mesh(INPUT_STL)
    agent = RLAgent(mesh)
    repaired_mesh, trained_model = agent.train(num_episodes=50, steps_per_episode=30, visualize=True)
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
