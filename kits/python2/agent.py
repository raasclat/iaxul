from lux.utils import direction_to
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

# Hyperparameters
GAMMA = 0.99
TAU = 0.95  # for GAE (Generalized Advantage Estimation)
LR = 1e-4  # Learning rate
EPSILON = 0.2  # PPO Clipping parameter
BATCH_SIZE = 64
UPDATE_STEPS = 5  # Number of policy updates after each set of experiences
ADVANTAGE_NORMALIZE = True  # Normalize advantage for stability

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class Agent:
    def __init__(self, player: str, env_cfg):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        # Initialize PPO networks for males and females
        self.male_actor = self.initialize_network(input_size=10, output_size=6)
        self.female_actor = self.initialize_network(input_size=10, output_size=6)

        self.male_critic = self.initialize_network(input_size=10, output_size=1)
        self.female_critic = self.initialize_network(input_size=10, output_size=1)

        # Optimizers for the PPO networks
        self.male_optimizer = optim.Adam(
            list(self.male_actor.parameters()) + list(self.male_critic.parameters()), lr=LR
        )
        self.female_optimizer = optim.Adam(
            list(self.female_actor.parameters()) + list(self.female_critic.parameters()), lr=LR
        )

        # Replay buffer
        self.memory = deque(maxlen=10000)

        # Step counter for saving and reloading
        self.step_counter = 0

    def initialize_network(self, input_size, output_size):
        class SimpleNN(nn.Module):
            def __init__(self, input_size, output_size):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, output_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        return SimpleNN(input_size, output_size)

    def save_weights(self, path_prefix):
        torch.save(self.male_actor.state_dict(), f"{path_prefix}_male_actor.pth")
        torch.save(self.male_critic.state_dict(), f"{path_prefix}_male_critic.pth")
        torch.save(self.female_actor.state_dict(), f"{path_prefix}_female_actor.pth")
        torch.save(self.female_critic.state_dict(), f"{path_prefix}_female_critic.pth")
        logging.info(f"Saved weights to {path_prefix}_*.pth")

    def reload_weights(self, path_prefix):
        self.male_actor.load_state_dict(torch.load(f"{path_prefix}_male_actor.pth"))
        self.male_critic.load_state_dict(torch.load(f"{path_prefix}_male_critic.pth"))
        self.female_actor.load_state_dict(torch.load(f"{path_prefix}_female_actor.pth"))
        self.female_critic.load_state_dict(torch.load(f"{path_prefix}_female_critic.pth"))
        logging.info(f"Reloaded weights from {path_prefix}_*.pth")

    def process_observation(self, obs, unit_mask, unit_positions, unit_energys, observed_relic_node_positions, observed_relic_nodes_mask, team_points, step, max_steps):
        features = []
        relic_density = np.sum(observed_relic_nodes_mask)  # Count of visible relic nodes
        match_progress = step / max_steps  # Match progress as a fraction

        for unit_id, mask in enumerate(unit_mask):
            if not mask:
                features.append(None)  # Skip units that are not visible or do not exist
                continue

            unit_energy = unit_energys[unit_id]
            unit_pos = unit_positions[unit_id]

            visible_relics = [observed_relic_node_positions[i] for i in np.where(observed_relic_nodes_mask)[0]]
            relic_distances = [abs(unit_pos[0] - r[0]) + abs(unit_pos[1] - r[1]) for r in visible_relics]
            closest_relic_dist = min(relic_distances) if relic_distances else float("inf")

            enemy_positions = np.array(obs["units"]["position"][self.opp_team_id])
            enemy_distances = [abs(unit_pos[0] - e[0]) + abs(unit_pos[1] - e[1]) for e in enemy_positions if np.all(e != -1)]
            closest_enemy_dist = min(enemy_distances) if enemy_distances else float("inf")

            norm_energy = unit_energy / 400.0
            norm_closest_relic_dist = closest_relic_dist / (self.env_cfg["map_width"] + self.env_cfg["map_height"])
            norm_closest_enemy_dist = closest_enemy_dist / (self.env_cfg["map_width"] + self.env_cfg["map_height"])
            norm_team_points = team_points[self.team_id] / 100.0

            features.append([
                norm_energy,
                unit_pos[0] / self.env_cfg["map_width"],
                unit_pos[1] / self.env_cfg["map_height"],
                norm_closest_relic_dist,
                float(relic_density > 0),
                norm_team_points,
                norm_closest_enemy_dist,
                match_progress,
                team_points[self.team_id] / 100.0,
                relic_density / 10.0
            ])

        return features

    def calculate_reward(self, unit_type, unit_energy, closest_enemy_dist, closest_relic_dist, energy_used, points_gained):
        if unit_type == "male":
            return points_gained - energy_used + (1 / (1 + closest_enemy_dist))
        elif unit_type == "female":
            return points_gained + (1 / (1 + closest_relic_dist)) - energy_used
        return 0

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # Save and reload every 95 steps
        self.step_counter += 1
        if self.step_counter % 95 == 0:
            path_prefix = f"/content/iaxul/saved_weights/"
            self.reload_weights(path_prefix)
            self.save_weights(path_prefix)

        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id])  # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"])  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])  # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"])  # Points of each team

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        features = self.process_observation(
            obs,
            unit_mask,
            unit_positions,
            unit_energys,
            observed_relic_node_positions,
            observed_relic_nodes_mask,
            team_points,
            step,
            max_steps=self.env_cfg["max_steps_in_match"]
        )

        for unit_id, feature in enumerate(features):
            if feature is None:
                continue

            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            input_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

            unit_energy = unit_energys[unit_id]
            is_male = unit_energy > 50

            if is_male:
                male_output = self.male_actor(input_tensor)
                predicted_action = torch.argmax(male_output).item()
                actions[unit_id] = [predicted_action, 0, 0]
            else:
                female_output = self.female_actor(input_tensor)
                predicted_action = torch.argmax(female_output).item()
                actions[unit_id] = [predicted_action, 0, 0]

        return actions

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough samples to train

        batch = random.sample(self.memory, BATCH_SIZE)

        for _ in range(UPDATE_STEPS):
            states, actions, rewards, next_states, dones = zip(*batch)

            # Prepare tensors
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Compute target values
            male_values = self.male_critic(states)
            next_male_values = self.male_critic(next_states)

            target_values = rewards + GAMMA * (1 - dones) * next_male_values

            # Update actor and critic
            advantages = target_values - male_values
            if ADVANTAGE_NORMALIZE:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute loss and optimize
            actor_loss = -(self.male_actor(states).gather(1, actions.unsqueeze(-1)) * advantages.detach()).mean()
            critic_loss = nn.MSELoss()(male_values, target_values.detach())

            total_loss = actor_loss + critic_loss
            self.male_optimizer.zero_grad()
            total_loss.backward()
            self.male_optimizer.step()

            # Repeat for the female models
            female_values = self.female_critic(states)
            next_female_values = self.female_critic(next_states)
            target_values = rewards + GAMMA * (1 - dones) * next_female_values

            advantages = target_values - female_values
            if ADVANTAGE_NORMALIZE:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = -(self.female_actor(states).gather(1, actions.unsqueeze(-1)) * advantages.detach()).mean()
            critic_loss = nn.MSELoss()(female_values, target_values.detach())

            total_loss = actor_loss + critic_loss
            self.female_optimizer.zero_grad()
            total_loss.backward()
            self.female_optimizer.step()
