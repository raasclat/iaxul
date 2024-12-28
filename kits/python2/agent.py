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
LR = 1e-4    # Learning rate
EPSILON = 0.2  # PPO Clipping parameter
BATCH_SIZE = 64
UPDATE_STEPS = 5  # Number of policy updates after each set of experiences
ADVANTAGE_NORMALIZE = True  # Normalize advantage for stability

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Agent:
    def __init__(self, player: str, env_cfg):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.batch_size = BATCH_SIZE  # Batch size for training

        # Initialize models and optimizers
        self.male_actor = self.build_model()
        self.female_actor = self.build_model()
        self.male_critic = self.build_model()
        self.female_critic = self.build_model()
        self.male_optimizer = optim.Adam(self.male_actor.parameters(), lr=LR)
        self.female_optimizer = optim.Adam(self.female_actor.parameters(), lr=LR)

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Assuming 4 possible actions
        )
        return model

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

            # Distance to nearest relic
            visible_relics = [observed_relic_node_positions[i] for i in np.where(observed_relic_nodes_mask)[0]]
            relic_distances = [abs(unit_pos[0] - r[0]) + abs(unit_pos[1] - r[1]) for r in visible_relics]
            closest_relic_dist = min(relic_distances) if relic_distances else float("inf")

            # Distance to nearest enemy unit
            enemy_positions = np.array(obs["units"]["position"][self.opp_team_id])
            enemy_distances = [abs(unit_pos[0] - e[0]) + abs(unit_pos[1] - e[1]) for e in enemy_positions if np.all(e != -1)]
            closest_enemy_dist = min(enemy_distances) if enemy_distances else float("inf")

            # Normalize features
            norm_energy = unit_energy / 400.0
            norm_closest_relic_dist = closest_relic_dist / (self.env_cfg["map_width"] + self.env_cfg["map_height"])
            norm_closest_enemy_dist = closest_enemy_dist / (self.env_cfg["map_width"] + self.env_cfg["map_height"])
            norm_team_points = team_points[self.team_id] / 100.0

            # Feature vector
            features.append([
                norm_energy,                      # Energy
                unit_pos[0] / self.env_cfg["map_width"],  # Normalized X position
                unit_pos[1] / self.env_cfg["map_height"], # Normalized Y position
                norm_closest_relic_dist,          # Distance to nearest relic
                float(relic_density > 0),         # Visibility of relic nodes
                norm_team_points,                 # Normalized team points
                norm_closest_enemy_dist,          # Distance to nearest enemy
                match_progress,                   # Match progress
                team_points[self.team_id] / 100.0, # Total team points (normalized)
                relic_density / 10.0              # Relic density (max normalized to 10 relics)
            ])
        return features

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id])  # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"])  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])  # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"])  # Points of each team

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # Process observations into feature vectors
        features = self.process_observation(
            obs,  # Pass the full observation
            unit_mask, 
            unit_positions, 
            unit_energys, 
            observed_relic_node_positions, 
            observed_relic_nodes_mask, 
            team_points, 
            step, 
            max_steps=self.env_cfg["max_steps_in_match"]
        )

        # Ensure that male_loss and female_loss are initialized as scalars with requires_grad
        male_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        female_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        for unit_id, feature in enumerate(features):
            if feature is None:
                continue

            # Ensure features are clean of NaN or infinity values
            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            input_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Ensure the models are in training mode
            self.male_actor.train()
            self.female_actor.train()

            # Random walk for the first 15 steps of each match
            if step % self.env_cfg["max_steps_in_match"] < 15:
                action = random.choice([0, 1, 2, 3])  # Randomly choose from 4 directions
                logging.debug(f"Step {step}: Unit {unit_id} performing random action {action}")
                actions[unit_id] = [action, 0, 0]
            else:
                # Predict action for male units
                unit_energy = unit_energys[unit_id]
                is_male = unit_energy > 50

                if is_male:
                    male_output = self.male_actor(input_tensor)
                    male_critic_value = self.male_critic(input_tensor)
                    male_loss = male_loss + self.calculate_loss(male_output, male_critic_value)  # Avoid in-place operation
                    action = torch.argmax(male_output).item()
                    logging.debug(f"Step {step}: Unit {unit_id} (male) performing action {action} with output {male_output.tolist()}")
                    actions[unit_id] = [action, 0, 0]  # Choose action based on the highest output
                else:
                    female_output = self.female_actor(input_tensor)
                    female_critic_value = self.female_critic(input_tensor)
                    female_loss = female_loss + self.calculate_loss(female_output, female_critic_value)  # Avoid in-place operation
                    action = torch.argmax(female_output).item()
                    logging.debug(f"Step {step}: Unit {unit_id} (female) performing action {action} with output {female_output.tolist()}")
                    actions[unit_id] = [action, 0, 0]  # Choose action based on the highest output

            # Store experience in replay buffer
            reward = 0  # Placeholder for actual reward calculation
            next_feature = feature  # Placeholder for next state feature
            done = False  # Placeholder for done flag
            self.memory.append((feature, action, reward, next_feature, done))

        # Log the total loss for the current step
        logging.debug(f"Male Loss: {male_loss.item()}")  # Now it should work
        logging.debug(f"Female Loss: {female_loss.item()}")  # Now it should work

        # Train the model using experience replay
        if len(self.memory) >= self.batch_size:
            self.train_model()

        # Save training progress every few steps (for example, after every 500 steps)
        if step % 500 == 0:
            self.save_model(step)

        return actions

    def train_model(self):
        batch = random.sample(self.memory, self.batch_size)
        for feature, action, reward, next_feature, done in batch:
            # Compute target and loss, then perform backpropagation
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            next_feature_tensor = torch.tensor(next_feature, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor([action], dtype=torch.long).unsqueeze(0)  # Ensure correct shape
            reward_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
            done_tensor = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

            # Compute target value
            with torch.no_grad():
                target_value = reward_tensor + (1 - done_tensor) * GAMMA * self.male_critic(next_feature_tensor)

            # Compute current value
            current_value = self.male_critic(feature_tensor)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_value, target_value)

            # Compute advantage
            advantage = target_value - current_value

            # Compute actor loss
            actor_output = self.male_actor(feature_tensor)
            actor_loss = -torch.mean(advantage.detach() * torch.log(actor_output.gather(1, action_tensor)))

            # Backpropagation
            self.male_optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            self.male_optimizer.step()

    def calculate_loss(self, output, critic_value):
        # Simple actor-critic loss example
        advantage = output - critic_value
        loss_fn = nn.MSELoss()
        return loss_fn(output, advantage)

    def save_model(self, step):
        # Save the model state (both actor and critic) and optimizer state
        model_path = f"C:/Users/George/models/agent_step_{step}.pth"
        torch.save({
            'male_actor_state_dict': self.male_actor.state_dict(),
            'female_actor_state_dict': self.female_actor.state_dict(),
            'male_optimizer_state_dict': self.male_optimizer.state_dict(),
            'female_optimizer_state_dict': self.female_optimizer.state_dict(),
        }, model_path)