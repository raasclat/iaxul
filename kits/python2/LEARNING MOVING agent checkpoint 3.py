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

        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        # Initialize male and female NNs
        self.male_nn = self.load_nn(r"C:\Users\George\models\male_nn.pth")
        self.female_nn = self.load_nn(r"C:\Users\George\models\female_nn.pth")
        
        # Initialize the PPO networks (actor and critic for male and female)
        self.male_actor = self.load_nn(r"C:\Users\George\models\male_actor.pth")
        self.female_actor = self.load_nn(r"C:\Users\George\models\female_actor.pth")
        
        self.male_critic = self.load_nn(r"C:\Users\George\models\male_critic.pth")
        self.female_critic = self.load_nn(r"C:\Users\George\models\female_critic.pth")

        # Optimizers for the PPO networks
        self.male_optimizer = optim.Adam(list(self.male_actor.parameters()) + list(self.male_critic.parameters()), lr=LR)
        self.female_optimizer = optim.Adam(list(self.female_actor.parameters()) + list(self.female_critic.parameters()), lr=LR)

        # Replay buffer
        self.memory = deque(maxlen=10000)

    def load_nn(self, model_path):
        class SimpleNN(nn.Module):
            def __init__(self, input_size, output_size):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, output_size)
                self.relu = nn.ReLU()
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.softmax(self.fc3(x))
                return x

        model = SimpleNN(input_size=10, output_size=6)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        return model

    def process_observation(self, obs, unit_mask, unit_positions, unit_energys, observed_relic_node_positions, observed_relic_nodes_mask, team_points, step, max_steps):
        """
        Converts the raw observations into a comprehensive feature vector for the NNs.
        """
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

        for unit_id, feature in enumerate(features):
            if feature is None:
                continue

            # Ensure features are clean of NaN or infinity values
            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            input_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Predict action for male units
            unit_energy = unit_energys[unit_id]
            is_male = unit_energy > 50

            if is_male:
                male_output = self.male_nn(input_tensor)
                if torch.isnan(male_output).any():
                    logging.warning(f"NaN detected in Male NN output for unit {unit_id}. Using default action.")
                    actions[unit_id] = [0, 0, 0]  # Default action if NaN detected
                else:
                    predicted_action = torch.argmax(male_output).item()
                    logging.debug(f"Male NN output: {male_output}")
                    if predicted_action == 5:  # Sap action
                        # Find nearest enemy unit to sap
                        enemy_positions = obs["units"]["position"][self.opp_team_id]
                        distances = [abs(unit_positions[unit_id][0] - e[0]) + abs(unit_positions[unit_id][1] - e[1]) for e in enemy_positions if np.all(e != -1)]
                        if distances:
                            nearest_enemy_idx = np.argmin(distances)
                            target = enemy_positions[nearest_enemy_idx]
                            delta_x = target[0] - unit_positions[unit_id][0]
                            delta_y = target[1] - unit_positions[unit_id][1]
                            actions[unit_id] = [5, delta_x, delta_y]
                        else:
                            actions[unit_id] = [0, 0, 0]  # No sap if no enemies visible
                    else:
                        actions[unit_id] = [predicted_action, 0, 0]
            else:
                female_output = self.female_nn(input_tensor)
                if torch.isnan(female_output).any():
                    logging.warning(f"NaN detected in Female NN output for unit {unit_id}. Using default action.")
                    actions[unit_id] = [0, 0, 0]  # Default action if NaN detected
                else:
                    predicted_action = torch.argmax(female_output).item()
                    logging.debug(f"Female NN output: {female_output}")
                    if len(self.relic_node_positions) > 0:
                        # Move towards the nearest relic node
                        nearest_relic = self.relic_node_positions[0]
                        actions[unit_id] = [direction_to(unit_positions[unit_id], nearest_relic), 0, 0]
                    else:
                        actions[unit_id] = [predicted_action, 0, 0]

        return actions