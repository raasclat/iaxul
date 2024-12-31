# things i want to change:
# change DQN to PPO
# change the AI to male-female algorithm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)
    
# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left) THIS IS FINE LEAVE IT
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1

# THIS IS FINE LEAVE IT
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, player: str, env_cfg, training=True) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.training = training
        
        # DQN parameters
        self.state_size = 6  # unit_pos(2) + closest_relic(2) + unit_energy(1) + step(1)
        self.action_size = 6  # stay, up, right, down, left, sap
        self.hidden_size = 128
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.maleactor = SimpleNN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.femaleactor = SimpleNN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.malecritic = SimpleNN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.femalecritic = SimpleNN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.malecritic.load_state_dict(self.maleactor.state_dict())
        self.femalecritic.load_state_dict(self.femaleactor.state_dict())
        
        self.maleoptimizer = optim.Adam(self.maleactor.parameters(), lr=self.learning_rate)
        self.femaleoptimizer = optim.Adam(self.femaleactor.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)
        
        self.load_model()
        self.epsilon = 0.0

    def _state_representation(self, unit_pos, unit_energy, relic_nodes, step, relic_mask):
        if not relic_mask.any():
            closest_relic = np.array([-1, -1])
        else:
            visible_relics = relic_nodes[relic_mask]
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]
        
        state = np.concatenate([
            unit_pos,
            closest_relic,
            [unit_energy],
            [step/505.0]  # Normalize step
        ])
        return torch.FloatTensor(state).to(self.device)
    
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        relic_nodes = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        self.score = np.array(obs["team_points"][self.team_id]) #this is the score that your team has gotten
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )

       # if step % 500 == 0:
          #print(f"memory:  {len(self.memory)}")

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_units = np.where(unit_mask)[0]
        
        for unit_id in available_units:
            state = self._state_representation(
                unit_positions[unit_id],
                unit_energys[unit_id],
                relic_nodes,
                step,
                relic_mask
            )

            # action_type = random.randrange(self.action_size)
            self.unit_explore_locations = dict()
            self.relic_node_positions = []
            self.discovered_relic_nodes_ids = set()

            # visible relic nodes
            visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
            # save any new relic nodes that we discover for the rest of the game.
            for id in visible_relic_node_ids:
                if id not in self.discovered_relic_nodes_ids:
                    self.discovered_relic_nodes_ids.add(id)
                    self.relic_node_positions.append(observed_relic_node_positions[id])

            
            if random.random() < self.epsilon and self.training:
                if len(self.relic_node_positions) > 0:
                    nearest_relic_node_position = self.relic_node_positions[0]
                    unit_pos = unit_positions[unit_id]
                    manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])

                    # if close to the relic node we want to move randomly around it and hope to gain points
                    if manhattan_distance <= 4:
                        random_direction = np.random.randint(0, 5)
                        actions[unit_id] = [random_direction, 0, 0]
                    else:
                        # otherwise we want to move towards the relic node
                        actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
                else:
                    #pick a random location on the map for the unit to explore
                    unit_pos = unit_positions[unit_id]
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                    # using the direction_to tool we can generate a direction that makes the unit move to the saved location
                    # note that the first index of each unit's action represents the type of action. See specs for more details
                    actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
            else:
                with torch.no_grad():
                    if unit_energys[unit_id] < 50:
                        q_values = self.femaleactor(state)
                        action_type = q_values.argmax().item()
                    else:
                        q_values = self.maleactor(state)
                        action_type = q_values.argmax().item()
                    #print(f"Q-values: {q_values}")
                if action_type == 5:  # Sap action
                    # Find closest enemy unit
                    opp_positions = obs["units"]["position"][self.opp_team_id]
                    opp_mask = obs["units_mask"][self.opp_team_id]
                    valid_targets = []

                    for opp_id, pos in enumerate(opp_positions):
                        if opp_mask[opp_id] and pos[0] != -1:
                            valid_targets.append(pos)

                    if valid_targets:
                        target_pos = valid_targets[0]  # Choose first valid target
                        actions[unit_id] = [5, target_pos[0], target_pos[1]]
                    else:
                        actions[unit_id] = [0, 0, 0]  # Stay if no valid targets
                else:
                    actions[unit_id] = [action_type, 0, 0]

    
        #print(f (Actions: {actions}")
        
        return actions

    def male_learn(self, step, last_obs, actions, obs, rewards, dones):
        if not self.training or len(self.memory) < self.batch_size:
          return
        
        rewards = np.array(obs["units"]["energy"][self.opp_team_id]) - np.array(last_obs["units"]["energy"][self.opp_team_id])
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.maleactor(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.malecritic(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.maleoptimizer.zero_grad()
        loss.backward()
        self.maleoptimizer.step()
        
        if step % 100 == 0:
            self.malecritic.load_state_dict(self.maleactor.state_dict())

        #print(f"Loss: {loss.item()} Epsilon: {self.epsilon} Score: {rewards} Step: {step}")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def female_learn(self, step, last_obs, actions, obs, rewards, dones):
        if not self.training or len(self.memory) < self.batch_size:
          return
            
        rewards = self.score - np.array(obs["units"]["energy"][self.team_id]) + np.array(last_obs["units"]["energy"][self.team_id])
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.femaleactor(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.femalecritic(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.femaleoptimizer.zero_grad()
        loss.backward()
        self.femaleoptimizer.step()
        
        if step % 100 == 0:
            self.femalecritic.load_state_dict(self.femaleactor.state_dict())

        #print(f"Loss: {loss.item()} Epsilon: {self.epsilon} Score: {rewards} Step: {step}")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self):
        torch.save({
            'maleactor': self.maleactor.state_dict(),
            'femaleactor': self.femaleactor.state_dict(),
            'malecritic': self.malecritic.state_dict(),
            'femalecritic': self.femalecritic.state_dict(),
            'maleoptimizer': self.maleoptimizer.state_dict(),
            'femaleoptimizer': self.femaleoptimizer.state_dict()
        }, f'/content/iaxul/kits/python2/dqn_model_{self.player}.pth')

    def load_model(self):
        try:
            checkpoint = torch.load(f'/content/iaxul/kits/python2/dqn_model_{self.player}.pth', weights_only = True)
            self.maleactor.load_state_dict(checkpoint['maleactor'])
            self.femaleactor.load_state_dict(checkpoint['femaleactor'])
            self.malecritic.load_state_dict(checkpoint['malecritic'])
            self.femalecritic.load_state_dict(checkpoint['femalecritic'])
            self.maleoptimizer.load_state_dict(checkpoint['maleoptimizer'])
            self.femaleoptimizer.load_state_dict(checkpoint['femaleoptimizer'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No trained model found for {self.player}")

##################################################################################
##################################################################################

from luxai_s3.wrappers import LuxAIS3GymEnv

def evaluate_agents(agent_1_cls, agent_2_cls, seed=42, training=True, games_to_play=3):
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=seed)
    
    env_cfg = info["params"]  

    player_0 = Agent("player_0", info["params"], training=training)
    player_1 = Agent("player_1", info["params"], training=training)

    for i in range(games_to_play):
        obs, info = env.reset()
        game_done = False
        step = 0
        last_obs = None
        last_actions = None
        print(f"{i}")
        while not game_done:
            
            actions = {}
            
            # Store current observation for learning
            if training:
                last_obs = {
                    "player_0": obs["player_0"].copy(),
                    "player_1": obs["player_1"].copy()
                }

            # Get actions
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

            if training:
                last_actions = actions.copy()

            # Environment step
            obs, rewards ,terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id],
                "player_1": obs["player_1"]["team_points"][player_1.team_id]
            }  
            # Store experiences and learn
            if training and last_obs is not None:
                # Store experience for each unit
                for agent in [player_0, player_1]:
                    for unit_id in range(env_cfg["max_units"]):
                        if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            current_state = agent._state_representation(
                                last_obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                                last_obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                                last_obs[agent.player]["relic_nodes"],
                                step,
                                last_obs[agent.player]["relic_nodes_mask"]
                            )
                            
                            next_state = agent._state_representation(
                                obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                                obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                                obs[agent.player]["relic_nodes"],
                                step + 1,
                                obs[agent.player]["relic_nodes_mask"]
                            )
                            
                            agent.memory.push(
                                current_state,
                                last_actions[agent.player][unit_id][0],
                                rewards[agent.player],
                                next_state,
                                dones[agent.player]
                            )
                
                # Learn from experiences
                player_0.male_learn(step, last_obs["player_0"], actions["player_0"], 
                             obs["player_0"], rewards["player_0"], dones["player_0"])
                player_1.male_learn(step, last_obs["player_1"], actions["player_1"], 
                             obs["player_1"], rewards["player_1"], dones["player_1"])
                player_0.female_learn(step, last_obs["player_0"], actions["player_0"], 
                             obs["player_0"], rewards["player_0"], dones["player_0"])
                player_1.female_learn(step, last_obs["player_1"], actions["player_1"], 
                             obs["player_1"], rewards["player_1"], dones["player_1"])
                

            if dones["player_0"] or dones["player_1"]:
                game_done = True
                if training:
                    player_0.save_model()
                    player_1.save_model()

            step += 1

    env.close()
    #if training:
    player_0.save_model()
    player_1.save_model()

# Training
#evaluate_agents(Agent, Agent, training=True, games_to_play=50) # 250*5

# Evaluation
#evaluate_agents(Agent, Agent, training=False, games_to_play=5)
