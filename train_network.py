import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

from DQN_neural_network import DQN
from environment.environment import Environment
from environment.constants import NUMBER_OF_RAYS

# Hyperparameters
num_episodes = 1000
epsilon = 1  # Start with full exploration
epsilon_min = 0.1  # Minimum exploration
epsilon_decay = 0.996  # Decay rate
gamma = 0.99
batch_size = 64
replay_buffer = deque(maxlen=15000)  # Experience buffer
action_threshold = 0
last_20_rewards = []

# Initialize environment and network
environment = Environment()
input_size = NUMBER_OF_RAYS * 3
output_size = 3  # Actions: turn left, forward, turn right
model = DQN(input_size, output_size)
# model.load_state_dict(torch.load("./models/last_model.pth"))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training Loop
for episode in range(num_episodes):
    state = environment.reset()
    done = False
    total_reward = 0

    while not done:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).reshape([1, NUMBER_OF_RAYS * 3])

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = [random.choice([0, 1]) for _ in range(output_size)]
            while action[0] == 1 and action[2] == 1:
                action = [random.choice([0, 1]) for _ in range(output_size)]
        else:
            with torch.no_grad():
                q_out = model(state_tensor)
                action = list((q_out > action_threshold).int()[0])

        # Step the environment
        next_state, reward, done = environment.step(action, last_20_rewards)
        total_reward += reward

        # Store experience
        replay_buffer.append((state, action, reward, next_state, done))

        # Sample and train
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)

            # Prepare batch data
            states, actions, rewards, next_states, dones = zip(*batch)
            states = np.array(states)
            next_states = np.array(next_states)
            states = torch.FloatTensor(states).reshape([batch_size, NUMBER_OF_RAYS * 3])
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states).reshape([batch_size, NUMBER_OF_RAYS * 3])
            dones = torch.FloatTensor(dones)

            # Compute Q-values and targets
            q_values = model(states)
            next_q_values = model(next_states)

            targets = rewards.unsqueeze(1).expand(-1, 3) + gamma * next_q_values  # * (1 - dones)
            # targets = targets.squeeze(1)

            # Update the model
            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update state
        state = next_state

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if len(last_20_rewards) < 20:
        last_20_rewards.append(total_reward)
    else:
        last_20_rewards.pop(0)
        last_20_rewards.append(total_reward)
        print(len(last_20_rewards))
    print(np.mean(last_20_rewards))
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    print(f"New epsilon: {epsilon}")

    if episode % 10 == 0:
        model_path = f"./models/last_model.pth"
        torch.save(model.state_dict(), model_path)
