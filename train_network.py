import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DQN_neural_network import DQN
from RL_map_exploration.rl_environment.constants import (
    NBR_OF_FISHES,
    NUMBER_OF_RAYS,
    SEQUENCE_LENGTH,
)
from RL_map_exploration.rl_environment.environment import Environment

# Hyperparameters
num_episodes = 1000
epsilon = 1  # Start with full exploration
epsilon_min = 0.05  # Minimum exploration
epsilon_decay = 0.994  # Decay rate
gamma = 0.99
batch_size = 256
replay_buffer = deque(maxlen=15000)  # Experience buffer
# action_threshold = 0
last_20_rewards = []

# Initialize environment and network
environment = Environment()
input_size = NUMBER_OF_RAYS * 3
output_size = 5  # Actions: turn left, forward, turn right, left+forward, right+forward
model = DQN(input_size, output_size)
target_network = DQN(input_size, output_size)
# initial_model = "./models/trained_model_v2.pth"
# initial_model = "./models/last_model.pth"
initial_model = None
if initial_model is not None:
    model.load_state_dict(torch.load(initial_model))
    target_network.load_state_dict(torch.load(initial_model))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


def get_sequence(history, sequence_length, input_size):
    if len(history) < sequence_length:
        padding = [torch.zeros(input_size)] * (sequence_length - len(history))
        sequence = padding + list(history)
    else:
        sequence = list(history)
    return torch.stack(sequence).unsqueeze(0)  # Add batch dimension


# Training Loop
for episode in range(num_episodes):
    state = environment.reset()
    all_done = False
    total_reward = 0
    current_histories = [deque(maxlen=SEQUENCE_LENGTH) for _ in range(NBR_OF_FISHES)]

    while not all_done:
        # Convert state to tensor
        state_tensor = []
        for i in range(NBR_OF_FISHES):
            state_tensor.append(torch.FloatTensor(state[i]).reshape([1, NUMBER_OF_RAYS * 3]))

        sequences = []
        for i in range(NBR_OF_FISHES):
            # Flatten the current state and add to history
            state_tensor = torch.FloatTensor(state[i]).flatten()
            current_histories[i].append(state_tensor)

            # Prepare the sequence for this agent
            seq = get_sequence(current_histories[i], SEQUENCE_LENGTH, NUMBER_OF_RAYS * 3)
            sequences.append(seq)

        # Epsilon-greedy action selection
        action = []
        if random.random() < epsilon:
            for i in range(NBR_OF_FISHES):
                act_index = random.randint(0, 4)
                act = [0, 0, 0, 0, 0]
                act[act_index] = 1
                action.append(act)
        else:
            for i in range(NBR_OF_FISHES):
                with torch.no_grad():
                    # Input shape: (1, SEQUENCE_LENGTH, input_size)
                    q_values = model(sequences[i])  # Output shape: (1, output_size)
                    q_values = q_values.squeeze(0)  # Remove batch dimension

                    # Choose greedy action
                    max_index = torch.argmax(q_values)
                    act = torch.zeros_like(q_values)
                    act[max_index] = 1
                    action.append(act.tolist())

                    # q_out = model(state_tensor[i]).squeeze()
                    # max_index = torch.argmax(q_out)

                    # # Create a one-hot encoded tensor
                    # act = torch.zeros_like(q_out)
                    # act[max_index] = 1
                    # action.append(list(act))

        # Step the environment
        next_state, reward, done = zip(*environment.step(action, last_20_rewards))
        total_reward += reward[0]

        # Check if all players have finished their episode
        all_done = True if np.all(np.array(done)) else False

        # Store experience
        for i in range(NBR_OF_FISHES):
            if not done[i]:
                if len(current_histories[i]) >= SEQUENCE_LENGTH:
                    next_seq = get_sequence(
                        current_histories[i], SEQUENCE_LENGTH, NUMBER_OF_RAYS * 3
                    )
                    replay_buffer.append((sequences[i], action[i], reward[i], next_seq, done[i]))
                # replay_buffer.append((state[i], action[i], reward[i], next_state[i], done[i]))

        # Sample and train
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            current_seqs, actions, rewards, next_seqs, dones = zip(*batch)

            current_seqs = torch.cat(current_seqs).float()
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_seqs = torch.cat(next_seqs).float()
            dones = torch.FloatTensor(dones)

            # Q-values for current states
            q_values = model(current_seqs)
            q_values = (q_values * actions).sum(dim=1)

            # Target Q-values
            with torch.no_grad():
                next_q_values = target_network(next_seqs)
            max_next_q = next_q_values.max(1).values
            targets = rewards + gamma * max_next_q * (1 - dones)

            # Update model
            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if len(replay_buffer) > batch_size:
        #     batch = random.sample(replay_buffer, batch_size)

        #     # Prepare batch data
        #     states, actions, rewards, next_states, dones = zip(*batch)
        #     states = np.array(states)
        #     next_states = np.array(next_states)
        #     states = torch.FloatTensor(states).reshape([batch_size, NUMBER_OF_RAYS * 3])
        #     actions = torch.FloatTensor(actions)
        #     rewards = torch.FloatTensor(rewards)
        #     next_states = torch.FloatTensor(next_states).reshape([batch_size, NUMBER_OF_RAYS * 3])
        #     dones = torch.FloatTensor(dones)

        #     # Compute Q-values and select the one linked to the performed action
        #     q_values = model(states)
        #     q_values = (q_values * actions).sum(dim=1, keepdim=True).squeeze()

        #     with torch.no_grad():
        #         next_q_values = target_network(next_states)

        #     next_q_values = next_q_values.max(1).values

        #     targets = rewards + gamma * next_q_values

        #     # Update the model
        #     loss = criterion(q_values, targets)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

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

    if episode % 5 == 0:
        model_path = "./models/last_model.pth"
        target_network.load_state_dict(model.state_dict())
        torch.save(model.state_dict(), model_path)
