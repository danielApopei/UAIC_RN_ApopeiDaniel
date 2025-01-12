import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import flappy_bird_gymnasium
import gymnasium

# -------------------
# 1) Q-Network
# -------------------
class QNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.fc(x)

# -------------------
# 2) Hyperparameters
# -------------------
REPLAY_MEMORY_SIZE = 100_000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1  # Maintain some exploration
EPSILON_DECAY = 30_000  # Slower decay for extended exploration
TARGET_UPDATE = 500
LEARNING_RATE = 1e-4
MAX_FRAMES = 60_000
FRAME_SKIP = 4  # Number of frames to skip after taking an action

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )

# -------------------
# 3) Environment Setup
# -------------------
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
num_actions = env.action_space.n
input_size = env.observation_space.shape[0] + 2  # LIDAR + bird velocity + gap center

# -------------------
# 4) Create Networks
# -------------------
policy_net = QNetwork(input_size, num_actions).to("cpu")
target_net = QNetwork(input_size, num_actions).to("cpu")
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

# -------------------
# 5) Epsilon-Greedy Policy
# -------------------
def epsilon_greedy_policy(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cpu")
    with torch.no_grad():
        q_values = policy_net(state_t)
        return q_values.argmax(dim=1).item()

# -------------------
# 6) Normalize State
# -------------------
def normalize_state(state):
    max_x_position = 1.0  # Adjust based on your environment
    max_y_position = 1.0  # Adjust based on your environment
    max_velocity = 10.0   # Adjust based on your environment
    max_gap_center = 1.0  # Adjust based on your environment

    state[0] = state[0] / max_x_position  # Normalize x position
    state[1] = state[1] / max_y_position  # Normalize y position
    state[2] = state[2] / max_velocity    # Normalize bird velocity
    state[3] = state[3] / max_gap_center  # Normalize gap center y position
    return state

# -------------------
# 7) Training Loop with Frame Skipping
# -------------------
epsilon = EPSILON_START
state, info = env.reset()
state = np.concatenate([state, [info.get("bird_velocity", 0), info.get("gap_center_y", 0)]])
state = normalize_state(state)
score = 0
episode = 1
max_vertical_distance = 1.0  # Adjust based on environment details

for frame_idx in range(MAX_FRAMES):
    # 1) Select action using epsilon-greedy policy
    action = epsilon_greedy_policy(state, epsilon)

    # 2) Initialize variables for frame skipping
    total_reward = 0
    for _ in range(FRAME_SKIP):  # Repeat the action for FRAME_SKIP frames
        next_state, reward, terminated, truncated, info = env.step(action)

        # Enhanced reward calculation
        vertical_distance = abs(next_state[1] - info.get("gap_center_y", 0))
        if terminated:
            reward = -1.0  # Strong penalty for crashing
        else:
            reward = 1.0 - (vertical_distance / max_vertical_distance)  # Reward proximity to gap center

        total_reward += reward  # Accumulate rewards

        if terminated or truncated:
            break  # Exit frame skipping if the episode ends

    # 3) Enhanced state representation
    next_state = np.concatenate([next_state, [info.get("bird_velocity", 0), info.get("gap_center_y", 0)]])
    next_state = normalize_state(next_state)

    # Store the transition in replay buffer
    replay_buffer.add((state, action, total_reward, next_state, terminated))

    # Update state
    if terminated or truncated:
        print(f"Episode {episode} ended with score: {score}")
        episode += 1
        state, info = env.reset()
        state = np.concatenate([state, [info.get("bird_velocity", 0), info.get("gap_center_y", 0)]])
        state = normalize_state(state)
        score = 0
    else:
        score += total_reward
        state = next_state

    # Perform a training step if replay buffer has enough samples
    if len(replay_buffer.buffer) > BATCH_SIZE:
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        # Q(s, a)
        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max Q(s', a') from target network
        next_q_values = target_net(next_states).max(dim=1)[0]

        # targets: r + gamma * max Q(s', a') * (1 - done)
        targets = rewards + GAMMA * next_q_values * (~dones)

        loss = nn.MSELoss()(q_values, targets.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Periodically update target network
    if frame_idx % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon
    epsilon = max(EPSILON_END, EPSILON_START - frame_idx / EPSILON_DECAY)

    # Debugging logs
    if frame_idx % 1000 == 0:
        print(f"Frame {frame_idx}, Episode {episode}, Score {score:.2f}, Epsilon {epsilon:.4f}")

# Clean up
env.close()

# Save the model
torch.save(policy_net.state_dict(), "flappy_bird_v3.pth")