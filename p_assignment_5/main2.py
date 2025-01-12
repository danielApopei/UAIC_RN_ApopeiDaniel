import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import flappy_bird_gymnasium
import gymnasium
import os

LOAD_MODEL = True
MODEL_PATH = "flappy_bird_v3.pth"


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


REPLAY_MEMORY_SIZE = 100_000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 0.7
EPSILON_END = 0.1  # Maintain some exploration
EPSILON_DECAY = 300_000  # Slower decay for extended exploration
TARGET_UPDATE = 500
LEARNING_RATE = 1e-4
MAX_FRAMES = 600_000
FRAME_SKIP = 4  # Number of frames to skip after taking an action


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


env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
num_actions = env.action_space.n
input_size = env.observation_space.shape[0] + 2  # LIDAR + bird velocity + gap center
print(env.observation_space.shape[0])

policy_net = QNetwork(input_size, num_actions).to("cpu")
target_net = QNetwork(input_size, num_actions).to("cpu")

if LOAD_MODEL:
    if os.path.exists(MODEL_PATH):
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        target_net.load_state_dict(policy_net.state_dict())
        print("Loaded pre-trained model.")
    else:
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
else:
    target_net.load_state_dict(policy_net.state_dict())

target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)


def epsilon_greedy_policy(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cpu")
    with torch.no_grad():
        q_values = policy_net(state_t)
        return q_values.argmax(dim=1).item()


def normalize_state(state):
    max_x_position = 1.0
    max_y_position = 1.0
    max_velocity = 10.0
    max_gap_center = 1.0

    state[0] = state[0] / max_x_position
    state[1] = state[1] / max_y_position
    state[2] = state[2] / max_velocity
    state[3] = state[3] / max_gap_center
    return state


epsilon = EPSILON_START
state, info = env.reset()
state = np.concatenate([state, [info.get("bird_velocity", 0), info.get("gap_center_y", 0)]])
state = normalize_state(state)
score = 0
episode = 1
max_vertical_distance = 1.0

for frame_idx in range(MAX_FRAMES):
    action = epsilon_greedy_policy(state, epsilon)

    total_reward = 0
    for _ in range(FRAME_SKIP):
        next_state, reward, terminated, truncated, info = env.step(action)

        vertical_distance = abs(next_state[1] - info.get("gap_center_y", 0))
        if terminated:
            reward = -1.0
        else:
            reward = 1.0 - (vertical_distance / max_vertical_distance)  # reward proximity to gap center

        total_reward += reward

        if terminated or truncated:
            break

    # add more stuff to state
    next_state = np.concatenate([next_state, [info.get("bird_velocity", 0), info.get("gap_center_y", 0)]])
    next_state = normalize_state(next_state)

    # store in replay buffer
    replay_buffer.add((state, action, total_reward, next_state, terminated))

    # update state
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

    # perform a training step if replay buffer has enough samples
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

    # update target rarely
    if frame_idx % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_END, EPSILON_START - frame_idx / EPSILON_DECAY)

    if frame_idx % 1000 == 0:
        print(f"Frame {frame_idx}, Episode {episode}, Score {score:.2f}, Epsilon {epsilon:.4f}")

env.close()
torch.save(policy_net.state_dict(), "flappy_bird_v3.pth")