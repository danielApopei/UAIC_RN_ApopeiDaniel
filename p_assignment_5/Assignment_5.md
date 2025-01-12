# Flappy Bird Deep Q-Learning Agent

## Architecture & Hyperparameters

**Q-Network:**
- Input: State vector combining environment observations, bird velocity, and gap center.
- Layers: 
  - Dense(128, ReLU) → Dense(64, ReLU) → Dense(num_actions)
- Two networks: 
  - Policy network for action selection.
  - Target network for stable Q-value targets.

**Hyperparameters:**
- Replay Memory Size: 100,000
- Batch Size: 64
- Gamma (γ): 0.99
- Epsilon (ε): Starts at 0.7, decays to 0.1 over 300,000 frames
- Learning Rate: 1e-4
- Target Update: Every 500 frames
- Max Frame: 600,000
- Frame Skip: 4

## Code Explanation

Architecture Explanation for the Flappy Bird Deep Q-Learning Agent

The architecture centers around a neural network designed to estimate the Q-value function 
𝑄
(
𝑠
,
𝑎
)
Q(s,a), which predicts the expected reward for taking a certain action given a particular state. Here's a breakdown of the key components:

Q-Network Structure
Input Layer:
The network receives a state vector that combines several pieces of information:

Raw observations from the environment (e.g., positions, LIDAR readings, mostly LIDAR).
Additional features such as the bird’s current velocity and the center position of the upcoming gap. This comprehensive input helps the network understand the current situation in the game.
Hidden Layers:
The network contains two fully connected hidden layers, each followed by a ReLU activation function:
Configuration of network:

[180+2] --> [128] --> [64] --> [2]

First Hidden Layer: 128 neurons.
Second Hidden Layer: 64 neurons.
The final layer is fully connected with a number of neurons equal to 2 (number of actions, jump or don't jump)
It outputs a Q-value for each action, which represents how good it is to take that action in the given state according to the current policy.
Dual-Network Setup: Policy and Target Networks

To improve learning stability, the setup uses two separate but identically structured networks:

Policy Network:
- Actively trained during the learning process.
- used to decide the best action to take in any given state based on the learned Q-values.
Target Network:
- Copy of the policy network that is updated less frequently.
- Provides target Q-values used to calculate the loss during training.
How It All Comes Together
Initialization:

Both the policy and target networks are created with the same architecture.
The target network is initially synchronized with the policy network.
Action Selection:

An epsilon-greedy strategy uses the policy network to select actions most of the time, occasionally choosing random actions to explore the environment.
Learning Process:
Agent interacts with the environment, it gathers experiences (state transitions, rewards, etc.) and stores them in the replay buffer.
During training, batches are sampled to update the policy network. The network adjusts its weights to minimize the difference between its Q-value predictions and the target Q-values.
The target Q-values are computed using the target network to ensure stability.
Periodically, the target network is updated to match the policy network, refreshing the basis for future target Q-value calculations.

## Experiments
We attempted training on multiple configurations of hyperparameters. 
* trained for 100 episodes
    * not learning enough, changed learning rate drastically
* trained for 500 episodes
    * tried changing the method of training between GameState / LIDAR / Pixels
* trained for 1000 episodes
    * changed reward based on how close bird is to gap
* trained for 24 hours / 39000 episodes
    * performance not improving, added normalization
* trained for 24 hours / 39000 episodes


## Performance
Score varies every episode:
```
Episode 1 ended with score: 1.9999999999999991
Episode 2 ended with score: -0.8999999999999986
Episode 3 ended with score: -0.8999999999999986
Episode 4 ended with score: 4.099999999999997
Episode 5 ended with score: -0.8999999999999986
Episode 6 ended with score: 3.799999999999997
Episode 7 ended with score: 3.899999999999996
Episode 8 ended with score: 3.5999999999999988
Episode 9 ended with score: 1.2000000000000046
```

At testing phase:
```
Episode 385 ended with score: 12.600000000000039
Episode 386 ended with score: -0.8999999999999986
Episode 387 ended with score: -0.8999999999999986
Episode 388 ended with score: 3.899999999999996
Episode 389 ended with score: 1.6999999999999993
Episode 390 ended with score: 1.300000000000002
Episode 391 ended with score: 10.899999999999967
```