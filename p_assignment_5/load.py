import numpy as np
import torch
import torch.nn as nn
import gymnasium


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


def main():
    gymnasium.register(
        id='FlappyBird-v0',
        entry_point='flappy_bird_gymnasium:FlappyBirdEnv',
        kwargs={'use_lidar': True},
    )


    env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    num_actions = env.action_space.n
    input_size = env.observation_space.shape[0] + 2  # add extra info: bird_velocity and gap_center_y

    policy_net = QNetwork(input_size, num_actions).to("cpu")
    policy_net.load_state_dict(torch.load("flappy_bird_v3.pth", map_location="cpu"))
    policy_net.eval()

    episode_count = 0

    try:
        while True:
            episode_count += 1
            state, info = env.reset()
            # add extra info to state
            state = np.concatenate([state, [info.get("bird_velocity", 0), info.get("gap_center_y", 0)]])

            done = False
            total_score = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = policy_net(state_tensor).argmax(dim=1).item()

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # add extra info for next state as well
                next_state = np.concatenate([next_state, [info.get("bird_velocity", 0), info.get("gap_center_y", 0)]])

                total_score += reward
                state = next_state

            print(f"Episode {episode_count} ended with score: {total_score}")

    except KeyboardInterrupt:
        # pentru ctrl+c
        print("\nTesting interrupted by user.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
