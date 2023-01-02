import numpy as np
import torch
import torch.nn.functional as F
from utils import ConnectFourGym
from config import ConnectFourConfig
from networks import ValueNetwork, device
from buffer import ReplayBuffer
from connectfour import minimax_move

def train(env, buffer, value_net, config, debug=False):
    if config.load_weights:
        print("Loading weights from file")
        value_net.load_checkpoint(config.save_path)

    num_episodes = 0
    state, done = env.reset(), False
    for step in range(1, config.train_steps):
        # check if we need to do stuff on this step
        if step % config.save_every:
            value_net.save_checkpoint(config.save_path)

        _, action = minimax_move(state, config.search_depth, value_net)
        next_state, reward, done, _ = env.step(action)
        obs, next_obs = [state.b], [next_state.b]
        buffer.store_transition(obs, action, reward, done)
        if debug:
            print(f"Step {step}")
            next_state.print_board()

        if done:
            loss = update_weights(buffer, value_net, config.search_depth)
            num_episodes += 1
            state, done = env.reset(), False
            buffer.clear()
            print(f"Episode {num_episodes} complete. Value loss: {loss:.3f}")
        else:
            state = next_state
    
def update_weights(buffer, v_net, N):
    """
    Update the value function using n-step targets

    Args:
        buffer - (ReplayBuffer) the replay buffer with episode info
        v_net - (ValueNetwork) the value network
        N - (int) the future rewards to consider when updating (i.e. n-step rewards)
    """
    states = torch.tensor(buffer.states, dtype=torch.float32).to(device)
    actions = torch.tensor(buffer.actions, dtype=torch.float32).to(device)
    rewards = torch.tensor(buffer.rewards, dtype=torch.float32).to(device)

    discount = 0.99 # put in config
    n_step_targets = torch.zeros(len(states), 1).to(device)
    n_step_targets[-1] = rewards[-1]

    # Note: in this environment, reward is only given on final timestep
    for i in range(1, N):
        idx = -1 - i
        reward = rewards[idx]
        n_step_targets[idx] = reward + discount * n_step_targets[idx + 1]
    target_states = states[N:]
    n_step_targets[:-N] = discount ** N * v_net(target_states).detach()

    v_net.optimizer.zero_grad()
    value_loss = F.mse_loss(n_step_targets, v_net(states))
    value_loss.backward()
    v_net.optimizer.step()

    return value_loss


if __name__ == "__main__":
    env = ConnectFourGym(6, 7)
    buffer = ReplayBuffer()
    value_net = ValueNetwork(env.observation_space.shape)
    config = ConnectFourConfig(
        train_steps=int(3e5),
        save_every=500,
        load_weights=False,
        target_update=100,
        search_depth=4,
        save_path="data/model.pth"
    )
    print(f"Initialized modules, beginning training")
    train(env, buffer, value_net, config)