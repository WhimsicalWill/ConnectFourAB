import numpy as np
import torch
import torch.nn.functional as F
import wandb
from utils import ConnectFourGym, update_target
from config import ConnectFourConfig
from networks import ValueNetwork, device
from buffer import ReplayBuffer
from connectfour import minimax_move

def train(env, buffer, v_net, target_v_net, config, debug=False):
    if config.load_weights:
        print("Loading weights from file")
        v_net.load_checkpoint(config.save_path)

    num_episodes = 0
    state, done = env.reset(), False
    for step in range(1, config.train_steps):
        # check if we need to do stuff on this step
        if step % config.save_every:
            v_net.save_checkpoint(config.save_path)

        _, action = minimax_move(state, config.search_depth, target_v_net)
        next_state, reward, done, _ = env.step(action)
        obs, next_obs = [state.b], [next_state.b]
        buffer.store_transition(obs, action, reward, done)
        if debug:
            print(f"Step {step}")
            next_state.print_board()

        if done:
            loss = update_weights(buffer, v_net, target_v_net, config)
            update_target(config.tau, v_net, target_v_net)
            num_episodes += 1
            state, done = env.reset(), False
            buffer.clear()

            print(f"Episode {num_episodes} complete. Value loss: {loss:.3f}")
            wandb.log({"value_loss": loss}, num_episodes)
        else:
            state = next_state
    
def update_weights(buffer, v_net, target_v_net, config):
    """
    Update the value function using n-step targets

    Args:
        buffer - (ReplayBuffer) the replay buffer with episode info
        v_net - (ValueNetwork) the value network
        target_v_net - (ValueNetwork) the target value network that provides stability
        config - (ConnectFourConfig) the config object
    Returns:
        loss - (float) the MSE loss of the minibatch
    """
    N, discount = config.search_depth, config.discount_factor
    states = torch.tensor(buffer.states, dtype=torch.float32).to(device)
    rewards = torch.tensor(buffer.rewards, dtype=torch.float32).to(device)

    n_step_targets = torch.zeros(len(states), 1).to(device)
    n_step_targets[-1] = rewards[-1]

    # Note: in this environment, reward is only given on final timestep
    for i in range(1, N):
        idx = -1 - i
        reward = rewards[idx]
        n_step_targets[idx] = reward + discount * n_step_targets[idx + 1]
    target_states = states[N:]
    n_step_targets[:-N] = discount ** N * target_v_net(target_states).detach()

    v_net.optimizer.zero_grad()
    value_loss = F.mse_loss(n_step_targets, v_net(states))
    value_loss.backward()
    v_net.optimizer.step()

    return value_loss


if __name__ == "__main__":
    wandb.login()
    env = ConnectFourGym(6, 7)
    buffer = ReplayBuffer()
    v_net = ValueNetwork(env.observation_space.shape)
    target_v_net = ValueNetwork(env.observation_space.shape)
    target_v_net.load_state_dict(v_net.state_dict())
    config = ConnectFourConfig(
        train_steps=int(3e5),
        save_every=500,
        load_weights=True,
        target_update=100,
        search_depth=2,
        discount_factor = 0.99,
        tau = 0.001,
        save_path="data/model.pth"
    )
    print(f"Initialized modules, beginning training")
    with wandb.init(project='Connect Four with learned value function', config=config.__dict__):
        train(env, buffer, v_net, target_v_net, config, debug=True)