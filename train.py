import numpy as np
import random
import torch
import torch.nn.functional as F
import wandb
from utils import ConnectFourGym, update_target
from config import ConnectFourConfig
from networks import ValueNetwork, device
from buffer import ReplayBuffer
from connectfour import minimax_move

def train(env, buffer, v_net, target_v_net, config, debug=False):
    """
    Trains a value function through many episodes of self play

    Note: MAX plays on state.turn == 1 and MIN plays on state.turn == 2
    """
    if config.load_weights:
        print("Loading weights from file")
        v_net.load_checkpoint(config.save_path)

    num_episodes = 0
    state, done = env.reset(), False
    for step in range(1, config.train_steps):
        obs = [state.b] # add channel dimension for v_net
        turn = 1 if state.turn == 1 else -1
        _, action = minimax_move(state, config.search_depth, target_v_net)
        next_state, reward, done, _ = env.step(action)
        buffer.store_transition(obs, reward, turn)
        if debug:
            print(f"Step {step}")
            next_state.print_board()

        if done:
            if num_episodes % config.log_interval == 0:
                v_net.save_checkpoint(config.save_path)
                win_rate = benchmark_agent(target_v_net)
                wandb.log({"win_rate": win_rate}, num_episodes)
            
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
    turns = torch.tensor(buffer.turns, dtype=torch.float32).to(device)

    turns = turns[:, None, None, None] # number of dims must match
    transformed_states = turns * states
    n_step_targets = torch.zeros(len(states), 1).to(device)

    # the last n+1 steps are within range of the final reward
    discounts = discount ** torch.arange(N, -1, -1)
    n_step_targets[-N-1:] = discounts.unsqueeze(-1) * buffer.final_reward

    # all other steps use a bootstrapped target
    turns = turns.reshape(-1, 1)
    undiscounted_targets = turns[N:-1] * target_v_net(transformed_states[N:-1]).detach()
    n_step_targets[:-N-1] = discount ** N * undiscounted_targets

    v_net.optimizer.zero_grad()
    predictions = turns * v_net(transformed_states)
    value_loss = F.mse_loss(n_step_targets, predictions)
    value_loss.backward()
    v_net.optimizer.step()

    return value_loss

def benchmark_agent(target_v_net, num_episodes=25, seed_steps=4):
    """
    Benchmarks trained agent (p1) against fixed agent (p2) for num_episodes

    p1 uses a learned value function, p2 uses a fixed value function
    """
    p1_wins, p2_wins = 0, 0
    for ep in range(num_episodes):
        state, done = env.reset(), False
        for i in range(seed_steps):
            action = random.choice(state.get_valid_moves())
            next_state, reward, done, _ = env.step(action)
            state = next_state
        while not done:
            if state.turn == 1:
                _, action = minimax_move(state, config.search_depth, target_v_net)
            else:
                _, action = minimax_move(state, config.search_depth)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        if reward == -1:
            p2_wins += 1
        else:
            p1_wins += 1
    ties = num_episodes - (p1_wins + p2_wins)
    print(f"Results (trained wins, fixed wins, ties): {p1_wins, p2_wins, ties}")
    return p1_wins / num_episodes


if __name__ == "__main__":
    wandb.login()
    env = ConnectFourGym(6, 7)
    buffer = ReplayBuffer()
    v_net = ValueNetwork(env.observation_space.shape)
    target_v_net = ValueNetwork(env.observation_space.shape)
    target_v_net.load_state_dict(v_net.state_dict())
    config = ConnectFourConfig(
        train_steps=int(3e5),
        log_interval=100,
        load_weights=False,
        target_update=100,
        search_depth=2,
        discount_factor = 0.99,
        tau = 0.001,
        save_path="data/model.pth"
    )
    print(f"Initialized modules, beginning training")
    with wandb.init(project='Connect Four with learned value function', config=config.__dict__):
        train(env, buffer, v_net, target_v_net, config, debug=False)