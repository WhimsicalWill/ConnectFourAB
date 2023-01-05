import random
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from utils import ConnectFourGym
from config import ConnectFourConfig
from networks import ValueNetwork, device
from buffer import ReplayBuffer
from connectfour import minimax_move
from evaluations import eval2
from torch.utils.data import DataLoader, TensorDataset

def collect_data(env, v_net, num_steps=1000):
    board_states = []
    state, done = env.reset(), False
    for step in tqdm(range(num_steps), "Collecting data"):
        action = random.choice(state.get_valid_moves())
        next_state, reward, done, _ = env.step(action)
        obs = [state.b]
        board_states.append(obs)

        if done:
            state, done = env.reset(), False
        else:
            state = next_state

    eval_scores = [eval2(s[0]) for s in board_states]
    return board_states, eval_scores

def train(env, buffer, v_net, config, debug=False, num_epochs=200):
    if config.load_weights:
        print("Loading weights from file")
        v_net.load_checkpoint(config.save_path)

    training_data = collect_data(env, v_net, 200000)
    test_data = collect_data(env, v_net, 10000)

    train_states, train_scores = training_data
    train_states = torch.tensor(train_states, dtype=torch.float32).to(device)
    train_scores = torch.tensor(train_scores, dtype=torch.float32).unsqueeze(-1).to(device)

    test_states, test_scores = test_data
    test_states = torch.tensor(test_states, dtype=torch.float32).to(device)
    test_scores = torch.tensor(test_scores, dtype=torch.float32).unsqueeze(-1).to(device)

    train_dataset = TensorDataset(train_states, train_scores)
    test_dataset = TensorDataset(test_states, test_scores)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    training_step = 0
    for epoch in range(num_epochs):
        for states, scores in train_dataloader:
            training_step += 1
            if training_step % 100 == 0:
                valid_loss = benchmark_net(test_dataloader, v_net)
                wandb.log({"validation_loss": valid_loss}, training_step)
                print(f"Validation loss (step {training_step}): {valid_loss}")

            v_net.optimizer.zero_grad()
            value_loss = F.mse_loss(scores, v_net(states))
            value_loss.backward()
            v_net.optimizer.step()
            wandb.log({"value_loss": value_loss}, training_step)
        v_net.save_checkpoint(config.save_path)

def benchmark_net(test_dataloader, v_net):
    """
    Benchmark the performance against a held out set of 1000 game states
    """
    loss = 0
    for states, scores in test_dataloader:
        predictions = v_net(states)
        loss += F.mse_loss(predictions, scores)
    return loss / len(test_dataloader)


if __name__ == "__main__":
    wandb.login()
    env = ConnectFourGym(6, 7)
    buffer = ReplayBuffer()
    v_net = ValueNetwork(env.observation_space.shape)
    target_v_net = ValueNetwork(env.observation_space.shape)
    target_v_net.load_state_dict(v_net.state_dict())
    config = ConnectFourConfig(
        train_steps=int(3e5),
        log_interval=500,
        load_weights=False,
        target_update=100,
        search_depth=2,
        discount_factor = 0.99,
        tau = 0.001,
        save_path="data2/model.pth"
    )
    print(f"Initialized modules, beginning training")
    with wandb.init(project='Connect Four (value func for eval2)', config=config.__dict__):
        train(env, buffer, v_net, config, debug=False)