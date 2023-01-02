import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ValueNetwork(nn.Module):
    """
    Value Network CNN

    Convolutions should be a good inductive bias
    for the spatial structure of ConnectFour
    """
    def __init__(self, input_shape):
        super(ValueNetwork, self).__init__()
        self.shape = input_shape
        activation = nn.ELU
        d = 8
        k1, k2  = 4, 3
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, d, k1),
            activation(),
            nn.Conv2d(d, 2*d, k2),
            activation(),
        )
        self.fc_1 = nn.Linear(4*d, 1)
        self.optimizer = optim.Adam(self.parameters())
        self.to(device)

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]

        # ensure the input is batched
        embed = self.convolutions(obs.reshape(-1, *img_shape))

        # verify that this works for singleton batch
        embed = torch.reshape(embed, (*batch_shape, -1))
        value = torch.tanh(self.fc_1(embed))
        return value

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
