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
        self.activation = nn.ELU()
        d = 12
        k1  = 4
        self.conv = nn.Conv2d(1, d, k1)
        self.fc_1 = nn.Linear(12*d, 128)
        self.fc_2 = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters())
        self.to(device)

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]

        # convert shape to (B, C, H, W)
        embed = self.conv(obs.reshape(-1, *img_shape))
        embed = self.activation(embed)
        embed = torch.reshape(embed, (*batch_shape, -1))
        embed = self.fc_1(embed)
        embed = self.activation(embed)
        value = self.fc_2(embed)
        return value

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
