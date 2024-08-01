import torch
import torch.nn as nn
import torch.nn.functional as F


class Z_conv(nn.Module):
    def __init__(self, HP):
        super().__init__()
        self.conv1 = nn.Conv2d(HP['channels'], 16, 2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        self.classifier = nn.Identity()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 4)
        x = F.max_pool2d(F.relu(self.conv3(x)), 6)

        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class T(nn.Module):
    def __init__(self, HP):
        super(T, self).__init__()
        self.HP = HP
        self.const1 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, 1)
        x_in = x
        x = self.const1(torch.ones_like(x_in)) + x_in
        return x


if __name__ == '__main__':
    pass
