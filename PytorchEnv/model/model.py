import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch

# COMPARABILE CON LA VECCHIA RETE DI SCKITLEARN
class Med2021(BaseModel):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, output_size)
        self.dropout_1 = nn.Dropout(p=0.075)
        self.dropout_2 = nn.Dropout(p=0.075)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout_1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_2(x)
        x = self.fc3(x)
        return x


class Med2020(BaseModel):
    def __init__(self, input_size, output_size):
        super(BaseModel, self).__init__()

        self.l1 = nn.Linear(input_size, 250)
        self.l2 = nn.Linear(250, 250)
        self.l3 = nn.Linear(250, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        out = F.relu(out)
        out = self.l3(out)
        return out
