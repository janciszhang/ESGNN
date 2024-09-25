import torch
import torch.nn as nn
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net0 = nn.Linear(20, 10).to('cuda:0')
        self.relu = nn.ReLU()
        self.net1 = nn.Linear(10, 10).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net0(x.to('cuda:0')))
        return self.net1(x.to('cuda:1'))

import torch
import torch.nn as nn
import torch.optim as optim

model = DummyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(30, 10))
labels = torch.randn(30, 10).to('cuda:1')
loss_fn(outputs, labels).backward()
optimizer.step()