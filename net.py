import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 10)

    def forward(self, x):
        # x = x.view(-1, 784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.softmax(self.linear5(x), dim=1)
        return x


batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# train_dataset = MNIST(root=r'C:/Users/Administrator/Desktop/data/',
#                       train=True,
#                       download=True,
#                       transform=transform)

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

# test_dataset = MNIST(root=r'C:/Users/Administrator/Desktop/data/',
#                      train=False,
#                      download=True,
#                      transform=transform)

test_loader = DataLoader(test_dataset,
                         shuffle=True,
                         batch_size=batch_size)

model = Net()

print(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

loss_list = []
for epoch in range(10):

    for batch, (X, y) in enumerate(train_loader):

        y_pred = model(X)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 300 == 0:
            loss_list.append(loss.data.item())
            print("loss------------", loss.data.item())

plt.plot(np.linspace(0, 1000, len(loss_list)), loss_list)
plt.show()

rets = []
total = 0
correct = 0
with torch.no_grad():
    for data in test_loader:
        X, y = data
        y_pred = model(X)

        _, predicted = torch.max(y_pred.data, dim=1)

        total += y.size(0)
        correct += (predicted == y).sum().item()

print('accuracy on test set: %.2f %% ' % (100.0 * (correct / total)))


