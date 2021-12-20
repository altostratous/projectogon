import torch.nn
import torch.utils.data
import torchvision.datasets
import torch.nn.functional as F
from torch import nn


# Build the neural network, expand on top of nn.Module
class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=28 * 28, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        self.relu = nn.ReLU()

    # define forward function
    def forward(self, t):
        t = self.relu(self.fc1(t.view(-1, 28 * 28)))
        # output
        t = self.fc2(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t


model = NN().cuda()
criterion = torch.nn.CrossEntropyLoss()
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])

dataset = torchvision.datasets.MNIST('data', download=True, transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
optimizer = torch.optim.Adam(model.parameters())
data_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True,)

for epoch in range(1000):
    total_loss = 0
    total = 0
    correct = 0
    for x, y in data_loader:
        x, y = x.cuda(), y.cuda()
        output = model(x)
        correct += int(torch.sum(torch.topk(output, 1).indices.reshape(y.shape) == y))
        total += y.nelement()
        loss = criterion(output, y)
        total_loss += float(loss)
        loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(correct / total, total_loss)

    total = 0
    correct = 0
    for x, y in test_data_loader:
        x, y = x.cuda(), y.cuda()
        output = model(x)
        correct += int(torch.sum(torch.topk(output, 1).indices.reshape(y.shape) == y))
        total += y.nelement()
    print(correct / total)
