import torch
from torch import nn
from torch import optim
import NeutralNetwork
import Data
import Script

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = NeutralNetwork.NeutralNetwork()

criterion = nn.BCELoss()
criterion = criterion.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)

data = Data.csvToData("Salary_Data.csv")
training = data.change()

training = training.to(device)

for epoch in range(training.shape[0]):
    y = net(training)
    y = torch.squeeze(y)
    test_loss = criterion(y)
    if epoch % 1 == 0:

        print(
            f'''epoch {epoch}
    Train set - loss: {test_loss}
    ''')
    optimizer.zero_grad()
    test_loss.backward()
    optimizer.step()