import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt

from Utils.ColorPrinter import *



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.n_epochs = 100
        self.batch_size_train = 32
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10

        self.flatten = nn.Flatten()

        '''
        f1 = lambda x: nn.Linear(105, 105)
        f2 = lambda x: nn.ReLU()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(105, 105),
            nn.ReLU(),
            *[f(x) for x in range(5) for f in (f1, f2)],
            nn.Linear(105, 3)
        )
        '''
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(105, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        # '''

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.to(self.device)

        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)


        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = []

        self.test_accuracy = []

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return F.softmax(logits, dim=1)

    def fit(self, dataset, epoch):

        self.train_loader = torch.utils.data.DataLoader(
            ProteinDataset(dataset),
            batch_size=self.batch_size_train, shuffle=True)

        self.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self(data)

            loss = F.cross_entropy(output, target)

            loss.backward()
            self.optimizer.step()

            # log
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append((batch_idx * self.batch_size_train) + ((epoch - 1) * len(self.train_loader.dataset)))
                #torch.save(self.network.state_dict(), 'results/model.pth')
                #torch.save(self.optimizer.state_dict(), 'results/optimizer.pth')

    def test(self, dataset: Datapack):
        self.test_loader = torch.utils.data.DataLoader(
            ProteinDataset(dataset),
            batch_size=self.batch_size_train, shuffle=True)
        self.test_counter = [i * len(self.test_loader.dataset) for i in range(self.n_epochs)]

        self.eval()
        test_loss = 0
        correct = 0
        correct2 = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self(data)
                #print(dataset.label_from_one_hot(output))
                test_loss += F.cross_entropy(output, target)
                pred = output.data.max(1, keepdim=True)[1]

                correct += pred.eq(torch.argmax(target.data, dim=1).view_as(pred)).sum()
                #correct += np.sum(np.where(pred == target.data))
            #greenprint(correct)

        self.test_accuracy.append(correct / len(self.test_loader.dataset))

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        blueprint('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))


    def plot_likelihood(self):
        fig = plt.figure()
        plt.plot(self.test_accuracy, color='blue')
        #plt.plot(self.train_counter, self.train_losses, color='blue')
        #cyanprint(len(self.test_counter), len(self.test_losses))
        #plt.scatter(self.test_counter, self.test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')
        plt.show()
