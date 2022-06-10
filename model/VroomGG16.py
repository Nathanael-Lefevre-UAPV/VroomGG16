import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from Utils.ColorPrinter import *
from Utils.Pushover import *


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        greenprint(self.pooling)
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        #cyanprint(out)
        blueprint(out.shape)
        redprint(out.shape, self.pooling)
        print(out.transpose(1, 3).shape)
        greenprint(7 * 7)
        out = self.pooling(out.transpose(1, 3))
        out = self.flatten(out)
        out = self.fc(out)
        return out

class VroomGG16(nn.Module):
    def __init__(self, train_loader, valid_loader, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2, device="cpu"):
        super(VroomGG16, self).__init__()
        self.device = device
        print("Using {} device".format(self.device))
        self.n_epochs = 4
        self.batch_size_train = 64
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10
        self.log_interval_test = 1

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.criterion = nn.NLLLoss() #nn.MSELoss()  # F.cross_entropy#nn.BCEWithLogitsLoss # F.cross_entropy # nn.MSELoss()

        '''
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier = self.vgg16.classifier[:-1]

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc0 = nn.Linear(16520, 320)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 5)

        self.fcA = nn.Linear(4096, 5)
        '''
        '''
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(7168, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 5)
        '''

        self.vgg16 = models.vgg16(pretrained=True)

        for param in self.vgg16.features.parameters():
            param.require_grad = False

        self.vgg16 = FeatureExtractor(self.vgg16)
        self.vgg16 = self.vgg16.to(device)
        print(self.vgg16)

        self.fc_out = nn.Linear(4096, 5)

        self.to(self.device)

        #self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        if train_loader is not None:
            self.train_loader = torch.utils.data.DataLoader(
                train_loader,
                batch_size=self.batch_size_train, shuffle=True)

        if valid_loader is not None:
            self.test_loader = torch.utils.data.DataLoader(
                valid_loader,
                batch_size=self.batch_size_train, shuffle=True)

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = []

        self.test_accuracy = []
        self.best_valid_accuracy = 0

    def forward(self, x):
        x = self.vgg16(x.transpose(1, 3))

        x = self.fc_out(x)

        '''
        x = x.transpose(1, 3).to(self.device)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)  # dropout was included to combat overfitting
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        '''
        '''
        x = self.vgg16(x)
        x = self.fcA(x)
        '''
        '''
        greenprint(x.shape)
        input()
        '''
        '''x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, 23600)
        x = nn.Flatten()(x)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)'''
        return F.softmax(x)

    def fit(self, epoch):

        self.train()
        print("Starting Training of VroomGG16 model")
        epoch_times = []
        # Start training loop
        for ep in range(1, epoch + 1):
            avg_loss = 0.
            counter = 0
            for batch_idx, (x, label) in enumerate(self.train_loader):
                counter += 1
                self.zero_grad()

                out = self(x.to(self.device))#.float())
                #print(out)
                #redprint(label)
                loss = self.criterion(out, torch.argmax(label, dim=1).to(self.device))

                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
                if counter % self.log_interval == 0:
                    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(ep, counter,
                                                                                               len(self.train_loader),
                                                                                               avg_loss / counter))
                    self.train_losses.append(loss.item())
                    self.train_counter.append(
                        (batch_idx * self.batch_size_train) + ((epoch - 1) * len(self.train_loader.dataset)))
            print("Epoch {}/{} Done, Total Loss: {}".format(ep, epoch, avg_loss / len(self.train_loader)))
            self.test()
            if self.test_accuracy[-1] > self.best_valid_accuracy:
                self.best_valid_accuracy = self.test_accuracy[-1]
                torch.save(self.state_dict(), os.path.dirname(__file__) + "/models/" + "Model.torch")
        print("Train finished")
        pushover("Train finished, acc: " + str(max(self.test_accuracy)))

    def test(self):
        self.test_counter = [i * len(self.test_loader.dataset) for i in range(self.n_epochs)]

        self.eval()
        test_loss = 0
        correct = 0
        correct2 = 0
        nb_pad = 0
        with torch.no_grad():
            for data, target in self.test_loader:

                output = self.forward(data.to(self.device))#.float())

                test_loss += self.criterion(output, torch.argmax(target, dim=1).to(self.device))

                correct += torch.sum(torch.where(torch.argmax(output.to(self.device), dim=1) == torch.argmax(target.data.to(self.device), dim=1), torch.tensor(True).to(self.device), torch.tensor(False).to(self.device))).cpu().detach()

        self.test_accuracy.append(100. * correct / (len(self.test_loader.dataset)))

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        blueprint('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}% - max: {:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), 100. * correct / (len(self.test_loader.dataset)), np.max(self.test_accuracy)))



