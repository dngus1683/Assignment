import torch
import torch.nn as nn
import torchvision
import  torchvision.transforms as transforms

#---------------Convolutional neural network-----------------
#---------------two convolutional layers-----------------
class ConvNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,8,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc = nn.Linear(8*8*16,num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out

#---------------parameters-----------------
device = 'cpu'

num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001

#---------------cifar dataset-----------------
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                              train=True,
                                              download=True,
                                              transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=False,
                                             download = True,
                                             transform=transform)

#---------------Data Load-----------------
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=4,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=4,
                                          shuffle=False)

model = ConvNet(num_classes).to(device)

#---------------Loss & optimizer-----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#---------------Training-----------------
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#---------------Save-----------------
torch.save(model.state_dict(), ' model_cifar.ckpt')

#---------------Test-----------------
className = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', ' truck')
model.eval()

with torch.no_grad():
    correct = list(0. for i in range(10))
    total = list(0. for i in range(10))
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            total[label] += 1
            correct[label] += c[i].item()


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            className[i], 100 * correct[i] / total[i]))

