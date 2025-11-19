import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
print ("Script started")

#Hyperparameters
batch_size=64
learning_rate=0.0005
epochs =20

#Loading data
transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_data= datasets.MNIST(root= "./data", train = True,download=True,transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader= DataLoader(train_data, batch_size=batch_size,shuffle=True)
test_loader= DataLoader(test_data, batch_size=batch_size)

#Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv_layers=nn.Sequential(nn.Conv2d(1,32,kernel_size=3,padding=1),nn.ReLU(),nn.Conv2d(32,64,kernel_size=3,padding=1),nn.ReLU(),nn.MaxPool2d(2,2),nn.Dropout(0.25),nn.Conv2d(64,128,kernel_size=3,padding=1),nn.ReLU(),nn.MaxPool2d(2,2),nn.Dropout(0.25))
        self.fc_layers=nn.Sequential(nn.Flatten(),nn.Linear(128*7*7,512), nn.ReLU(),nn.Dropout(0.5),nn.Linear(512,10))
    def forward(self,x):
        x= self.conv_layers(x)
        x= self.fc_layers(x)
        return x
    
model = ConvNet()

#Loss function and optimizer
criterion= nn.CrossEntropyLoss()
#optimizer= optim.SGD(model.parameters(), lr=learning_rate)
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#Education
for epoch in range(epochs):
    for batch in train_loader:
        images, labels = batch

        outputs= model (images)
        loss= criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs},Loss: {loss.item():.4f}")

#Test
correct=0
total= 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images) 
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct/total:.2f}%")

torch.save(model.state_dict(),"mnist_model._v3.pth")
    
