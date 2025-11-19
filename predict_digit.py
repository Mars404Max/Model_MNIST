from PIL import Image, ImageOps
import PIL.ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Define Net class directly here since model.py is missing
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.fc(x)

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

img= Image.open("/Users/romanschulz/Desktop/TEST/my_digit.png").convert("L")
img =ImageOps.invert(img)
img = PIL.ImageOps.autocontrast(img)
img = ImageOps.crop(img, border=10)
img=ImageOps.pad(img,(28,28),method=Image.Resampling.LANCZOS,color=0,centering=(0.5,0.5))

transform=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])

img_tensor= transform(img).unsqueeze(0)
model=ConvNet()
model.load_state_dict(torch.load("fine_tuned_7.pth", map_location=torch.device("cpu")))
model.eval()

with torch.no_grad():
    output=model(img_tensor)
    pred=torch.argmax(output,dim=1)
    print(f"Predicted digit:{pred.item()}")