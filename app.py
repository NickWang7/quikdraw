from flask import Flask, render_template, request
import torch
import glob
import os
from torch import nn
from torchvision import models
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import base64
from io import BytesIO

app = Flask(__name__, template_folder="templates", static_folder="staticFiles")

# PyTorch Neural Network
class QuickDrawModelV0(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2), # Default stride value is same as kernel size
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2), # Default stride value is same as kernel size
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*16*16,
                  out_features=output_shape)
    )


  def forward(self, x):
    return self.classifier(self.conv_block_2(self.conv_block_1(x)))

# Load in the model baby
newPath = Path("C:\\Users\\wangn6\\TkinterPractice\\8_quikdraw_workflow_model_0.pth")
numClasses = len(datasets.ImageFolder(root=Path("C:\\Users\\wangn6\\Pictures\\quikdraw\\test")).classes)
classDict = datasets.ImageFolder(root=Path("C:\\Users\\wangn6\\Pictures\\quikdraw\\test")).class_to_idx
model_0 = QuickDrawModelV0(input_shape=1, hidden_units=10, output_shape=numClasses)
model_0.load_state_dict(torch.load(f=newPath))
model_0.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
allClasses = datasets.ImageFolder(root=Path("C:\\Users\\wangn6\\Pictures\\quikdraw\\test")).classes


# Transform the image
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

# Function to make prediction
def make_predictions(model: torch.nn.Module, data, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        # Prepare sample
        sample = torch.unsqueeze(data, dim=0).to(device) # Add an extra dimension and send sample to device

        # Forward pass (model outputs raw logit)
        pred_logit = model(sample)

        # Get prediction probability (logit -> prediction probability)
        pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

        # Get pred_prob off GPU for further calculations
        pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

@app.route('/')
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile=request.files['imagefile']
    image_path= "./images/" + imagefile.filename
    imagefile.save(image_path) 
    image = Image.open(image_path)
    print("HERE IS IMAGE FILE: \n", image, "\n")
    train_photo = data_transform(image)
    pred_probs = make_predictions(model=model_0, data=train_photo)
    pred_classes = pred_probs.argmax(dim=1)

    # Delete the file, we dont want it
    os.remove(image_path)

    return render_template("index.html", prediction=allClasses[pred_classes])

if __name__ == '__main__':
    app.run(port=3000, debug=True)