from flask import Flask, render_template, request
import torch
import os
import json
import urllib.request
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import requests
import io



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




# Load in all the classes and its respective dictionary
allClasses = []
classDict = {}
repository = "chickenwang6/quikdraw"
folder_path = "data/test"
url = f"https://api.github.com/repos/{repository}/contents/{folder_path}"
response = requests.get(url)
contents = response.json()
index = 0
for item in contents:
    allClasses.append(item['name'])
    classDict[item['name']] = index
    index += 1
# Load in the model baby
model_url = 'https://github.com/chickenwang6/quikdraw/raw/main/data/8_quikdraw_workflow_model_0.pth'
urllib.request.urlretrieve(model_url, 'model.pth')
model_0 = QuickDrawModelV0(input_shape=1, hidden_units=10, output_shape=len(allClasses))
model_0.load_state_dict(torch.load('model.pth'))
model_0.eval()


# Transform the image
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

# Function to make prediction
def make_predictions(model: torch.nn.Module, data, device: torch.device = "cpu"):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        # Prepare sample
        sample = torch.unsqueeze(data, dim=0).to("cpu") # Add an extra dimension and send sample to device

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

@app.route('/submit', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
       return render_template("index.html")
    elif request.method == 'POST':
        # Save the image
        image = request.files['file']
        image_path= "./images/" + image.filename
        image.save(image_path) 

        # Load the image
        byteImgIO = io.BytesIO()
        byteImg = Image.open(image_path)
        byteImg.save(byteImgIO, "PNG")
        byteImgIO.seek(0)
        byteImg = byteImgIO.read()

        dataBytesIO = io.BytesIO(byteImg)
        finalPass = Image.open(dataBytesIO)

        # Test the photo
        train_photo = data_transform(finalPass)
        pred_probs = make_predictions(model=model_0, data=train_photo)
        print(pred_probs)
        pred_classes = pred_probs.argmax(dim=1)

        print(allClasses[pred_classes])

        # Delete the file, we dont want it
        os.remove(image_path)

        # return render_template("index.html", prediction=allClasses[pred_classes])
        prediction = (allClasses[pred_classes])[0:-1]
        return json.dumps({"prediction": prediction})

if __name__ == '__main__':
    app.run(port=3000, debug=True)