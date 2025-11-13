from flask import Flask, request, jsonify, send_from_directory
import torch as t
import torch.nn as nn
import numpy as np
from flask_cors import CORS
import torchvision.transforms as transforms

app = Flask(__name__, static_folder="../Frontend/dist", static_url_path="")

CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})


import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),  
            nn.ReLU(),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )




model = AlexNet()  
model.load_state_dict(t.load("model.pth"))
model.eval()  


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", None)
    if features is None or len(features) != 784:
        return jsonify({"error": f"Expected {784} features, got {len(features) if features else 0}"}), 400
    
    x = np.array(features, dtype=np.float32)  
    x = x.reshape(1, 28, 28)                  
    
    x = t.tensor(x, dtype=t.float32).unsqueeze(0)  


    with t.no_grad():
        logits = model(x)  
        probs = nn.Softmax(dim=1)(logits) 
        pred_class = t.argmax(probs, dim=1).item()  
        pred_prob = probs[0, pred_class].item()   

    return jsonify({
        "probability": pred_prob,
        "prediction": pred_class,
    })


if __name__ == "__main__":
    app.run(debug=True)
