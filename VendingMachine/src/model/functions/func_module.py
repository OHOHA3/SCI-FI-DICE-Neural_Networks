import requests
import torchvision.models as models
from src.model import model_classes as mc
import torchvision.transforms as transforms
from PIL import Image
import torch
import cv2

CAMERA_IMAGE = "src/camera_pictures/camera_image.png"
MODEL_PATH = "src/model/model.pth"
SERVER_URL = "http://37.230.195.102:8080/neuronet/setPeople"


def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


def make_image(cap):
    for i in range(30):
        cap.read()
    ret, frame = cap.read()
    cv2.imwrite(CAMERA_IMAGE, frame)


def image_processing():
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    with Image.open(CAMERA_IMAGE) as img:
        tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    return tensor


def predict_class(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    predicted_class = mc.classes[predicted[0]]
    return predicted_class


def send_result(predicted_class):
    url = f"{SERVER_URL}?state{predicted_class}"
    with open(CAMERA_IMAGE, "rb") as picture:
        files = {"image": picture}
        requests.post(url, files=files)
