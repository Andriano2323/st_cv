import streamlit as st
from PIL import Image
import torch
import json
import sys
from pathlib import Path
import requests
from io import BytesIO
import time
import PIL
from PIL import ImageDraw
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision
from torchvision import transforms as T

st.write("# Локализация объектов")
st.write("Загрузите картинку для локализации")

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "models" / "model_1"))


class LocModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        # фризим слои, обучать их не будем (хотя технически можно)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # задай классификационный блок
        self.clf = nn.Sequential(
            nn.Linear(512*8*8, 128),
            nn.Sigmoid(),
            nn.Linear(128, 3)
        )

        # задай регрессионный блок
        self.box = nn.Sequential(
            nn.Linear(512*8*8, 128),
            nn.Sigmoid(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, img):
        # задай прямой проход
        resnet_out = self.feature_extractor(img)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)
        pred_classes = self.clf(resnet_out)
        pred_boxes = self.box(resnet_out)
        print(pred_classes.shape, pred_boxes.shape)
        return pred_classes, pred_boxes
    

preprocessing_func = T.Compose(
    [T.Resize((227, 227)),
     T.ToTensor()
     ]
)

def preprocess(img):
    return preprocessing_func(img)


# from model_localization import LocModel
# from preprocessing_localization import preprocess

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = LocModel()
    weights_path = project_root / 'models/model_1/weights_localization.pt'
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

id_class = json.load(open(project_root / 'models/model_1/id_class_localization.json'))
id_class = {int(k): v for k, v in id_class.items()}

def predict(image):
    img = preprocess(image)
    with torch.no_grad():
        start_time = time.time()
        preds = model(img.unsqueeze(0))
        end_time = time.time()
    pred_class = preds[0].argmax(dim=1).item()
    bbox_coords = preds[1].tolist()
    pred_bbox = (bbox_coords[0][0], bbox_coords[0][1], bbox_coords[0][2], bbox_coords[0][3])
    return id_class[pred_class], end_time - start_time, pred_bbox

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def load_image(image):
    if isinstance(image, BytesIO):
        return Image.open(image)
    else:
        return load_image_from_url(image)

def predict_images(images):
    predictions = []
    for img in images:
        image = load_image(img)
        prediction, inference_time, pred_bbox = predict(image)
        predictions.append((image, prediction, inference_time, pred_bbox))
    return predictions

def draw_bbox(image, bbox):
    width, height = image.size
    bbox_pixels = (int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height))
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_pixels, outline="red", width=3)
    return image

def display_results(predictions):
    for img, prediction, inference_time, pred_bbox in predictions:
        img_with_bbox = draw_bbox(img, pred_bbox)
        st.image(img_with_bbox)
        st.write(f'Prediction: {prediction}')
        st.write(f'Inference Time: {inference_time:.4f} seconds')
        st.write(f'Bbox: {pred_bbox}')

images = st.file_uploader('Upload file(s)', accept_multiple_files=True)

if not images:
    image_urls = st.text_area('Enter image URLs (one URL per line)', height=100).strip().split('\n')
    images = [url.strip() for url in image_urls if url.strip()]

if images:
    predictions = predict_images(images)
    display_results(predictions)

st.write("Использовалась предобученная модель - ResNet18 с заменой последних двух слоев")
st.write("Модель обучалась на предсказание 3 классов")
st.write("Размер train датасета - 148 картинок")
st.write("Размер valid датасета - 38 картинок")
st.write("Время обучения модели - 70 эпох = 18 минут, batch_size = 32")
st.image(str(project_root / 'images/photo_2024-05-23 17.16.53.jpeg'), width=900)
