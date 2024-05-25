import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from io import BytesIO

model = YOLO('models/model-2/yolo200ep.pt')

st.title("Определение объектов с помощью YOLOv8")
st.write("Загрузите одно или несколько изображений или укажите их прямые URL-адреса.")

uploaded_files = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_urls = st.text_area("Или введите URL-адреса изображений (по одному в строке)...")

images = []

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        images.append((uploaded_file.name, image))

if image_urls:
    urls = image_urls.splitlines()
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            images.append((url, image))
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading image from URL {url}: {e}")

if images:
    for image_name, image in images:
        image_np = np.array(image)
        results = model.predict(source=image_np)
        result = results[0]
        rendered_image = result.plot()
        st.image(rendered_image, caption=f'Обработанное изображение: {image_name}', use_column_width=True)

st.title("Информация об обучении модели YOLOv8")
st.header("Информация о модели и обучении")
epochs = 200
batch_size = 32
imgsz = 640
st.write(f"**Количество эпох:** {epochs}")
st.write(f"**Размер партии:** {batch_size}")
st.write(f"**Размер изображения:** {imgsz}")

train_size = 2643
val_size = 247
st.write(f"**Кол-во образцов обучения:** {train_size}")
st.write(f"**Кол-во образцов валидации:** {val_size}")

st.header("Метрики модели")
map50 = 0.7210
map = 0.4300
st.write(f"**mAP 0.5:** {map50:.4f}")
st.write(f"**mAP 0.5:0.95:** {map:.4f}")

st.subheader("Precision-Recall Curve")
st.image('images/PR_curve.png')

st.subheader("Confusion Matrix")
st.image('images/confusion_matrix.png')

st.subheader("F1_curve")
st.image('images/F1_curve.png')

st.subheader("PR_curve")
st.image('images/PR_curve.png')

st.subheader("Results")
st.image('images/results.png')

st.subheader("P_curve")
st.image('images/P_curve.png')

st.subheader("R_curve")
st.image('images/R_curve.png')
