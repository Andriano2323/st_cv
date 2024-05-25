# YOLOv8 Model Training and Inference with Streamlit

This project demonstrates how to train a YOLOv8 model for object detection and perform inference using a Streamlit web application. Additionally, it integrates TensorBoard for logging and visualizing training metrics.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Running the Streamlit App](#running-the-streamlit-app)
- [TensorBoard Integration](#tensorboard-integration)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides an end-to-end solution for training a YOLOv8 model on a custom dataset and performing inference using a Streamlit application. The project also includes integration with TensorBoard to log and visualize training metrics such as mAP, PR curves, and confusion matrices.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/st_cv.git
    cd st_cv
    ```

2. Set up a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Install additional dependencies for YOLOv8, Streamlit, and TensorBoard:
    ```bash
    pip install ultralytics streamlit tensorflow tensorboard pillow numpy requests matplotlib
    ```

## Usage

### Project Structure

st_cv/
│
├── models/
│ ├── model-2/
│ │ └── yolo200ep.pt # Trained YOLOv8 model
│
├── data/
│ └── wind_turbine_dataset/ # Custom dataset for training
│ └── data.yaml
│
├── logs/ # Directory to store TensorBoard logs
│
├── pages/
│ └── detection.py # Streamlit app for inference and metrics visualization
│
├── requirements.txt
├── README.md
└── train.py # Training script with TensorBoard logging


### Training the Model

1. Modify the `train.py` script to suit your dataset and training configuration:
    ```python
    from ultralytics import YOLO
    import tensorflow as tf

    # Load a model
    model = YOLO('yolov8n.yaml')

    # TensorBoard Summary Writer
    log_dir = "./logs"
    summary_writer = tf.summary.create_file_writer(log_dir)

    class TensorBoardLogger:
        def __init__(self, writer):
            self.writer = writer

        def log_metrics(self, metrics, step):
            with self.writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(key, value, step=step)
                self.writer.flush()

    logger = TensorBoardLogger(summary_writer)

    # Train the model and log metrics
    epochs = 200
    for epoch in range(epochs):
        results = model.train(data='./data/wind_turbine_dataset/data.yaml', epochs=1, imgsz=640, batch=32)
        metrics = {
            'map50': results['metrics']['map50'],
            'map': results['metrics']['map'],
        }
        logger.log_metrics(metrics, step=epoch)
    ```

2. Run the training script:
    ```bash
    python train.py
    ```

### Running the Streamlit App

1. Modify the `pages/detection.py` script to include the correct paths to your model and logs:
    ```python
    import streamlit as st
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np
    import requests
    from io import BytesIO
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing import event_accumulator

    model = YOLO('/home/a/ds-phase-2/09-cv/cv_project/models/model-2/yolo200ep.pt')

    st.title("YOLOv8 Inference with Streamlit")
    # ... (rest of the script)
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run pages/detection.py
    ```

### TensorBoard Integration

To visualize the training metrics using TensorBoard:

1. Run TensorBoard pointing to the logs directory:
    ```bash
    tensorboard --logdir=./logs
    ```

2. Open the URL provided by TensorBoard (usually `http://localhost:6006`) in your web browser to see the logged metrics.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
