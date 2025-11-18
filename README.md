#🌿 Leaf Disease Detection using Machine Learning / Deep Learning

This project focuses on automatically detecting plant leaf diseases using machine learning/deep learning. The system takes an image of a leaf, processes it, and predicts whether the leaf is healthy or diseased along with the specific disease class.

🚀 Features

Detects multiple plant leaf diseases

Image preprocessing using OpenCV

Model built using ML/CNN/Transfer Learning

High accuracy with fast predictions

Easy to deploy locally or on the web

Supports custom dataset training

🧪 Supported Diseases (example)

You can modify this based on your dataset:

Healthy

Bacterial Blight

Leaf Spot

Rust

Powdery Mildew

Early / Late Blight<img width="960" height="1032" alt="Screenshot 2025-11-18 230212" src="https://github.com/user-attachments/assets/c57b711d-4637-49e2-86dd-b26135053cd9" />

🛠️ Technologies Used

Python

OpenCV

TensorFlow / Keras

Scikit-Learn

NumPy / Pandas

Matplotlib / Seaborn

📊 Workflow
1. Dataset Collection

You can use:

PlantVillage dataset

Custom field images

Kaggle datasets

2. Preprocessing (OpenCV)

Resize image

Remove noise

Convert to RGB

Apply thresholding / segmentation

Normalize pixels

3. Model Training

Options:

CNN from scratch

Transfer learning (MobileNet, VGG16, ResNet50)

Machine-learning SVM/RandomForest on extracted features

4. Prediction

The model outputs:

Disease label

Confidence score

5. Visualization

Bounding box (optional), probability bar, final classification.
