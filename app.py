from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os
from PIL import Image

app = Flask(__name__)

# Try to load trained model
MODEL_PATH = "leaf_disease_model.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model Loaded Successfully!")
else:
    model = None
    print("⚠ No model found. Using random predictions.")


# Disease Database (same as your frontend)
diseaseData = {
    "Healthy": {
        "description": "The plant shows no signs of disease. Leaves are vibrant.",
        "remedies": "Maintain good watering and sunlight."
    },
    "Early Blight": {
        "description": "Dark circular spots with yellow halos.",
        "remedies": "Remove old leaves, spray fungicide."
    },
    "Late Blight": {
        "description": "Water-soaked brown/black patches.",
        "remedies": "Destroy infected plants and apply copper fungicide."
    },
    "Leaf Spot": {
        "description": "Small brown or black circular spots.",
        "remedies": "Use neem oil and prune infected leaves."
    },
    "Powdery Mildew": {
        "description": "White powder on leaf surface.",
        "remedies": "Increase airflow and use sulfur spray."
    },
    "Rust": {
        "description": "Orange or brown rust-like pustules.",
        "remedies": "Remove infected leaves and use fungicide."
    },
    "Scab": {
        "description": "Raised corky lesions on leaves.",
        "remedies": "Remove fallen leaves and spray fungicide."
    },
    "Bacterial Blight": {
        "description": "Water-soaked brown patches.",
        "remedies": "Use copper bactericide."
    },
    "Mosaic Virus": {
        "description": "Yellow/green mosaic pattern.",
        "remedies": "No cure. Remove infected plant."
    }
}

class_labels = list(diseaseData.keys())


# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Change based on your model
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")

    input_image = preprocess_image(image)

    if model:
        prediction = model.predict(input_image)[0]
        index = np.argmax(prediction)
        disease_name = class_labels[index]
    else:
        import random
        disease_name = random.choice(class_labels)

    return jsonify({
        "disease": disease_name,
        "description": diseaseData[disease_name]["description"],
        "remedies": diseaseData[disease_name]["remedies"]
    })


if __name__ == "__main__":
    app.run(debug=True)
