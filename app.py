from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
import io
import json

app = Flask(__name__)

# Load model and classes
model = load_model("best_resnet50_model.h5")
with open("class_names.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = 224

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        img_array = preprocess_image(file.read())
        preds = model.predict(img_array)[0]  # single sample

        # Get top 3 predictions
        top_idx = preds.argsort()[-3:][::-1]
        top_3 = []
        for i in top_idx:
            top_3.append({
                "class": class_names[i],
                "probability": round(float(preds[i]*100), 2)  # percentage
            })

        return jsonify({"top_3": top_3})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
