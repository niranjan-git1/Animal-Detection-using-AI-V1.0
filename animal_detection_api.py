import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import numpy as np

app = Flask(__name__)
CORS(app)  # This allows your index.html to talk to the server

# 1. Configuration
MODEL_PATH = 'trained_animal_detector.keras'
CLASS_NAMES_PATH = 'animal_class_names.txt'
IMAGE_SIZE = (128, 128)

# 2. Encyclopedia Data (FEATURE 5)
ANIMAL_DATA = {
    "cat": {
        "description": "Domestic cats are small carnivorous mammals known for their agility and companionship.",
        "diet": "Carnivore",
        "lifespan": "12 - 18 Years",
        "habitat": "Domestic",
        "fun_fact": "Cats can make over 100 different sounds!"
    },
    "dog": {
        "description": "Dogs are domesticated mammals, often called 'man's best friend' for their loyalty.",
        "diet": "Omnivore",
        "lifespan": "10 - 13 Years",
        "habitat": "Global",
        "fun_fact": "A dog's sense of smell is 40 times stronger than a human's."
    },
    "elephant": {
        "description": "Elephants are the largest existing land animals, known for their trunks and tusks.",
        "diet": "Herbivore",
        "lifespan": "60 - 70 Years",
        "habitat": "Savannah/Forest",
        "fun_fact": "Elephants are the only mammals that cannot jump."
    },
    "horse": {
    "description": "The horse is a large, odd-toed ungulate mammal. Humans began domesticating horses around 3500 BC, and they have played a vital role in transport, agriculture, and warfare throughout history.",
    "diet": "Herbivore",
    "lifespan": "25 - 30 Years",
    "habitat": "Grasslands / Domestic",
    "fun_fact": "Horses can sleep both lying down and standing up thanks to a special locking mechanism in their legs."
    },
    "lion": {
        "description": "The lion is a large cat of the genus Panthera native to Africa and India. It has a muscular, deep-chested body, a short, rounded head, and a hairy tuft at the end of its tail.",
        "diet": "Carnivore",
        "lifespan": "10 - 14 Years",
        "habitat": "Savannah / Grasslands",
        "fun_fact": "A lion's roar can be heard from as far as 8 kilometers (5 miles) away!"
    } 
}

# Load Model and Classes
model = load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip().lower() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['file']
    img_path = "temp_image.jpg"
    file.save(img_path)

    # Preprocess image
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    score = np.max(predictions)
    class_idx = np.argmax(predictions)
    predicted_class = class_names[class_idx]

    # Get Fact Sheet Data
    # .get() is safer than using [] because it won't crash if the animal isn't in our dictionary
    animal_facts = ANIMAL_DATA.get(predicted_class, {
        "description": "No additional information found for this animal.",
        "diet": "Unknown",
        "lifespan": "Unknown",
        "habitat": "Unknown",
        "fun_fact": "N/A"
    })

    return jsonify({
        "success": True,
        "prediction": predicted_class,
        "confidence": float(score),
        "data": animal_facts  # This is what your JavaScript is looking for!
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)