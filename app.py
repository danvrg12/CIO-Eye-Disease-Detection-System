from flask import Flask, render_template, request, jsonify
import io
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model('model/eye_disease_model.h5')

# Class names
CLASS_NAMES = ['Bulging_Eyes', 'Cataracts', 'Crossed_Eyes', 'Glaucoma', 'Uveitis']

# Serve main page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image_file')
    if not file:
        return jsonify({"error": "No image uploaded!"}), 400

    try:
        # Preprocess image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        pred_class = CLASS_NAMES[np.argmax(preds)]

        # Return JSON
        return jsonify({"prediction": pred_class})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
