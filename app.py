from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import json
from PIL import Image
import io

app = Flask(__name__)

print("Loading...")
model = load_model('food_model.h5')

with open('labels.json', 'r') as f:
    labels = json.load(f)
    labels = {int(k): v for k, v in labels.items()}

with open('nutrients.json', 'r') as f:
    nutrients_db = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'})

    file = request.files['file']
    
    with open("debug_image.jpg", "wb") as f:
        f.write(file.read())
    file.seek(0)

    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_arr = image.img_to_array(img)
    
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr) 

    pred = model.predict(img_arr)
    class_id = np.argmax(pred)
    confidence = float(np.max(pred))
    food_name = labels.get(class_id, "Unknown")
    
    print(f"Predicted: {food_name} with {confidence:.2f} confidence")

    info = nutrients_db.get(food_name, {"calories": 0, "msg": "No data"})
    
    return jsonify({
        'food': food_name, 
        'nutrients': info,
        'confidence': f"{confidence*100:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)