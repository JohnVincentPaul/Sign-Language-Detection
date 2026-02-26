from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model_path = os.path.join('model_output', 'alphabet_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print("V3 Model Loaded!")

def preprocess_landmarks(landmarks_list):
    """ V3 Logic: Relative + Scale Normalized """
    if len(landmarks_list) == 0:
        return np.zeros(42)

    wrist = landmarks_list[0]
    wx, wy = wrist['x'], wrist['y']
    
    features = []
    # 1. Subtract Wrist
    for lm in landmarks_list:
        features.extend([lm['x'] - wx, lm['y'] - wy])
        
    # 2. Normalize Scale (Divide by max distance)
    max_val = np.max(np.abs(features))
    if max_val > 0:
        features = np.array(features) / max_val
    else:
        features = np.array(features)
        
    return features.reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        landmarks = data.get('landmarks', [])
        
        if len(landmarks) == 0:
            return jsonify({'gesture': 'None', 'confidence': 0.0})
            
        # We only process the PRIMARY (first) hand detected to match our new logic
        final_features = preprocess_landmarks(landmarks[0])
        
        prediction = model.predict(final_features)[0]
        confidence = np.max(model.predict_proba(final_features))
        
        if confidence < 0.3:
            prediction = "Unsure"
        
        return jsonify({
            'gesture': prediction,
            'confidence': float(confidence)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)