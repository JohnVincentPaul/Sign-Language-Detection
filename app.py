from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the NEW Kaggle Alphabet Model
model_path = os.path.join('model_output', 'alphabet_model.pkl')
print(f"Loading Model from: {model_path}...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print("Model Loaded. Server Ready!")

def preprocess_landmarks(landmarks_list):
    """ Converts raw coordinates to Relative 2D Coordinates """
    if len(landmarks_list) == 0:
        return np.zeros(42)

    wrist = landmarks_list[0]
    wx, wy = wrist['x'], wrist['y']
    
    features = []
    for lm in landmarks_list:
        features.extend([lm['x'] - wx, lm['y'] - wy])
    
    return np.array(features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        landmarks = data.get('landmarks', [])
        handedness = data.get('handedness', [])
        
        # Prepare empty slots for Left and Right hands
        left_hand_features = np.zeros(42)
        right_hand_features = np.zeros(42)
        
        # Map detected hands to the correct slots
        for i in range(len(landmarks)):
            hand_label = handedness[i]['label'] # "Left" or "Right"
            features = preprocess_landmarks(landmarks[i])
            
            # Note: MediaPipe selfie-camera is mirrored. 
            # If the model predicts poorly, swap 'Left' and 'Right' here!
            if hand_label == 'Left':
                left_hand_features = features
            elif hand_label == 'Right':
                right_hand_features = features
                
        # Combine exactly as trained: Left first, then Right
        final_features = np.concatenate([left_hand_features, right_hand_features]).reshape(1, -1)
        
        # Predict
        prediction = model.predict(final_features)[0]
        confidence = np.max(model.predict_proba(final_features))
        
        # Filter low confidence
        if confidence < 0.4:
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