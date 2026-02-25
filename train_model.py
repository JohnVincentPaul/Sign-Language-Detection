import json
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DATASET_FILE = 'tourism_dataset_ISL.json'
OUTPUT_FOLDER = 'model_output'
MODEL_FILE = os.path.join(OUTPUT_FOLDER, 'gesture_model.pkl')

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def preprocess_landmarks(landmarks_list):
    """
    V2 LOGIC: Relative Coordinates (2D Only)
    We ignore Z to make it robust. We subtract Wrist to make it position-invariant.
    """
    # Wrist is at index 0
    if len(landmarks_list) == 0:
        return np.zeros(42)

    wrist = landmarks_list[0]
    wx, wy = wrist['x'], wrist['y']
    
    features = []
    for lm in landmarks_list:
        # Relative X, Relative Y (No Z)
        relative_x = lm['x'] - wx
        relative_y = lm['y'] - wy
        features.extend([relative_x, relative_y])
        
    return np.array(features)

print("--- STARTING TRAINING V2 (84 FEATURES) ---")

# 1. Load Data
with open(DATASET_FILE, 'r') as f:
    raw_data = json.load(f)

X = []
y = []

for sample in raw_data:
    # Process Hand 1
    hand1 = preprocess_landmarks(sample['landmarks'][0])
    
    # Process Hand 2
    if len(sample['landmarks']) > 1:
        hand2 = preprocess_landmarks(sample['landmarks'][1])
    else:
        hand2 = np.zeros(42) # 21 points * 2 coords = 42 zeros
    
    combined = np.concatenate([hand1, hand2])
    X.append(combined)
    y.append(sample['label'])

X = np.array(X)
y = np.array(y)

print(f"Feature Shape: {X.shape}") 
# CRITICAL CHECK: This MUST say (350, 84) or similar. 
# If it says 126, you are using the old code!

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_test, y_pred=clf.predict(X_test)) * 100:.2f}%")

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(clf, f)

print(f"✅ NEW Model saved to {MODEL_FILE}")