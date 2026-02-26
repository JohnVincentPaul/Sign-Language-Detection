import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

CSV_FILE = 'isl_kaggle_dataset.csv'
OUTPUT_FOLDER = 'model_output'
MODEL_FILE = os.path.join(OUTPUT_FOLDER, 'alphabet_model.pkl')

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("Loading Kaggle CSV Dataset...")
df = pd.read_csv(CSV_FILE)

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
target_map = {i: alphabet[i] for i in range(26)}
df['label'] = df['target'].map(target_map)

X = []
y = []

print("Applying V3 Logic: Scale Normalization & Ignoring Handedness...")

for index, row in df.iterrows():
    # We will just grab whichever hand is NOT missing (zeros)
    # This makes the model handedness-agnostic.
    
    is_left_missing = pd.isna(row['left_hand_x_0']) or (row['left_hand_x_0'] == 0 and row['left_hand_y_0'] == 0)
    
    if not is_left_missing:
        wrist_x = row['left_hand_x_0']
        wrist_y = row['left_hand_y_0']
        prefix = 'left_hand'
    else:
        wrist_x = row['right_hand_x_0']
        wrist_y = row['right_hand_y_0']
        prefix = 'right_hand'
        
    # 1. Translate (Subtract Wrist)
    features = []
    for i in range(21):
        rel_x = row[f'{prefix}_x_{i}'] - wrist_x
        rel_y = row[f'{prefix}_y_{i}'] - wrist_y
        features.extend([rel_x, rel_y])
        
    # 2. Normalize (Fix the Camera Distance issue)
    # Find the maximum absolute value in the features to scale everything between -1 and 1
    max_val = np.max(np.abs(features))
    if max_val > 0:
        features = np.array(features) / max_val
    else:
        features = np.array(features)
        
    X.append(features)
    y.append(row['label'])

X = np.array(X)
y = np.array(y)

print(f"Feature Shape: {X.shape} (Must be exactly 42 features per hand)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Dataset Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(clf, f)

print(f"✅ V3 Normalized Model saved to: {MODEL_FILE}")