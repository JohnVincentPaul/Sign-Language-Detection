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

print("Loading Kaggle CSV Dataset...")
df = pd.read_csv(CSV_FILE)

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
target_map = {i: alphabet[i] for i in range(26)}
df['label'] = df['target'].map(target_map)

X = []
y = []

print("Applying V3 Logic: Global Anchor & Scale Invariance...")

for index, row in df.iterrows():
    has_left = not (pd.isna(row['left_hand_x_0']) or (row['left_hand_x_0'] == 0 and row['left_hand_y_0'] == 0))
    has_right = not (pd.isna(row['right_hand_x_0']) or (row['right_hand_x_0'] == 0 and row['right_hand_y_0'] == 0))
    
    if not has_left and not has_right:
        continue

    # 1. Find GLOBAL ANCHOR and SCALE
    if has_left:
        ax, ay = row['left_hand_x_0'], row['left_hand_y_0']
        # Distance from wrist (0) to middle finger base (9) to calculate hand size
        dx = row['left_hand_x_9'] - ax
        dy = row['left_hand_y_9'] - ay
        scale = np.sqrt(dx**2 + dy**2) + 1e-6
    else:
        ax, ay = row['right_hand_x_0'], row['right_hand_y_0']
        dx = row['right_hand_x_9'] - ax
        dy = row['right_hand_y_9'] - ay
        scale = np.sqrt(dx**2 + dy**2) + 1e-6

    features = []

    # 2. Process LEFT Hand (Relative to Global Anchor)
    for i in range(21):
        if has_left:
            rel_x = (row[f'left_hand_x_{i}'] - ax) / scale
            rel_y = (row[f'left_hand_y_{i}'] - ay) / scale
            features.extend([rel_x, rel_y])
        else:
            features.extend([0, 0]) # 42 zeros if no left hand

    # 3. Process RIGHT Hand (Relative to Global Anchor)
    for i in range(21):
        if has_right:
            rel_x = (row[f'right_hand_x_{i}'] - ax) / scale
            rel_y = (row[f'right_hand_y_{i}'] - ay) / scale
            features.extend([rel_x, rel_y])
        else:
            features.extend([0, 0]) # 42 zeros if no right hand

    X.append(features)
    y.append(row['label'])

X = np.array(X)
y = np.array(y)

print(f"Feature Shape: {X.shape} (Must be 84 columns)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Advanced ISL Model...")
clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(clf, f)
print(f"Model saved to: {MODEL_FILE}")