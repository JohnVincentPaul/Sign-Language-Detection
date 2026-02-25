import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
CSV_FILE = 'isl_kaggle_dataset.csv'
OUTPUT_FOLDER = 'model_output'
MODEL_FILE = os.path.join(OUTPUT_FOLDER, 'alphabet_model.pkl')

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("Loading Kaggle CSV Dataset...")
df = pd.read_csv(CSV_FILE)

# 1. Map Targets 0-25 to Letters A-Z
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
target_map = {i: alphabet[i] for i in range(26)}
df['label'] = df['target'].map(target_map)

print(f"Total rows loaded: {len(df)}")

X = []
y = df['label'].values

# 2. Extract and Preprocess Landmarks
print("Converting to Relative 2D Coordinates (V2 Logic)...")

for index, row in df.iterrows():
    # --- PROCESS LEFT HAND ---
    left_features = []
    # Kaggle datasets often use 0.0 to represent a missing hand
    if pd.isna(row['left_hand_x_0']) or (row['left_hand_x_0'] == 0 and row['left_hand_y_0'] == 0):
        left_features = np.zeros(42) # Fill missing left hand with 42 zeros
    else:
        # Get Left Wrist
        wx_l = row['left_hand_x_0']
        wy_l = row['left_hand_y_0']
        for i in range(21):
            left_features.append(row[f'left_hand_x_{i}'] - wx_l)
            left_features.append(row[f'left_hand_y_{i}'] - wy_l)

    # --- PROCESS RIGHT HAND ---
    right_features = []
    if pd.isna(row['right_hand_x_0']) or (row['right_hand_x_0'] == 0 and row['right_hand_y_0'] == 0):
        right_features = np.zeros(42) # Fill missing right hand with 42 zeros
    else:
        # Get Right Wrist
        wx_r = row['right_hand_x_0']
        wy_r = row['right_hand_y_0']
        for i in range(21):
            right_features.append(row[f'right_hand_x_{i}'] - wx_r)
            right_features.append(row[f'right_hand_y_{i}'] - wy_r)

    # Combine Left (42) + Right (42) = 84 features
    combined = np.concatenate([left_features, right_features])
    X.append(combined)

X = np.array(X)

print(f"Feature Shape: {X.shape} (Must be N rows, 84 columns)")

# 3. Train the Model
print("Training Random Forest Classifier on A-Z...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1) # n_jobs=-1 uses all CPU cores to train faster
clf.fit(X_train, y_train)

# 4. Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- RESULTS ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(clf, f)

print(f"\n✅ Alphabet Model saved to: {MODEL_FILE}")