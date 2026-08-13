import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
IMAGE_DATASET_CSV = "dataset_features.csv"      # Your A-Z, 0-9 data
CUSTOM_DATASET_JSON = "tourism_dataset_ISL.json" # Your Water, Police, Good data
OUTPUT_FOLDER = "model_output"
MODEL_FILE = os.path.join(OUTPUT_FOLDER, "master_static_model.pkl")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("1. Loading Image Dataset (A-Z, 0-9)...")
# FIX 1: Added low_memory=False to remove the DtypeWarning in the terminal
df_images = pd.read_csv(IMAGE_DATASET_CSV, low_memory=False)
print(f"   -> Loaded {len(df_images)} samples.")

print("\n2. Loading Custom Tourism Dataset (JSON)...")
with open(CUSTOM_DATASET_JSON, 'r') as f:
    raw_json = json.load(f)

custom_data_list = []

# Convert the JSON data into the exact same 84-column format
for sample in raw_json:
    label = sample['label']
    landmarks = sample['landmarks']
    handedness = sample.get('handedness', []) 
    
    left_raw, right_raw = None, None
    
    # Sort into Left and Right
    for i in range(len(landmarks)):
        hand_label = handedness[i]['label'] if i < len(handedness) else 'Right'
        if hand_label == 'Left':
            left_raw = landmarks[i]
        else:
            right_raw = landmarks[i]
            
    has_left = left_raw is not None
    has_right = right_raw is not None
    
    if not has_left and not has_right:
        continue
        
    # --- V3 ANCHOR MATH ---
    anchor = left_raw[0] if has_left else right_raw[0]
    ax, ay = anchor['x'], anchor['y']
    
    ref_hand = left_raw if has_left else right_raw
    dx = ref_hand[9]['x'] - ax
    dy = ref_hand[9]['y'] - ay
    scale = np.sqrt(dx**2 + dy**2) + 1e-6
    
    features = {'label': label}
    
    for i in range(21):
        if has_left:
            features[f'left_x_{i}'] = (left_raw[i]['x'] - ax) / scale
            features[f'left_y_{i}'] = (left_raw[i]['y'] - ay) / scale
        else:
            features[f'left_x_{i}'] = 0.0
            features[f'left_y_{i}'] = 0.0
            
    for i in range(21):
        if has_right:
            features[f'right_x_{i}'] = (right_raw[i]['x'] - ax) / scale
            features[f'right_y_{i}'] = (right_raw[i]['y'] - ay) / scale
        else:
            features[f'right_x_{i}'] = 0.0
            features[f'right_y_{i}'] = 0.0
            
    custom_data_list.append(features)

df_custom = pd.DataFrame(custom_data_list)
print(f"   -> Processed {len(df_custom)} custom gesture samples.")

print("\n3. Merging Datasets...")
df_master = pd.concat([df_images, df_custom], ignore_index=True)

# Force all labels to be strings so 0-9 don't clash with A-Z and custom gestures
df_master['label'] = df_master['label'].astype(str)
print(f"   -> Total Master Dataset Size: {len(df_master)} samples.")

# Prepare for training
y = df_master['label'].values
X = df_master.drop('label', axis=1).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n4. Training Master Random Forest Model...")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

print("\n5. Calculating Accuracy Metrics...")
# Check accuracy on the data it studied (Training Accuracy)
y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

# Check accuracy on new data it hasn't seen before (Testing Accuracy)
y_test_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print("-" * 50)
print(f"📊 STATIC MODEL ACCURACY:")
print(f"Training Accuracy: {(train_acc * 100):.2f}%")
print(f"Testing Accuracy:  {(test_acc * 100):.2f}%")
print("-" * 50)

# ==========================================
# FIX 2: Explicitly pass 'labels' to handle missing test classes
# ... [your existing prediction code] ...
test_acc = accuracy_score(y_test, y_test_pred)

# --- CALCULATE ERROR RATE ---
error_rate = (1.0 - test_acc) * 100

print(f"Overall Testing Accuracy: {(test_acc * 100):.2f}%")
print(f"📉 Error Rate: {error_rate:.2f}%")
