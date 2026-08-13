import json
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DYNAMIC_DATASET_JSON = "dynamic_dataset.json"
OUTPUT_FOLDER = "model_output"
MODEL_FILE = os.path.join(OUTPUT_FOLDER, "dynamic_model.h5")
LABEL_ENCODER_FILE = os.path.join(OUTPUT_FOLDER, "dynamic_labels.pkl")

SEQ_LEN = 30
FEATURE_DIM = 84  

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def frame_to_v3_features_84(landmarks):
    if not landmarks or len(landmarks) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    wrists = []
    for hand in landmarks:
        if not hand or len(hand) < 21: continue
        wrists.append((float(hand[0]["x"]), hand))

    if not wrists: return np.zeros(FEATURE_DIM, dtype=np.float32)

    wrists.sort(key=lambda t: t[0])
    left_raw = wrists[0][1]
    right_raw = wrists[-1][1] if len(wrists) > 1 else None

    has_left = left_raw is not None
    has_right = right_raw is not None

    anchor = left_raw[0] if has_left else right_raw[0]
    ax, ay = float(anchor["x"]), float(anchor["y"])

    ref_hand = left_raw if has_left else right_raw
    dx = float(ref_hand[9]["x"]) - ax
    dy = float(ref_hand[9]["y"]) - ay
    scale = float(np.sqrt(dx * dx + dy * dy) + 1e-6)

    # Prevent Division by Zero
    if scale < 0.02: scale = 0.02 

    frame_features = []

    for i in range(21):
        if has_left:
            frame_features.extend([
                (float(left_raw[i]["x"]) - ax) / scale,
                (float(left_raw[i]["y"]) - ay) / scale
            ])
        else: frame_features.extend([0.0, 0.0])

    for i in range(21):
        if has_right:
            frame_features.extend([
                (float(right_raw[i]["x"]) - ax) / scale,
                (float(right_raw[i]["y"]) - ay) / scale
            ])
        else: frame_features.extend([0.0, 0.0])

    # Clip Features to prevent exploding gradients
    features_array = np.array(frame_features, dtype=np.float32)
    return np.clip(features_array, -5.0, 5.0)

# ==========================================
# THE NORMALIZER FUNCTION
# ==========================================
def normalize_sequence_length(frames, target_len=SEQ_LEN):
    if len(frames) == 0:
        return None
    
    # Pad short sequences
    if len(frames) < target_len:
        padding_needed = target_len - len(frames)
        last = frames[-1]
        frames = list(frames) + [last] * padding_needed
        
    # Downsample long sequences evenly
    elif len(frames) > target_len:
        idx = np.linspace(0, len(frames) - 1, target_len).astype(int)
        frames = [frames[i] for i in idx]
        
    return frames


print("1. Loading dynamic sequence dataset...")
with open(DYNAMIC_DATASET_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

X_raw, y_raw = [], []

print("2. Normalizing frame counts & applying V3 Math...")
for sequence in raw_data:
    label = sequence.get("label")
    frames = sequence.get("frames", [])

    if label is None or len(frames) == 0: continue

    frames = normalize_sequence_length(frames, SEQ_LEN)
    if frames is None: continue

    sequence_features = [frame_to_v3_features_84(frame.get("landmarks", [])) for frame in frames]
    X_raw.append(sequence_features)
    y_raw.append(str(label).strip())

X = np.array(X_raw, dtype=np.float32)
y = np.array(y_raw)

print(f"   -> X shape safely normalized to: {X.shape}")

print("\n3. Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
num_classes = y_categorical.shape[1]

with open(LABEL_ENCODER_FILE, "wb") as f:
    pickle.dump(label_encoder, f)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)

print("\n4. Building Advanced LSTM...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, FEATURE_DIM)))
model.add(BatchNormalization()) 
model.add(LSTM(128, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax"))

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

print("\n5. Training...")
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=16, verbose=1)

print("\n6. Calculating Final Accuracy Metrics...")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("-" * 30)
print(f"📊 DYNAMIC MODEL (LSTM) ACCURACY:")
print(f"Training Accuracy: {(train_acc * 100):.2f}%")
print(f"Validation Accuracy: {(test_acc * 100):.2f}%")
print("-" * 30)

model.save(MODEL_FILE)
print(f"\n✅ Done. Model saved!")



# ... [your existing prediction code] ...
test_acc = accuracy_score(y_test_classes, y_pred_classes)

# --- CALCULATE ERROR RATE ---
error_rate = (1.0 - test_acc) * 100

print(f"Testing Accuracy: {(test_acc * 100):.2f}%")
print(f"📉 Error Rate: {error_rate:.2f}%")