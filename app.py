# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import numpy as np
# import os
# from collections import deque
# from tensorflow.keras.models import load_model

# app = Flask(__name__)
# CORS(app)

# # ==========================================
# # 1. LOAD MODELS
# # ==========================================
# print("Loading Models into Memory...")
# static_model_path = os.path.join('model_output', 'master_static_model.pkl')
# with open(static_model_path, 'rb') as f:
#     static_model = pickle.load(f)

# dynamic_model_path = os.path.join('model_output', 'dynamic_model.h5')
# dynamic_labels_path = os.path.join('model_output', 'dynamic_labels.pkl')

# dynamic_model = load_model(dynamic_model_path)
# with open(dynamic_labels_path, 'rb') as f:
#     dynamic_label_encoder = pickle.load(f)

# print("✅ Both Models Loaded. Server Ready!")

# # ==========================================
# # 2. THE ROLLING MEMORY BUFFER
# # ==========================================
# sequence_buffer = deque(maxlen=30) 

# # ==========================================
# # 3. PREDICTION ENDPOINT
# # ==========================================
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         landmarks = data.get('landmarks', [])

#         if not landmarks or len(landmarks) == 0:
#             sequence_buffer.clear()
#             return jsonify({'gesture': 'Waiting...', 'confidence': 0.0})

#         # --- Helper: Extract Features using V3 Math ---
#         def extract_v3_features(flip_x=False):
#             wrists = []
#             for hand in landmarks:
#                 if not hand or len(hand) < 21: continue
#                 x = hand[0]['x']
#                 if flip_x: x = 1.0 - x
#                 wrists.append((x, hand))

#             if not wrists: return None
#             wrists.sort(key=lambda t: t[0]) 
#             left_raw = wrists[0][1]
#             right_raw = wrists[-1][1] if len(wrists) > 1 else None

#             has_left, has_right = left_raw is not None, right_raw is not None
#             if not has_left and not has_right: return None

#             def get_xy(pt): return (1.0 - pt['x'], pt['y']) if flip_x else (pt['x'], pt['y'])

#             anchor = left_raw[0] if has_left else right_raw[0]
#             ax, ay = get_xy(anchor)
            
#             ref_hand = left_raw if has_left else right_raw
#             rx, ry = get_xy(ref_hand[9])
#             scale = np.sqrt((rx - ax)**2 + (ry - ay)**2) + 1e-6

#             feats = []
#             for i in range(21):
#                 if has_left:
#                     lx, ly = get_xy(left_raw[i])
#                     feats.extend([(lx - ax) / scale, (ly - ay) / scale])
#                 else: 
#                     feats.extend([0.0, 0.0])

#             for i in range(21):
#                 if has_right:
#                     rx_i, ry_i = get_xy(right_raw[i])
#                     feats.extend([(rx_i - ax) / scale, (ry_i - ay) / scale])
#                 else: 
#                     feats.extend([0.0, 0.0])

#             return np.array(feats, dtype=np.float32)

#         # THIS IS THE LINE THAT WAS MISSING!
#         features_1d = extract_v3_features(flip_x=False)
#         features_1d_flipped = extract_v3_features(flip_x=True)

#         if features_1d is None:
#             return jsonify({'gesture': 'Waiting...', 'confidence': 0.0})

#         sequence_buffer.append(features_1d)

#         best_pred = "Unsure"
#         best_conf = 0.0

#         # ==========================================
#         # 4. RUN STATIC BRAIN (Random Forest)
#         # ==========================================
#         feats_orig = features_1d.reshape(1, -1)
#         feats_flip = features_1d_flipped.reshape(1, -1)

#         # -- Get Probabilities for Normal Hand --
#         probs_o = static_model.predict_proba(feats_orig)[0]
        
#         # Sort to find the Top 3 guesses
#         top3_idx_o = np.argsort(probs_o)[-3:][::-1]
#         top3_labels_o = [static_model.classes_[i] for i in top3_idx_o]
#         top3_probs_o = [probs_o[i] for i in top3_idx_o]

#         # -- Get Probabilities for Mirrored Hand --
#         probs_f = static_model.predict_proba(feats_flip)[0]
#         top3_idx_f = np.argsort(probs_f)[-3:][::-1]
#         top3_probs_f = [probs_f[i] for i in top3_idx_f]

#         # Choose the best orientation (Normal vs Flipped)
#         if top3_probs_o[0] >= top3_probs_f[0]:
#             best_conf = float(top3_probs_o[0])
#             best_pred = top3_labels_o[0]
#             # Print the X-Ray Vision to Terminal!
#             print(f"STATIC TOP 3: 1.{top3_labels_o[0]}({top3_probs_o[0]*100:.1f}%) | 2.{top3_labels_o[1]}({top3_probs_o[1]*100:.1f}%) | 3.{top3_labels_o[2]}({top3_probs_o[2]*100:.1f}%)")
#         else:
#             best_conf = float(top3_probs_f[0])
#             best_pred = static_model.classes_[top3_idx_f[0]]
#             print(f"STATIC [FLIPPED] TOP 3: 1.{static_model.classes_[top3_idx_f[0]]}({top3_probs_f[0]*100:.1f}%) | 2.{static_model.classes_[top3_idx_f[1]]}({top3_probs_f[1]*100:.1f}%)")

#         # ==========================================
#         # 5. MOVEMENT GATEKEEPER & DYNAMIC BRAIN
#         # ==========================================
#         if len(sequence_buffer) == 30:
#             buffer_array = np.array(sequence_buffer)
            
#             # Calculate how much the coordinates changed over 30 frames
#             movement_score = np.std(buffer_array, axis=0).mean()
            
#             print(f"Static: {best_pred} ({best_conf*100:.1f}%) | Movement Score: {movement_score:.4f}")

#             # Only run LSTM if hand is actually moving AND static isn't 98% confident
#             if movement_score > 0.015 and best_conf < 0.98:
                
#                 lstm_input = buffer_array.reshape(1, 30, 84)
#                 lstm_probs = dynamic_model.predict(lstm_input, verbose=0)[0] 
#                 lstm_idx = np.argmax(lstm_probs)
#                 lstm_conf = float(lstm_probs[lstm_idx])
#                 lstm_pred = dynamic_label_encoder.inverse_transform([lstm_idx])[0]

#                 print(f"   -> LSTM Fired! Guessed: {lstm_pred} ({lstm_conf*100:.1f}%)")

#                 if lstm_conf > 0.60:
#                     best_conf = lstm_conf
#                     best_pred = lstm_pred
#                     sequence_buffer.clear() # Cooldown
#             else:
#                 if movement_score <= 0.015:
#                     print("   -> LSTM Skipped (Hand is too still)")
#                 else:
#                     print("   -> LSTM Skipped (Static model is highly confident)")

#         # ==========================================
#         # 6. FINAL QUALITY GATE
#         # ==========================================
#         if best_conf < 0.35:
#             best_pred = "Unsure"

#         return jsonify({
#             'gesture': str(best_pred),
#             'confidence': float(best_conf)
#         })

#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from collections import deque
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. LOAD MODELS
# ==========================================
print("Loading Models into Memory...")
static_model_path = os.path.join('model_output', 'master_static_model.pkl')
with open(static_model_path, 'rb') as f:
    static_model = pickle.load(f)

dynamic_model_path = os.path.join('model_output', 'dynamic_model.h5')
dynamic_labels_path = os.path.join('model_output', 'dynamic_labels.pkl')

dynamic_model = load_model(dynamic_model_path)
with open(dynamic_labels_path, 'rb') as f:
    dynamic_label_encoder = pickle.load(f)

print("✅ Both Models Loaded. Server Ready!")

# ==========================================
# 2. THE ROLLING MEMORY BUFFER
# ==========================================
# This automatically caps at 30, so we never have <30 or >30 frames!
sequence_buffer = deque(maxlen=30) 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        landmarks = data.get('landmarks', [])

        if not landmarks or len(landmarks) == 0:
            sequence_buffer.clear()
            return jsonify({'gesture': 'Waiting...', 'confidence': 0.0})

        # --- Extract Features with V3 Math ---
        def extract_v3_features(flip_x=False):
            wrists = []
            for hand in landmarks:
                if not hand or len(hand) < 21: continue
                x = hand[0]['x']
                if flip_x: x = 1.0 - x
                wrists.append((x, hand))

            if not wrists: return None
            wrists.sort(key=lambda t: t[0]) 
            left_raw = wrists[0][1]
            right_raw = wrists[-1][1] if len(wrists) > 1 else None

            has_left, has_right = left_raw is not None, right_raw is not None
            if not has_left and not has_right: return None

            def get_xy(pt): return (1.0 - pt['x'], pt['y']) if flip_x else (pt['x'], pt['y'])

            anchor = left_raw[0] if has_left else right_raw[0]
            ax, ay = get_xy(anchor)
            
            ref_hand = left_raw if has_left else right_raw
            rx, ry = get_xy(ref_hand[9])
            scale = np.sqrt((rx - ax)**2 + (ry - ay)**2) + 1e-6
            
            if scale < 0.02: scale = 0.02

            feats = []
            for i in range(21):
                if has_left:
                    lx, ly = get_xy(left_raw[i])
                    feats.extend([(lx - ax) / scale, (ly - ay) / scale])
                else: feats.extend([0.0, 0.0])

            for i in range(21):
                if has_right:
                    rx_i, ry_i = get_xy(right_raw[i])
                    feats.extend([(rx_i - ax) / scale, (ry_i - ay) / scale])
                else: feats.extend([0.0, 0.0])

            return np.clip(np.array(feats, dtype=np.float32), -5.0, 5.0)

        features_1d = extract_v3_features(flip_x=False)
        features_1d_flipped = extract_v3_features(flip_x=True)

        if features_1d is None:
            return jsonify({'gesture': 'Waiting...', 'confidence': 0.0})

        sequence_buffer.append(features_1d)
        best_pred, best_conf = "Unsure", 0.0

        # ==========================================
        # 3. RUN STATIC BRAIN
        # ==========================================
        feats_orig = features_1d.reshape(1, -1)
        feats_flip = features_1d_flipped.reshape(1, -1)

        probs_o = static_model.predict_proba(feats_orig)[0]
        idx_o = np.argmax(probs_o)
        if probs_o[idx_o] > best_conf:
            best_conf, best_pred = float(probs_o[idx_o]), static_model.classes_[idx_o]

        probs_f = static_model.predict_proba(feats_flip)[0]
        idx_f = np.argmax(probs_f)
        if probs_f[idx_f] > best_conf:
            best_conf, best_pred = float(probs_f[idx_f]), static_model.classes_[idx_f]

        # ==========================================
        # 4. MOVEMENT GATEKEEPER & DYNAMIC BRAIN
        # ==========================================
        if len(sequence_buffer) == 30:
            buffer_array = np.array(sequence_buffer)
            
            # Math Trick: Calculate physical hand movement variance
            movement_score = np.std(buffer_array, axis=0).mean()
            
            # Print X-Ray Diagnostics to terminal
            print(f"Static: {best_pred} ({best_conf*100:.1f}%) | Movement: {movement_score:.4f}")

            # Only wake up the LSTM if the hand is moving and static isn't completely locked
            if movement_score > 0.015 and best_conf < 0.98:
                
                lstm_input = buffer_array.reshape(1, 30, 84)
                lstm_probs = dynamic_model.predict(lstm_input, verbose=0)[0] 
                lstm_idx = np.argmax(lstm_probs)
                lstm_conf = float(lstm_probs[lstm_idx])
                lstm_pred = dynamic_label_encoder.inverse_transform([lstm_idx])[0]

                print(f"   >>> LSTM OVERRIDE: {lstm_pred} ({lstm_conf*100:.1f}%)")

                if lstm_conf > 0.60:
                    best_conf = lstm_conf
                    best_pred = lstm_pred
                    sequence_buffer.clear() # Cooldown
            else:
                if movement_score <= 0.015:
                    print("   -> LSTM Asleep (Hand is still)")
                else:
                    print("   -> LSTM Asleep (Static model locked it out)")

        if best_conf < 0.35: best_pred = "Unsure"

        return jsonify({'gesture': str(best_pred), 'confidence': float(best_conf)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)