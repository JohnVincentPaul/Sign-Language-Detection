import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import type_of_target

CSV_FILE = "dataset_features.csv"
OUTPUT_FOLDER = "model_output"
MODEL_FILE = os.path.join(OUTPUT_FOLDER, "alphabet_model.pkl")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("Loading Extracted Features...")
# low_memory=False avoids mixed-type chunk guessing
df = pd.read_csv(CSV_FILE, low_memory=False)

print("Raw dtypes:")
print(df.dtypes)

# Ensure label is treated as a categorical/string class label
df["label"] = df["label"].astype(str).str.strip()

# Convert all feature columns to numeric, coerce errors to NaN
feature_cols = [c for c in df.columns if c != "label"]
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with any NaNs (either in features or label)
before = len(df)
df = df.dropna(subset=feature_cols + ["label"])
after = len(df)
print(f"Dropped {before - after} rows with invalid/missing values. Remaining: {after}")

# Build X and y
y = df["label"].values
X = df[feature_cols].values

print(f"Total Samples: {len(X)}")
print(f"Features per Sample: {X.shape[1]} (Should be {len(feature_cols)})")
print("Unique labels:", sorted(pd.unique(y)))
print("type_of_target(y):", type_of_target(y))

# Now y should be a proper classification target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest classifier...")
clf = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy on test data: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Save the model
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(clf, f)

print(f"\nModel successfully saved to {MODEL_FILE}")