import matplotlib.pyplot as plt
import numpy as np

# --- The Data ---
models = [
    'Sequential CNN', 
    'SVM Classifier', 
    'ResNet50', 
    'EfficientNetB4\n+ Wiener', 
    'ResNet50 +\nEfficientNetB0', 
    'Proposed Edge AI\n(Our Model)'
]
accuracies = [82.0, 91.0, 95.0, 95.0, 99.0, 99.8]

# --- Setup the Graph ---
plt.figure(figsize=(10, 6))

# Create bars. We make your proposed model a special color (e.g., green) to stand out!
colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#2ca02c']
bars = plt.bar(models, accuracies, color=colors)

# --- Add the exact percentage text on top of each bar ---
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontweight='bold')

# --- Formatting ---
plt.ylim(0, 110) # Give room at the top for the text
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Comparison for Sign Language Recognition', fontsize=14, fontweight='bold')
plt.xticks(rotation=25, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# --- Save and Show ---
plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=300) # Saves a high-res image for your Word doc!
print("✅ Chart saved successfully as 'model_comparison_chart.png'")
plt.show()