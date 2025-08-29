#!/usr/bin/env python3
import subprocess
import sys

def install_packages():
    packages = ['numpy', 'matplotlib', 'seaborn', 'scikit-learn']
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"Installed {package}")
        except Exception as e:
            print(f"Warning: {package} install failed: {e}")

# Install packages
install_packages()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

# 13x13 Confusion matrix data for Voting Ensemble (based on your requirements)
confusion_matrix = np.array([
    [9894, 43, 40, 33, 32, 28, 23, 23, 22, 19, 16, 16, 20],
    [33, 4743, 32, 27, 23, 20, 19, 15, 14, 15, 11, 12, 25],
    [21, 27, 4666, 25, 19, 19, 18, 13, 14, 10, 13, 10, 15],
    [20, 21, 26, 4553, 23, 19, 16, 15, 14, 11, 11, 9, 22],
    [11, 11, 14, 15, 4640, 14, 13, 11, 9, 9, 6, 9, 12],
    [9, 13, 11, 13, 18, 4922, 18, 14, 14, 10, 10, 7, 14],
    [6, 5, 8, 10, 11, 11, 4876, 12, 13, 9, 7, 8, 17],
    [12, 13, 12, 15, 16, 18, 25, 4099, 25, 19, 19, 16, 14],
    [9, 9, 9, 9, 10, 14, 15, 15, 3805, 15, 12, 11, 20],
    [7, 9, 11, 8, 12, 13, 16, 17, 21, 3081, 19, 18, 9],
    [4, 5, 9, 8, 9, 8, 11, 13, 13, 17, 2980, 14, 25],
    [4, 7, 4, 7, 4, 6, 9, 10, 8, 9, 12, 2664, 17],
    [8, 12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45, 4847]
])

# Class labels (0-12 for simplicity like in your image)
classes = list(range(13))

# Calculate actual metrics
total_samples = confusion_matrix.sum()
correct_predictions = np.diag(confusion_matrix).sum()
accuracy = correct_predictions / total_samples

print(f"Generated Accuracy: {accuracy:.4f}")

# Create the visualization to match your image style
plt.figure(figsize=(14, 12))

# Create heatmap similar to your image
ax = sns.heatmap(confusion_matrix, 
                 annot=True, 
                 fmt='d',
                 cmap='Blues',
                 xticklabels=classes,
                 yticklabels=classes,
                 cbar_kws={'label': 'Count'},
                 square=True,
                 linewidths=0.5)

# Set labels and title to match your image
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.title(f'Voting Ensemble - E-commerce Dataset\nAccuracy: {accuracy:.4f}', 
          fontsize=16, fontweight='bold', pad=20)

# Adjust tick labels
ax.set_xticklabels(classes, fontsize=12)
ax.set_yticklabels(classes, fontsize=12)

# Make it look more like your image
plt.tight_layout()

# Save the image
output_path = '/mnt/c/Users/ghose/Coding_Projects/PythonProjects/ML-Models-for-Characterizing-and-Forecasting-Gen-Z-Impulse-Buying-Trends/confusion_matrices/voting_ensemble_fake_13x13.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Confusion matrix image saved to: {output_path}")

plt.show()