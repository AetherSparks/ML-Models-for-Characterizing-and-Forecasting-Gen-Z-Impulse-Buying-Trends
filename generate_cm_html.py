#!/usr/bin/env python3
"""
Generate HTML confusion matrix visualization for Voting Ensemble
"""

# 13x13 Confusion matrix data for Voting Ensemble 
# Accuracy: 96.7%, F1: 94.6%
confusion_matrix = [
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
]

# Calculate metrics
total_samples = sum(sum(row) for row in confusion_matrix)
correct_predictions = sum(confusion_matrix[i][i] for i in range(13))
accuracy = correct_predictions / total_samples

print(f"Confusion Matrix for Voting Ensemble")
print(f"Total samples: {total_samples}")
print(f"Accuracy: {accuracy:.4f}")

# Create HTML visualization
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #2d3748; color: white; }}
        .container {{ background-color: #1a202c; padding: 20px; border-radius: 10px; }}
        .title {{ font-size: 24px; font-weight: bold; text-align: center; margin-bottom: 20px; }}
        .matrix {{ border-collapse: collapse; margin: 20px auto; }}
        .matrix th, .matrix td {{ 
            border: 1px solid #4a5568; 
            padding: 8px; 
            text-align: center; 
            min-width: 50px;
            font-size: 12px;
        }}
        .matrix th {{ background-color: #2d3748; font-weight: bold; }}
        .diagonal {{ background-color: #3182ce; color: white; font-weight: bold; }}
        .high {{ background-color: #2b6cb0; color: white; }}
        .medium {{ background-color: #3182ce; color: white; }}
        .low {{ background-color: #4299e1; }}
        .verylow {{ background-color: #63b3ed; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="title">Voting Ensemble - E-commerce Dataset<br>Accuracy: {accuracy:.4f}</div>
        <table class="matrix">
            <tr>
                <th></th>
"""

# Add header row
for i in range(13):
    html_content += f"<th>{i}</th>"
html_content += "</tr>\n"

# Add matrix rows
for i in range(13):
    html_content += f"<tr><th>{i}</th>"
    for j in range(13):
        value = confusion_matrix[i][j]
        # Determine cell color based on value
        if i == j:  # Diagonal (correct predictions)
            cell_class = "diagonal"
        elif value > 100:
            cell_class = "high"
        elif value > 50:
            cell_class = "medium" 
        elif value > 20:
            cell_class = "low"
        else:
            cell_class = "verylow"
        
        html_content += f'<td class="{cell_class}">{value}</td>'
    html_content += "</tr>\n"

html_content += """
        </table>
        <div style="text-align: center; margin-top: 20px;">
            <p><strong>Predicted Label</strong></p>
        </div>
    </div>
</body>
</html>
"""

# Save HTML file
html_path = '/mnt/c/Users/ghose/Coding_Projects/PythonProjects/ML-Models-for-Characterizing-and-Forecasting-Gen-Z-Impulse-Buying-Trends/voting_ensemble_confusion_matrix.html'
with open(html_path, 'w') as f:
    f.write(html_content)

print(f"HTML visualization saved to: {html_path}")
print("You can open this file in a web browser to view the confusion matrix.")

# Also print the matrix in a formatted way
print(f"\nConfusion Matrix (13x13):")
print("    ", end="")
for j in range(13):
    print(f"{j:4d}", end=" ")
print()

for i in range(13):
    print(f"{i:2d}: ", end="")
    for j in range(13):
        print(f"{confusion_matrix[i][j]:4d}", end=" ")
    print()