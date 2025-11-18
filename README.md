# Classification and Prediction Chili Leaf Disease

## 📋 Problem Statement
Many farmers and agricultural workers cannot identify chili leaf diseases, making it difficult to take appropriate action for treatment and prevention. Early detection is crucial to prevent crop loss and ensure healthy chili production.

## 🎯 Goals
Create an AI model that can automatically predict and classify chili leaf diseases from images, helping farmers identify plant health conditions quickly and accurately.

## 🔄 Process Overview
This project follows a complete machine learning pipeline:
1. **Data Loading** - Import and preprocess chili leaf images
2. **Image Segmentation** - HSV color space segmentation for disease detection
3. **Feature Extraction** - Extract 11 key features from images
4. **Model Training** - Train classification and regression models
5. **Evaluation** - Test model performance and accuracy
6. **Prediction** - Deploy model for disease prediction

## 📁 Dataset Structure
```
Your-Drive/
└── dataset/
    └── penyakit_cabai/
        ├── sehat/          # Healthy leaf images
        ├── yellow_light/   # Yellow leaf disease
        ├── kriting/        # Leaf curl disease
        └── leaf_spot/      # Leaf spot disease
```

**Dataset Requirements:**
- Image format: JPG/PNG
- Recommended: 50+ images per category
- Images will be resized to 128x128 pixels automatically

## 🛠️ Technology Stack
- **Platform**: Google Colab (GPU/CPU)
- **Language**: Python 3.x
- **Libraries**:
  - OpenCV (`cv2`) - Image processing
  - NumPy - Numerical operations
  - Pandas - Data manipulation
  - Matplotlib/Seaborn - Visualization
  - Scikit-learn - Machine Learning models
  - Pickle - Model serialization

## 🚀 Installation & Setup

### 1. Open Google Colab
Go to [Google Colab](https://colab.research.google.com/)

### 2. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Install Required Libraries
```python
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## 📊 Features Extracted

The model extracts **11 features** from each leaf image:

| No | Feature Name | Description |
|----|--------------|-------------|
| 1  | Green% | Percentage of green color (healthy leaf indicator) |
| 2  | Yellow% | Percentage of yellow color (disease indicator) |
| 3  | Brown% | Percentage of brown color (spot/damage indicator) |
| 4  | Num_Contours | Number of contours (shape complexity) |
| 5  | Edge_Density | Density of edges (texture roughness) |
| 6  | Dark_Area% | Percentage of dark areas (damage detection) |
| 7  | Mean_Hue | Average hue value |
| 8  | Mean_Saturation | Average saturation value |
| 9  | Mean_Value | Average brightness value |
| 10 | Texture_STD | Standard deviation of texture |
| 11 | Total_Leaf_Area% | Total leaf area percentage |

## 🤖 Models Used

### 1. Classification Models
- **Decision Tree Classifier**
  - Max depth: 5
  - Purpose: Classify leaf into 4 categories
  - Advantage: Easy to interpret

- **Random Forest Classifier** (Recommended)
  - N estimators: 100
  - Max depth: 10
  - Purpose: Classify leaf into 4 categories
  - Advantage: Higher accuracy, robust

### 2. Regression Model
- **Linear Regression**
  - Purpose: Predict disease severity score
  - Target: Yellow% + Brown% (Disease Score)

## 📈 Model Performance

### Classification Results
```
Model             | Accuracy
------------------|----------
Decision Tree     | ~XX%
Random Forest     | ~XX%
```

### Regression Results
```
Metric            | Value
------------------|----------
R² Score          | ~X.XX
RMSE              | ~XX.XX
MAE               | ~XX.XX
```

## 💾 Saving Models

Models are saved in Google Drive for future use:
```python
import pickle
import os

MODEL_DIR = "/content/drive/MyDrive/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Save Random Forest model
with open(os.path.join(MODEL_DIR, 'random_forest_model.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)

# Save metadata
metadata = {
    'categories': CATEGORIES,
    'feature_names': feature_names
}
with open(os.path.join(MODEL_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump(metadata, f)
```

## 🔮 Making Predictions

### Load Model
```python
import pickle

# Load model
with open('/content/drive/MyDrive/models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load metadata
with open('/content/drive/MyDrive/models/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    CATEGORIES = metadata['categories']
```

### Predict Single Image
```python
# Load and preprocess image
test_image = cv2.imread('path/to/your/image.jpg')
test_image = cv2.resize(test_image, (128, 128))

# Extract features
test_features = extract_features(test_image).reshape(1, -1)

# Predict
prediction = model.predict(test_features)[0]
result = CATEGORIES[prediction]

print(f"Prediction: {result}")
# Output: sehat / yellow_light / kriting / leaf_spot
```

## 📂 Project Structure
```
project/
├── notebooks/
│   └── chili_disease_detection.ipynb
├── models/
│   ├── random_forest_model.pkl
│   ├── decision_tree_model.pkl
│   ├── regression_model.pkl
│   └── metadata.pkl
├── dataset/
│   └── penyakit_cabai/
│       ├── sehat/
│       ├── yellow_light/
│       ├── kriting/
│       └── leaf_spot/
└── README.md
```

## 🎓 Usage Example

1. **Upload your dataset** to Google Drive following the structure above
2. **Open the notebook** in Google Colab
3. **Run all cells** sequentially
4. **View results** including:
   - Sample images from each category
   - HSV segmentation visualization
   - Feature distribution plots
   - Model accuracy and confusion matrix
   - Prediction on test images
5. **Save models** to Google Drive
6. **Use saved models** for future predictions

## 📊 Results Visualization

The project includes various visualizations:
- Sample images from each disease category
- HSV color space segmentation
- Threshold analysis (Binary, Otsu, Adaptive, Canny)
- Feature distribution histograms
- Confusion matrix heatmaps
- Regression prediction plots
- Feature importance charts

## 🔍 Disease Categories

| Category | Description | Characteristics |
|----------|-------------|-----------------|
| **Sehat** | Healthy leaf | High green%, low yellow/brown% |
| **Yellow_Light** | Yellowing disease | High yellow%, medium green% |
| **Kriting** | Leaf curl disease | High green%, complex contours |
| **Leaf_Spot** | Leaf spot disease | High brown%, high dark_area% |

## ⚠️ Important Notes

- Ensure dataset has balanced samples (similar amount per category)
- Image quality affects prediction accuracy
- Model performs best on clear, well-lit images
- Regularly retrain model with new data for better accuracy

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Add more disease categories
2. Improve feature extraction methods
3. Test different ML algorithms
4. Enhance visualization

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Abdul Rosyid

## 📧 Contact

For questions or suggestions, please contact: rosidabdul66@gmail.com

---

**Note**: This is an educational project for chili leaf disease detection using computer vision and machine learning techniques.
