# BMI Predictor Using Front Images of Faces

This project implements a BMI predictor by extracting features from front-facing images of faces. The extracted features are then used to train a predictive model.

## Feature Extraction

The feature extraction process leverages the `dlib` library for facial landmark detection. Key steps include:

1. **Loading Pre-trained Models**
   - A face detector and a 68-point facial landmark predictor are used to extract facial features.

2. **Feature Calculation**
   - Various features are calculated based on facial landmarks:
     - Distances (e.g., jawline width, face height, nose width)
     - Ratios (e.g., face aspect ratio, nose-to-face height ratio)
     - Angles (e.g., jaw angle, nose-to-cheekbone angle)
     - Symmetry metrics
     - Eye and mouth features

3. **Visualization**
   - Detected landmarks are overlaid on the input image for visualization.

### Code Snippet
```python
# Load the Dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to extract facial features
def extract_facial_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if faces:
        landmarks = predictor(gray, faces[0])
        landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
        return landmarks_points
    return None
```

## Dataset Conversion to CSV

The extracted features are saved to a CSV file along with BMI and other metadata.

### Steps:
- Load image metadata (e.g., height, weight, sex) from a reference CSV.
- Extract features for each image.
- Calculate BMI using height and weight.
- Save features and BMI to a new CSV file.

### Code Snippet
```python
# Convert dataset to CSV
def converttocsv(basepath):
    for image in os.listdir(basepath):
        features = extract_facial_features(os.path.join(basepath, image))
        if features:
            calculated_features = calculate_all_features_combined(features)
            # Save to CSV with additional metadata like BMI
```

## BMI Prediction

A Random Forest Regressor is used to predict BMI based on extracted features.

### Steps:
1. Split the dataset into training and testing sets.
2. Train the model using features as input and BMI as the target variable.
3. Evaluate the model using metrics such as:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R² Score
   - Pearson Correlation

### Code Snippet
```python
# Train and evaluate the model
model = RandomForestRegressor(random_state=42, n_estimators=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

## Sample Output
- **Best Model Parameters**: `{ 'random_state': 42, 'n_estimators': 200 }`
- **R² Score**: `0.89`
- **Evaluation Metrics**:
  - MAE: `2.34`
  - MSE: `1.76`
  - Pearson Correlation: `0.95`

## Dependencies

- Python Libraries:
  - OpenCV
  - Dlib
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib

## Usage

1. Place the input images in a directory.
2. Run the feature extraction script to generate the CSV file.
3. Train the predictive model using the generated dataset.
4. Evaluate the model's performance on a test set.

## Notes

- Ensure `shape_predictor_68_face_landmarks.dat` is available in the working directory.
- Handle invalid images, missing faces, or incomplete data gracefully.

---

This project demonstrates the integration of computer vision and machine learning for health-related predictions. Feel free to contribute or suggest improvements!
