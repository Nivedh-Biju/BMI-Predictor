import cv2
import dlib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load the Dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_facial_features(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    if not faces:
        # print("No face detected")
        return []

    # Process the first face detected
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = []

        # Extract the (x, y) coordinates of the 68 facial landmarks
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            cv2.circle(img, (x, y), 10, (255, 0, 0), -1)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Facial Landmarks")
        plt.axis('off')
        plt.show()

        return landmarks_points
    
features = extract_facial_features("./photos for presentation/Nuaim.jpg")
    
def calculate_all_features_combined(landmarks):
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def slope(p1, p2):
        return (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-6)  # Avoid division by zero

    def angle(p1, p2, p3):
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    # Feature dictionary
    features = dict({})

    # Basic Facial Distances
    features['jawline_width'] = distance(landmarks[0], landmarks[16])
    features['face_height'] = distance(landmarks[8], landmarks[27])
    features['cheekbone_width'] = distance(landmarks[1], landmarks[15])
    features['nose_width'] = distance(landmarks[31], landmarks[35])
    features['mouth_width'] = distance(landmarks[48], landmarks[54])

    # Eye Features
    features['eye_width_left'] = distance(landmarks[36], landmarks[39])
    features['eye_width_right'] = distance(landmarks[42], landmarks[45])
    features['eye_symmetry'] = abs(features['eye_width_left'] - features['eye_width_right'])

    # Ratios
    features['face_aspect_ratio'] = features['face_height'] / features['jawline_width']
    features['nose_to_face_height_ratio'] = distance(landmarks[27], landmarks[33]) / features['face_height']
    features['eye_to_face_width_ratio'] = (features['eye_width_left'] + features['eye_width_right']) / (2 * features['jawline_width'])

    # Angles
    features['jaw_angle'] = angle(landmarks[0], landmarks[8], landmarks[16])
    features['nose_to_cheekbone_angle'] = angle(landmarks[27], landmarks[31], landmarks[35])
    features['jawline_curvature'] = angle(landmarks[6], landmarks[8], landmarks[10])

    # Symmetry
    features['face_symmetry'] = sum(
        abs(distance(landmarks[i], landmarks[27]) - distance(landmarks[16 - i], landmarks[27]))
        for i in range(9)
    )

    # Mouth Features
    features['mouth_aspect_ratio'] = distance(landmarks[51], landmarks[57]) / features['mouth_width']

    # Jawline Features
    jawline_points = landmarks[0:17]
    features['jawline_length'] = sum(
        distance(jawline_points[i], jawline_points[i + 1]) for i in range(len(jawline_points) - 1)
    )
    features['jawline_ratio'] = features['jawline_width'] / features['jawline_length']

    # Eye Area
    features['left_eye_area'] = 0.5 * abs(
        landmarks[36][0] * landmarks[37][1] +
        landmarks[37][0] * landmarks[38][1] +
        landmarks[38][0] * landmarks[39][1] +
        landmarks[39][0] * landmarks[36][1] -
        (landmarks[37][0] * landmarks[36][1] +
         landmarks[38][0] * landmarks[37][1] +
         landmarks[39][0] * landmarks[38][1] +
         landmarks[36][0] * landmarks[39][1])
    )

    features['right_eye_area'] = 0.5 * abs(
        landmarks[42][0] * landmarks[43][1] +
        landmarks[43][0] * landmarks[44][1] +
        landmarks[44][0] * landmarks[45][1] +
        landmarks[45][0] * landmarks[42][1] -
        (landmarks[43][0] * landmarks[42][1] +
         landmarks[44][0] * landmarks[43][1] +
         landmarks[45][0] * landmarks[44][1] +
         landmarks[42][0] * landmarks[45][1])
    )

    # Eyebrow Features
    left_eye_center = midpoint(landmarks[37], landmarks[40])
    right_eye_center = midpoint(landmarks[43], landmarks[46])

    left_eyebrow_mid = midpoint(landmarks[19], landmarks[21])
    right_eyebrow_mid = midpoint(landmarks[22], landmarks[24])

    features['left_eyebrow_eye_distance'] = abs(left_eyebrow_mid[1] - left_eye_center[1])
    features['right_eyebrow_eye_distance'] = abs(right_eyebrow_mid[1] - right_eye_center[1])

    features['left_eyebrow_slope'] = slope(landmarks[17], landmarks[19])
    features['right_eyebrow_slope'] = slope(landmarks[22], landmarks[24])

    features['eyebrow_symmetry'] = abs(features['left_eyebrow_eye_distance'] - features['right_eyebrow_eye_distance'])

    # Nose Length
    features['nose_length'] = distance(landmarks[27], landmarks[33])

    # Nose-to-Mouth Ratio
    features['nose_to_mouth_ratio'] = features['nose_length'] / distance(landmarks[33], landmarks[51])

    return features

invalid=[]
noface=[]
missing_data=[]

def converttocsv(basepath):
    person_df=pd.read_csv('./illinois_doc_dataset/csv/person.csv',delimiter=";",index_col=0)
    total=0
    count_invalid=0
    count_noface=0
    count_missing_data=0
    csv=open('./temp.csv','w')
    for image in os.listdir(basepath):
        id=(image.replace('.jpg','')).upper()
        try:
            person=person_df.loc[id]
            img=os.path.join(basepath,image)
            features=extract_facial_features(img)
            if(features is None):
                # print("Invalid Image")
                count_invalid+=1
                invalid.append(id)
                continue
            elif(len(features)==0):
                count_noface+=1
                noface.append(id)
                continue
            landmarks=[(x,y) for x,y in features]
            all_features=calculate_all_features_combined(landmarks)
            
            if(total==0):
                for columns in all_features.keys():
                    csv.write(f",{columns}")
                csv.write(",BMI,Sex")
                csv.write("\n")
            
            csv.write(id)
            for feature in all_features.values():
                csv.write(f",{feature}")
            
            height_in_inches = person['height']
            weight_in_pounds = person['weight']
            sex = person['sex']

            # Calculate BMI
            if height_in_inches > 0:  # Prevent division by zero
                bmi = (weight_in_pounds * 703) / (height_in_inches ** 2)
            else:
                bmi = None
            csv.write(f",{bmi},{sex}\n")
            total+=1
        except KeyError:
            count_missing_data+=1
            missing_data.append(id)
            # print("Key not found")
            continue

    print(f"Count Total: {total}")
    print(f"Count Invalid: {count_invalid}")
    print(f"Count No Face: {count_noface}")
    print(f"Count Missing Data: {count_missing_data}")

# directory='./illinois_doc_dataset/front/'
# converttocsv(directory)

# Count Total: 59811
# Count Invalid: 969
# Count No Face: 330
# Count Missing Data: 7382
