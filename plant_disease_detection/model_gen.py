import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import cv2
# https://www.kaggle.com/datasets/emmarex/plantdisease

# -----------------------------
# 1. Load and Preprocess Data
# -----------------------------
def load_and_preprocess_data(data_dir="image_data", img_size=(128, 128)):
    images = []
    labels = []

    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(category)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    images = np.array(images, dtype="float32") / 255.0
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    print("Classes:", lb.classes_)
    return images, labels, lb



# -----------------------------
# 2. Split Data
# -----------------------------
def split_data(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42)


# -----------------------------
# 3. Build Model
# -----------------------------
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        
        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# -----------------------------
# 4. Train Model
# -----------------------------
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return model, history


# -----------------------------
# 5. Save Model
# -----------------------------
def save_model(model, save_path="saved_model/plant_disease_model.h5"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved at {save_path}")


# -----------------------------
# 6. Load Model
# -----------------------------
def load_model(save_path="saved_model/plant_disease_model.h5"):
    return keras.models.load_model(save_path)



# -----------------------------
# 7. Predict Function
# -----------------------------
def predict_disease(model, img_path, lb, img_size=(128,128)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = np.array(img, dtype="float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    return lb.classes_[class_idx], float(confidence)


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load dataset
    images, labels, lb = load_and_preprocess_data()

    # Split
    X_train, X_val, y_train, y_val = split_data(images, labels)

    # Build model
    model = build_model(input_shape=X_train.shape[1:], num_classes=labels.shape[1])

    # Train model
    model, history = train_model(model, X_train, y_train, X_val, y_val, epochs=10)

    # Save model
    save_model(model)

    # Example prediction
    example_img = "image_data/Potato___healthy/0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG"  # replace with real path
    if os.path.exists(example_img):
        disease, confidence = predict_disease(model, example_img, lb)
        print(f"Predicted: {disease} ({confidence*100:.2f}%)")
