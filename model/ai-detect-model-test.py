import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import json

# Load the trained model
model = tf.keras.models.load_model('./ai_vs_real_images_model.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.

def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    return prediction[0][0]

def classify_image(image_path):
    confidence = predict_image(image_path)
    return confidence > 0.50

def process_directory(directory):
    results = {'total': 0, 'correct': 0}
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):  # Check file extension
            image_path = os.path.join(directory, filename)
            is_real = 'real' in directory
            is_classified_as_real = classify_image(image_path)
            results['total'] += 1
            if is_real == is_classified_as_real:
                results['correct'] += 1
    return results

def main():
    input_dirs = ['./input/real']
    results = {}

    for directory in input_dirs:
        dir_name = os.path.basename(directory)
        results[dir_name] = process_directory(directory)

    # Output results to a JSON file
    with open('classification_results.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

main()
