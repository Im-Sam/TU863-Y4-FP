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
    predictions = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):  # Check file extension
            image_path = os.path.join(directory, filename)
            is_real = 'real' in directory
            confidence = predict_image(image_path)
            is_classified_as_real = confidence > 0.50
            results['total'] += 1
            if is_real == is_classified_as_real:
                results['correct'] += 1
            # Store the prediction details
            predictions.append({
                'file': filename,
                'directory': os.path.basename(directory),
                'confidence': float(confidence)  # Ensure the confidence is JSON serializable
            })
    # Append results and predictions to a JSON file
    with open('detailed_classification_results.json', 'w') as outfile:
        json.dump(predictions, outfile, indent=4)
    return results

def main():
    input_dirs = ['./input/real', './input/ai_generated']
    results = {}

    for directory in input_dirs:
        dir_name = os.path.basename(directory)
        results[dir_name] = process_directory(directory)

    # Optionally, also save a summary of results
    with open('summary_classification_results.json', 'w') as summary_outfile:
        json.dump(results, summary_outfile, indent=4)

main()
