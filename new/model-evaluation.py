import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import json

# Load the trained model
model = tf.keras.models.load_model('./ai_vs_real_image_model.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.

def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    return prediction[0][0]

def process_directory(directory, thresholds):
    category = 'real' if 'real' in directory else 'ai_generated'
    results = []
    summary = {threshold: {'total': 0, 'correct': 0} for threshold in thresholds}
    category_summary = {'total': 0, 'correct': 0}

    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            image_path = os.path.join(directory, filename)
            confidence = predict_image(image_path)
            # Determine verdict based on the best suitable threshold (for simplicity, using the first threshold in this example)
            verdict = 'real' if confidence > thresholds[0] else 'ai_generated'
            image_result = {
                'image_path': image_path,
                'confidence': float(confidence),
                'category': category,
                'verdict': verdict  # Add verdict based on the confidence
            }
            results.append(image_result)
            
            # Increment totals for each threshold
            category_summary['total'] += 1
            for threshold in thresholds:
                summary[threshold]['total'] += 1
                correct_classification = (category == 'real' and confidence > threshold) or (category == 'ai_generated' and confidence <= threshold)
                if correct_classification:
                    summary[threshold]['correct'] += 1
                    category_summary['correct'] += 1

    return results, summary, category_summary


def main():
    thresholds = [0.5]  # Define multiple confidence thresholds
    input_dirs = ['./input/ai_generated', './input/real']
    detailed_results = []
    overall_summary = {threshold: {'total': 0, 'correct': 0} for threshold in thresholds}
    category_summary = {'real': {'total': 0, 'correct': 0}, 'ai_generated': {'total': 0, 'correct': 0}}

    for directory in input_dirs:
        category = 'real' if 'real' in directory else 'ai_generated'
        dir_results, dir_summary, cat_summary = process_directory(directory, thresholds)
        detailed_results.extend(dir_results)
        for threshold in thresholds:
            overall_summary[threshold]['total'] += dir_summary[threshold]['total']
            overall_summary[threshold]['correct'] += dir_summary[threshold]['correct']
        category_summary[category]['total'] += cat_summary['total']
        category_summary[category]['correct'] += cat_summary['correct']

    # Save detailed results for each image
    with open('detailed_classification_results.json', 'w') as detail_outfile:
        json.dump(detailed_results, detail_outfile, indent=4)

    # Save a summary of results grouped by thresholds
    with open('summary_classification_by_threshold.json', 'w') as summary_outfile:
        json.dump(overall_summary, summary_outfile, indent=4)

    # Save category-wise summary
    with open('category_summary.json', 'w') as cat_summary_outfile:
        json.dump(category_summary, cat_summary_outfile, indent=4)

main()
