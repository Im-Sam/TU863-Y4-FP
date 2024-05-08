# Import necessary libraries
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import os
import pandas as pd

def load_model():
    # Load a pre-trained Faster R-CNN model with default weights
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()  # Set the model to evaluation mode
    return model

def detect_humans(image_path, model, confidence_threshold=0.9):
    # Open an image file
    image = Image.open(image_path)
    # Convert the image to a tensor and add a batch dimension
    image_tensor = F.to_tensor(image).unsqueeze(0)
    
    # Run the model on the image tensor to get predictions
    predictions = model(image_tensor)
    num_humans = 0  # Initialize the number of humans detected
    human_probs = []  # List to store probabilities of human detections
    
    # Iterate over the scores of the predictions
    for i, score in enumerate(predictions[0]['scores']):
        # Check if the score exceeds the confidence threshold and the label is 1 (human)
        if score.item() > confidence_threshold and predictions[0]['labels'][i] == 1:
            num_humans += 1  # Increment human count
            human_probs.append(score.item())  # Append probability to the list
    
    return num_humans, human_probs  # Return the number of humans and their probabilities

def ingest_images(directory, model):
    results = []  # List to store results for each image
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        # Check if file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)  # Get full path of the image
            # Detect humans in the image
            num_humans, human_probs = detect_humans(image_path, model)
            # Append the results to the list
            results.append({
                "Filename": filename,
                "Detected Humans": num_humans,
                "Probabilities": human_probs
            })
    return results  # Return the list of results

# Main execution starts here

# Load the pre-trained model
model = load_model()

# Specify the directory path containing the images
directory_path = 'in/'

# Process all images in the directory to detect humans
results = ingest_images(directory_path, model)

# Convert the list of results into a pandas table for easier viewing
results_df = pd.DataFrame(results)

# Print the DataFrame to the console
print(results_df.to_string(index=False))
