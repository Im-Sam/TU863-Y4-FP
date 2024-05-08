import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('./ai_vs_real_images_model.h5')

#Load an image and preprocess it for model prediction.
def preprocess_image(image_path):
    # Load the image with the target size of 150x150 pixels
    img = image.load_img(image_path, target_size=(150, 150))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand the shape of the array for model prediction
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    # Scale the pixel values to be between 0 and 1
    return img_array_expanded_dims / 255.

#Make a prediction on a preprocessed image.
def predict_image(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    # Use the model to predict whether the image is real or AI-generated
    prediction = model.predict(processed_image)
    # Return the prediction confidence score
    return prediction[0][0]

#Classify an image as real or AI-generated with confidence.
def classify_image(image_path):
    # Get the prediction confidence score for the image
    confidence = predict_image(image_path)
    # Interpret the confidence score
    if confidence > 0.50:
        print(confidence)
        print(f"The image is real with {(confidence) * 100:.2f}% confidence.")
    else:
        print(confidence)
        print(f"The image is AI-generated with {100 - confidence * 100:.2f}% confidence.")

#dataset/ai_generated/ai_image5453.jpg
#dataset/real/real_image5453.jpg
image_path = 'in\known-ai-but-human.jpg'
classify_image(image_path)
