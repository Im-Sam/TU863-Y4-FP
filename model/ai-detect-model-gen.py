import tensorflow as tf
import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image 
from matplotlib import pyplot as plt
Image.MAX_IMAGE_PIXELS = 1000000000 

# Step 1: Set up data generators for training and validation sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

'''
rudimentary imagedatagen parameters, modified with a mix of parameters found online.

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
'''

train_generator = train_datagen.flow_from_directory(
    './dataset/', 
    target_size=(64, 64),
    batch_size=64,
    class_mode='binary',
    classes=['ai_generated', 'real'],
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    './dataset/',
    target_size=(64, 64),
    batch_size=64,
    class_mode='binary',
    classes=['ai_generated', 'real'],
    subset='validation')

# Updated Model Architecture with Dropout
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Adding Dropout
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Adding Dropout
    layers.Dense(1, activation='sigmoid')
])

'''
this resulted in the high ai skew for all images, modifying with drop out.

# Step 2: Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
'''

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 3: Train the modelQ
history = model.fit(
    train_generator,
    steps_per_epoch=90, #260
    epochs=20, #30
    validation_data=validation_generator,
    validation_steps=25) #65

model.summary()

# Step 4: Plotting the training and validation loss and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Optionally, save the trained model
model.save('ai_vs_real_images_model.h5')
