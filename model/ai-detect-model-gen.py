import tensorflow as tf
import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 1000000000 

# Step 1: Set up data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    './dataset/', 
    target_size=(64, 64),
    batch_size=64,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    './dataset/',
    target_size=(64, 64),
    batch_size=64,
    class_mode='binary',
    subset='validation')

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

# Optionally, save the trained model
model.save('ai_vs_real_images_model.h5')
