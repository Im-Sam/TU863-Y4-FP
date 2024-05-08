import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Set up data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dataset/', 
    target_size=(64, 64),  # Updated to downsize images to 64x64
    batch_size=64,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(64, 64),  # Updated to downsize images to 64x64
    batch_size=64,
    class_mode='binary',
    subset='validation')

# Step 2: Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # Updated input shape
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

# Step 3: Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=260, # Adjusted according to new calculations if needed
    epochs=30, # Set the number of epochs
    validation_data=validation_generator,
    validation_steps=65) # Adjusted according to new calculations if needed

model.summary()

# Optionally, save the trained model
model.save('ai_vs_real_images_model.h5')
