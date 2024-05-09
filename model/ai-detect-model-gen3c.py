import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from matplotlib import pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Define image dimensions and batch size
IMG_WIDTH, IMG_HEIGHT = 64, 64
BATCH_SIZE = 32

# Data Augmentation setup with validation split
data_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Set validation split to 20%
)

# Training data generator
train_generator = data_gen.flow_from_directory(
    '../../dataset/',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['ai_generated', 'real'],
    subset='training'
)

# Validation data generator
validation_generator = data_gen.flow_from_directory(
    '../../dataset/',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['ai_generated', 'real'],
    subset='validation'
)

# Load ResNet50 model pre-trained on ImageNet without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

# Add custom layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the convolutional base to prevent training of pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks for early stopping and TensorBoard logging
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    #steps_per_epoch=train_generator.samples // train_generator.batch_size,
    steps_per_epoch=90,
    #validation_data=validation_generator,
    validation_data=validation_generator,
    #validation_steps=validation_generator.samples // validation_generator.batch_size,
    validation_steps=25,
    #epochs=50,
    epochs=10,
    callbacks=[early_stopping]
)

# Save the final model
model.save('ai_vs_real_images_model-gen3.h5')

# Plot training and validation accuracy and loss
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
