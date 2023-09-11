# Import necessary TensorFlow and Keras modules
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

# Load the MNIST dataset and split it into training and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
train_images, test_images = train_images / 255, test_images / 255

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(train_images.reshape(-1, 28, 28, 1))

# Define the neural network model using a Sequential API
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks to save the best model and early stopping
checkpoint = ModelCheckpoint('the_model.h5', save_best_only=True)
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Train the model using data augmentation and callbacks
history = model.fit(datagen.flow(train_images.reshape(-1, 28, 28, 1), train_labels, batch_size=64),
                    epochs=30,
                    validation_data=(test_images.reshape(-1, 28, 28, 1), test_labels),
                    callbacks=[checkpoint, early_stopping])

# Check if early stopping was triggered
if early_stopping.stopped_epoch > 5:
    print("Training ended:", early_stopping.stopped_epoch)
else:
    print("Training completed")

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels, verbose=2)
print(f'Test set precision: {test_acc*100:.2f}%')

# Load the best saved model
model = tf.keras.models.load_model('the_model.h5')

# Make predictions on the test set
predictions = model.predict(test_images)
predictions_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Print the true and predicted classes for each image in the test set
for i in range(len(test_images)):
    print(f'Image {i+1}: Real class = {true_classes[i]}, Predicted class = {predictions_classes[i]}')
