# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images to a 0-1 scale
train_images = train_images / 255.0
test_images = test_images / 255.0

# Label names for display
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(10, activation='softmax') # Output layer with 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("\nTest accuracy:", test_accuracy)

# Make predictions
predictions = model.predict(test_images)

# Plot predictions for 15 random test images
plt.figure(figsize=(15, 8))
random_indices = random.sample(range(len(test_images)), 15)  # Pick 15 random indices

for i, idx in enumerate(random_indices):
    plt.subplot(3, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[idx], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions[idx])
    actual_label = test_labels[idx]

    color = 'green' if predicted_label == actual_label else 'red'
    plt.xlabel(f"P: {class_names[predicted_label]}\nA: {class_names[actual_label]}", color=color)

plt.tight_layout()
plt.show()
