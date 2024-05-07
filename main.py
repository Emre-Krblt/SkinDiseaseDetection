import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.metrics import Precision,Recall
from utils import *




train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(size='28')


print(f'The shape of Training images: {train_images.shape}')
print(f'The shape of Training labels: {train_labels.shape}\n')
print(f'The shape of Validation images: {val_images.shape}')
print(f'The shape of Validation labels: {val_labels.shape}\n')
print(f'The shape of Test images: {test_images.shape}')
print(f'The shape of Test labels: {test_labels.shape}')

print(f'The number of class 0 (akiec): {sum(train_labels == 0).item():3}')
print(f'The number of class 1 (bcc): {sum(train_labels == 1).item():5}')
print(f'The number of class 2 (bkl): {sum(train_labels == 2).item():5}')
print(f'The number of class 3 (df): {sum(train_labels == 3).item():5}')
print(f'The number of class 4 (nv): {sum(train_labels == 4).item():6}')
print(f'The number of class 5 (mel): {sum(train_labels == 5).item():6}')
print(f'The number of class 6 (vasc): {sum(train_labels == 6).item():3}')

# Normalize images so they can be between 0 and 1
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')  # 7 classes for Dermamnist
])

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=7)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=7)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=7)


model.summary()

# Check Model Output Shape
print("Model Output Shape:", model.output_shape)

# Check Label Shape
print("Train Labels Shape:", train_labels.shape)
print("Validation Labels Shape:", val_labels.shape)
print("Test Labels Shape:", test_labels.shape)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',Precision(),Recall()])

# Set the number of epochs and batch size
epochs = 500
batch_size = 32

# Starting to train model
history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels))

# Extract the history from the training process
history_dict = history.history

# Loss
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

# Accuracy
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

# Precision
precision_values = history_dict['precision']
val_precision_values = history_dict['val_precision']

# Recall
recall_values = history_dict['recall']
val_recall_values = history_dict['val_recall']

epochs = range(1, len(loss_values) + 1)

# Create a new figure
plt.figure(figsize=(20, 5))


# Plot the accuracy
plt.subplot(1, 4, 1)
plt.plot(epochs, acc_values, 'b-', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'orange', label='Validation accuracy')
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot the loss
plt.subplot(1, 4, 2)
plt.plot(epochs, loss_values, 'b-', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot the precision
plt.subplot(1, 4, 3)
plt.plot(epochs, precision_values, 'b-', label='Training precision')
plt.plot(epochs, val_precision_values, 'orange', label='Validation precision')
plt.title('Model precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

# Plot the recall
plt.subplot(1, 4, 4)
plt.plot(epochs, recall_values, 'b-', label='Training recall')
plt.plot(epochs, val_recall_values, 'orange', label='Validation recall')
plt.title('Model recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

# Display the figure
plt.tight_layout()
plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_images, test_labels, batch_size=batch_size)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')

