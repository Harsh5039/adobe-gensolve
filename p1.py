import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, Input
from sklearn.model_selection import train_test_split

# Define image dimensions and input shape
img_width, img_height = 128, 128
input_shape = (img_width, img_height, 1)  # Assuming grayscale images

# Function to load images from CSV
def load_images_from_csv(file_path, img_width, img_height):
    df = pd.read_csv(file_path)
    images = df.values
    images = images.reshape(-1, img_width, img_height, 1)  # Reshape to (num_images, img_width, img_height, 1)
    images = images / 255.0  # Normalize
    return images

# Function to save images to CSV
def save_images_to_csv(images, file_path):
    images = images.reshape(images.shape[0], -1)  # Flatten each image
    df = pd.DataFrame(images)
    df.to_csv(file_path, index=False)

# Load images from CSV
irregular_images = load_images_from_csv('path_to_irregular_shapes.csv', img_width, img_height)
regular_images = load_images_from_csv('path_to_regular_shapes.csv', img_width, img_height)

# Ensure the number of images in both CSV files are the same
assert irregular_images.shape[0] == regular_images.shape[0], "Mismatch in the number of irregular and regular images."

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(irregular_images, regular_images, test_size=0.2, random_state=42)

# Define the autoencoder model
input_img = Input(shape=input_shape)

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
encoded = Dense(1024, activation='relu')(x)

# Decoder
x = Dense(16 * 16 * 128, activation='relu')(encoded)
x = Reshape((16, 16, 128))(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Evaluate the model
loss = autoencoder.evaluate(x_val, y_val)
print(f'Validation Loss: {loss}')

# Generate and save regularized images
decoded_images = autoencoder.predict(irregular_images)
save_images_to_csv(decoded_images, 'path_to_output_regularized_shapes.csv')
