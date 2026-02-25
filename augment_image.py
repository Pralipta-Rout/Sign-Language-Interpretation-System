import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Define the directory where your images are stored
directory = 'SignImage48x48/'

# Initialize ImageDataGenerator with augmentations
datagen = ImageDataGenerator(
    rotation_range=15,  # Rotate up to 15 degrees
    zoom_range=0.1,     # Zoom in or out by 10%
    width_shift_range=0.1,  # Shift image horizontally by 10%
    height_shift_range=0.1,  # Shift image vertically by 10%
    shear_range=0.1,    # Apply shear transformation
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Fill missing pixels after transformations
)

# Function to augment images for each folder
def augment_images_for_folder(letter_folder):
    folder_path = os.path.join(directory, letter_folder)
    images = os.listdir(folder_path)
    current_count = len(images)

    # If the folder already has 500 images, skip it
    if current_count >= 500:
        print(f"Folder '{letter_folder}' already has enough images.")
        return

    # Calculate how many more images are needed to reach 500
    target_count = 500
    additional_images_needed = target_count - current_count

    print(f"Folder '{letter_folder}' has {current_count} images. Generating {additional_images_needed} more.")

    # Process each image in the folder
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path)
        img = img.resize((48, 48))  # Ensure all images are the same size

        # If the image is grayscale, no need to convert to RGB, we keep it single-channel (48, 48, 1)
        if img.mode == 'L':  # Grayscale (1 channel)
            x = np.array(img)
            x = x.reshape((1, 48, 48, 1))  # Add batch dimension and 1 channel (grayscale)
        else:
            x = np.array(img)
            x = x.reshape((1, 48, 48, 3))  # For RGB images, reshape to include 3 channels (RGB)

        # Generate the needed number of augmentations
        augmentations_done = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=folder_path, save_prefix='aug', save_format='jpg'):
            augmentations_done += 1
            if augmentations_done >= additional_images_needed:
                break  # Stop once we've generated enough images

# Process each letter and the "blank" folder
letters = [chr(i) for i in range(65, 91)]  # A-Z
letters.append('blank')  # Include "blank" folder

for letter in letters:
    augment_images_for_folder(letter)