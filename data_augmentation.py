import cv2
import os
import numpy as np

# Create output folder
output_folder = "augmented_dataset"
os.makedirs(output_folder, exist_ok=True)

# Path to your dataset
input_folder ="C:/Users/Lenovo/dataset"


# Function to change brightness
def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Augmentation functions
def augment_image(image, image_name, output_folder, label):
    augmented_images = []
    augmented_names = []

    # Original image
    augmented_images.append(image)
    augmented_names.append(f"{label}_original_{image_name}")

    # Brightness adjustments
    augmented_images.append(adjust_brightness(image, 50))  # Brighter
    augmented_names.append(f"{label}_bright_{image_name}")
    augmented_images.append(adjust_brightness(image, -50))  # Darker
    augmented_names.append(f"{label}_dark_{image_name}")

    # Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Maintain 3 channels
    augmented_images.append(gray_image)
    augmented_names.append(f"{label}_gray_{image_name}")

    # Horizontal flip
    h_flip = cv2.flip(image, 1)
    augmented_images.append(h_flip)
    augmented_names.append(f"{label}_hflip_{image_name}")

    # Vertical flip
    v_flip = cv2.flip(image, 0)
    augmented_images.append(v_flip)
    augmented_names.append(f"{label}_vflip_{image_name}")

    # Save augmented images
    for aug_img, aug_name in zip(augmented_images, augmented_names):
        save_path = os.path.join(output_folder, aug_name)
        cv2.imwrite(save_path, aug_img)


# Loop through dataset
for label in os.listdir(input_folder):
    person_folder = os.path.join(input_folder, label)
    if not os.path.isdir(person_folder):
        continue

    # Create a folder for each label in the augmented dataset
    person_output_folder = os.path.join(output_folder, label)
    os.makedirs(person_output_folder, exist_ok=True)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = cv2.imread(image_path)
        augment_image(image, image_name, person_output_folder, label)

print("Data augmentation complete. Augmented images saved in 'augmented_dataset'.")
