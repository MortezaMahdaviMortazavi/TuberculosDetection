import cv2
import os
import random
import numpy as np

from tqdm import tqdm
from utils import trim

class Augmentation:
    def __init__(self, output_path, probability=0.4):
        self.output_path = output_path
        self.probability = probability

    def random_rotation(self, image, angle_range=(-30, 30)):
        if random.random() < self.probability:
            angle = random.uniform(angle_range[0], angle_range[1])
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        return image

    def random_flip(self, image):
        if random.random() < self.probability:
            image = cv2.flip(image, 1)  # Horizontal flip
        return image
    
    def random_zoom(self, image, zoom_range=(0.8, 1.2)):
        if random.random() < self.probability:
            zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
            h, w = image.shape[:2]

            # Calculate new zoomed image size
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

            # Ensure the new size does not exceed the original size
            if new_h >= h:
                new_h = h - 1
            if new_w >= w:
                new_w = w - 1

            # Resize the image
            zoomed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Compute border size for padding
            top = (h - new_h) // 2
            bottom = h - new_h - top
            left = (w - new_w) // 2
            right = w - new_w - left

            # Pad the zoomed image to the original size
            image = cv2.copyMakeBorder(zoomed_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        return image

    def color_jittering(self, image, jitter_range=20):
        if random.random() < self.probability:
            b, g, r = cv2.split(image)
            b = np.clip(b.astype(np.int32) + random.randint(-jitter_range, jitter_range), 0, 255).astype(np.uint8)
            g = np.clip(g.astype(np.int32) + random.randint(-jitter_range, jitter_range), 0, 255).astype(np.uint8)
            r = np.clip(r.astype(np.int32) + random.randint(-jitter_range, jitter_range), 0, 255).astype(np.uint8)
            image = cv2.merge((b, g, r))
        return image


    def __call__(self, image_path,index,class_name):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as a color image

        # Apply augmentations
        image = self.random_rotation(image)
        image = self.random_flip(image)
        image = self.random_zoom(image)
        image = self.color_jittering(image)
        image = np.array(trim(image))
        # Save the augmented image
        filename = os.path.basename(image_path)
        output_image_path = os.path.join(self.output_path, f"Augment-{class_name}-{index}.png")
        cv2.imwrite(output_image_path, np.array(trim(image)))
        return image


def augmentation_main():
    # Define input and output directories
    input_dir = 'resized_dataset'
    output_dir_normal = os.path.join(input_dir, 'Augmented_normal')
    output_dir_tuberculous = os.path.join(input_dir, 'Augmented_tuberculosis')

    # Create output directories if they don't exist
    os.makedirs(output_dir_normal, exist_ok=True)
    os.makedirs(output_dir_tuberculous, exist_ok=True)

    # Create the Augmentation objects
    augmentor_normal = Augmentation(output_path=output_dir_normal, probability=0.6)  # Set lower probability for Normal images
    augmentor_tb = Augmentation(output_path=output_dir_tuberculous, probability=0.8)  # Set higher probability for TB images

    # Augment Normal images (randomly choose 500 samples)
    normal_images_dir = os.path.join(input_dir, 'Normal')
    normal_images = os.listdir(normal_images_dir)
    selected_normal_images = random.sample(normal_images, 500)
    num = 1
    for img_file in tqdm(selected_normal_images):
        img_path = os.path.join(normal_images_dir, img_file)
        augmented_image = augmentor_normal(img_path,index=num,class_name='Normal')
        num+=1

    # Augment Tuberculous images to have a total of 2000 augmented images
    tb_images_dir = os.path.join(input_dir, 'Tuberculosis')
    tb_images = os.listdir(tb_images_dir)
    num_tb_images = 700
    num_augmentations_needed = 2000 - num_tb_images

    for i in tqdm(range(num_augmentations_needed)):
        img_file = random.choice(tb_images)
        img_path = os.path.join(tb_images_dir, img_file)
        augmented_image = augmentor_tb(img_path,index=i+1,class_name='Tuberculosis')

# Call the function to perform the augmentation
augmentation_main()
