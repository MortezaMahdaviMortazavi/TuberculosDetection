import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import cv2
import config
import preprocessing
import numpy as np

from tqdm import tqdm
from PIL import Image



import os
import pandas as pd
import cv2


def plot_images_with_labels(data_df, num_images=5, random_seed=42, show_image_mode=True):
    """
    Plot some images with their corresponding labels from the given DataFrame.

    Args:
        data_df (pd.DataFrame): The DataFrame containing 'filepaths' and 'labels' columns.
        num_images (int, optional): Number of images to plot. Defaults to 5.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        show_image_mode (bool, optional): Whether to show image mode (RGB or not) alongside the labels. 
                                          Defaults to True.
    """
    random.seed(random_seed)
    sampled_data = data_df.sample(n=num_images)

    num_rows = (num_images - 1) // 5 + 1
    num_cols = min(num_images, 5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    for idx, (filepath, label) in enumerate(zip(sampled_data['filepaths'], sampled_data['labels'])):
        image = Image.open(filepath)

        row_idx = idx // 5
        col_idx = idx % 5

        axes[row_idx, col_idx].imshow(image)
        axes[row_idx, col_idx].axis('off')

        if show_image_mode:
            is_rgb = image.mode == 'RGB'
            axes[row_idx, col_idx].set_title(f'Label: {label} | RGB: {is_rgb}')
        else:
            axes[row_idx, col_idx].set_title(f'Label: {label}')

    for idx in range(num_images, num_rows * 5):
        row_idx = idx // 5
        col_idx = idx % 5
        fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout()
    plt.show()

def trim(im):
    """
    Converts image to grayscale using cv2, then computes binary matrix
    of the pixels that are above a certain threshold, then takes out
    the first row where a certain percetage of the pixels are above the
    threshold will be the first clip point. Same idea for col, max row, max col.
    """
    percentage = 0.02

    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = img_gray > 0.1 * np.mean(img_gray[img_gray != 0])
    row_sums = np.sum(im, axis=1)
    col_sums = np.sum(im, axis=0)
    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[0] * percentage)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(im_crop)



def resize_and_save_images(data_df, save_dir, new_size=(224, 224)):
    """
    Resize images from the DataFrame and save them to the specified directory.

    Args:
        data_df (pd.DataFrame): The DataFrame containing 'filepaths' and 'labels' columns.
        save_dir (str): The directory path where the resized images will be saved.
        new_size (tuple, optional): The new size to which the images will be resized. Defaults to (224, 224).
    """
    # Create the save directories if they don't exist
    normal_save_dir = os.path.join(save_dir, 'Normal')
    tuberculosis_save_dir = os.path.join(save_dir, 'Tuberculosis')
    os.makedirs(normal_save_dir, exist_ok=True)
    os.makedirs(tuberculosis_save_dir, exist_ok=True)

    for filepath, label in tqdm(zip(data_df['filepaths'], data_df['labels'])):
        image = cv2.imread(filepath)
        trimmed_image = np.array(trim(image))
        resized_image = cv2.resize(trimmed_image, new_size, interpolation=cv2.INTER_CUBIC)

        if label == 'Normal':
            label_save_dir = normal_save_dir
        else:
            label_save_dir = tuberculosis_save_dir

        filename_without_ext = os.path.splitext(os.path.basename(filepath))[0]

        save_filename = f"{filename_without_ext}.png"
        save_path = os.path.join(label_save_dir, save_filename)
        cv2.imwrite(save_path, resized_image)

# # Example usage:
# Assuming your DataFrame is called 'data_df' and has columns 'filepaths' and 'labels'
train_df, test_df, valid_df, class_count, average_height, average_weight, aspect_ratio = preprocessing.make_dataframes(config.DATASET_DIR)
save_directory = 'resized_dataset/'  # Set the directory path where resized images will be saved

resize_and_save_images(train_df, save_directory)
resize_and_save_images(valid_df, save_directory)
resize_and_save_images(test_df, save_directory)





# xray_image = cv2.imread('dataset/Normal/Normal-3434.png')
# processed_image = trim(xray_image)
# # Display the original and processed images side by side
# cv2.imshow('Original Image', xray_image)
# cv2.imshow('Processed Image', processed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
