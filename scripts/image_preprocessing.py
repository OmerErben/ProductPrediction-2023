import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_images_by_directory(directory, target_size, amount_of_images=1500):
    mode = 'RGB'
    cat_images = []
    labels = []
    for file_name in tqdm(os.listdir(directory)):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            with Image.open(os.path.join(directory, file_name)) as image:
                if image.mode != 'RGB':
                    image = image.convert(mode)
                cat_images.append(image.resize(target_size, Image.LANCZOS))
                labels.append(directory.split('/')[-1])  # Use the folder name as label
            if len(cat_images) == amount_of_images:
                return cat_images, labels
    return cat_images, labels

def load_images():
    images = []
    labels = []
    for folder_name in tqdm(os.listdir('images/train')):
        if os.path.isdir(os.path.join('images/train', folder_name)):
            cat_images, cat_labels = get_images_by_directory(f'images/train/{folder_name}', (144, 144), 1500)
            images.extend(cat_images)
            labels.extend(cat_labels)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def create_image_index_dict(images):
    return {idx: img for idx, img in enumerate(images)}
