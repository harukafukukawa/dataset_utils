from PIL import Image
import glob, os
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random

# This list is from usable_images.txt. Copied here for convenience.
nice_sequences = ['1-0-0-2', '11-0-0-1', '11-0-0-2', '118-0-0-1', '118-0-0-2', '12-0-0-1', '12-0-0-2', '123-1-0-2', '124-0-0-1', '124-0-0-2', '126-0-0-1', '126-0-0-2', '13-0-0-1', '13-0-0-2', '133-0-0-1', '133-0-0-2', '14-0-0-1', '14-0-0-2', '14-1-0-1', '14-1-0-2', '16-0-0-1', '16-0-0-2', '163-0-0-2', '168-1-0-1', '168-1-0-2', '17-0-0-1', '17-0-0-2', '18-0-0-1', '18-0-0-2', '181-0-0-2', '182-0-0-1', '185-0-0-2', '187-0-0-2', '2-0-0-1', '2-0-0-2', '20-0-0-1', '20-0-0-2', '20-1-1-2', '215-0-0-2', '216-0-0-1', '216-0-0-2', '26-1-0-2', '27-0-1-1', '29-0-0-1', '29-0-0-2', '3-0-0-1', '3-0-0-2', '4-0-0-1', '4-0-0-2', '4-1-0-1', '4-1-0-2', '44-0-0-2', '45-0-0-1', '45-0-0-2', '46-0-0-1', '46-0-0-2', '47-0-0-1', '47-0-0-2', '47-1-0-2', '48-0-0-1', '49-0-0-1', '49-0-0-2', '49-1-0-2', '51-0-0-2', '55-0-1-2', '57-0-0-1', '57-0-0-2', '57-1-0-2', '58-0-0-1', '58-0-0-2', '58-1-0-2', '58-1-1-2', '59-0-0-1', '59-0-0-2', '6-0-0-1', '6-0-0-2', '60-0-0-1', '61-0-0-1', '61-0-0-2', '61-1-0-2', '62-0-0-1', '62-0-0-2', '63-1-1-2', '64-0-0-1', '64-0-0-2', '65-0-0-1', '65-0-0-2', '66-0-0-1', '66-0-0-2', '67-0-1-1', '67-0-1-2', '67-1-1-1', '68-0-0-1', '68-0-0-2', '68-0-1-1', '69-0-0-1', '69-0-0-2', '75-0-0-1', '75-0-0-2', '84-0-0-1', '84-0-0-2']

IMG_SIZE = 128, 128
SIZE_FACTOR = 0.8
BLACK_THRESHOLD = 1.2

np.random.seed(seed=903727949)

def setup_dataset_directory():
    # Create directory to store processed images
    if os.path.exists("./processed_images") is False:
        print("The processed images will be stored in the current directory in a folder called processed_images")
        os.makedirs("./processed_images")
    os.makedirs("./processed_images/usable_images") if os.path.exists("./processed_images/usable_images") is False else None
    os.makedirs("./processed_images/unusable_images") if os.path.exists("./processed_images/unusable_images") is False else None

    if os.path.exists("./processed_images/nice_image_sequences") is False:
        os.makedirs("./processed_images/nice_image_sequences")


# Helper method when handpicking images (called in select_usable_images())
def is_usable(img):
    """
    Check if image is usable for training
    """
    # Convert image to greyscale
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
     
    vals = img.flatten()
    unique, counts = np.unique(vals, return_counts=True)

    if len(counts) < 3:
        return False
    # If the most common pixel value is black and is BLACK_THRESHOLD times larger than the second most common pixel value, mark as unusable
    # The BLACK_THRESHOLD times threshold was arbitrarily chosen
    if unique[np.argmax(counts)] == 0 and sorted(counts)[-1] > BLACK_THRESHOLD * sorted(counts)[-2]:
        return False
    return True

def divide_and_crop_image(img):
    # Crop and create two images. This will be one top and one bottom image centered horizontally. We do this to increase our dataset size.
    width, height = img.size
    left = width / 2 - IMG_SIZE[0] * SIZE_FACTOR + 66 # Shift images because of the 2014 images having a black border on the left
    right = width / 2 + IMG_SIZE[0] * SIZE_FACTOR + 66 
    top_1 = height / 2 - 2 * IMG_SIZE[1] * SIZE_FACTOR
    bottom_1 = height / 2
    top_2 = height / 2
    bottom_2 = height / 2 + 2 * IMG_SIZE[1] * SIZE_FACTOR
    img_1 = img.crop((left, top_1, right, bottom_1))
    img_2 = img.crop((left, top_2, right, bottom_2))
    return img_1, img_2

# This is used to aid in manually picking usable images. It does not need to be run once the sequences are already picked.
def select_usable_images(dataset_path=None):
    nice_ids = nice_sequences # This is the list on the top of the file
    use_handpicked_list = True # This is in case a different list of ids is desired
    if use_handpicked_list:
        handpicked_list_file = Path("./usable_images3.txt")
        with open(handpicked_list_file, "r") as f:
            nice_ids = f.read().splitlines()
    p = Path('.')
    if dataset_path is None:
        numbered_dirs = list(p.glob('../datasets/cracks/*/*'))
    else:
        numbered_dirs = list(p.glob(f'{dataset_path}/*/*'))
    for dir in numbered_dirs:
        if not dir.is_dir():
            continue
        for img_path in list(dir.glob('*.png')):
            img = Image.open(img_path)
            cropped_imgs = divide_and_crop_image(img)
            crop_id = 1
            id = dir.name
            for cropped_img in cropped_imgs:
                cropped_img = cropped_img.resize(IMG_SIZE)
                new_img_id = f"{id}-{crop_id}"
                new_img_name = new_img_id + "_" + img_path.name 
                # Check if images have too much black in them (filtering for the black border)
                if is_usable(cropped_img):
                    cropped_img.save("./processed_images/usable_images/" + new_img_name) # Saving usable images here in case we want to go through these manually to handpick nice sequences from scratch
                    if id + "-" + str(crop_id) in nice_ids:
                        cropped_img.save(f"./processed_images/nice_image_sequences/{new_img_name}") # This is where the Haruka's manually handpicked nice sequences are saved
                else:
                    cropped_img.save("./processed_images/unusable_images/" + new_img_name)
                crop_id += 1

def get_train_test_ids(sequences):
    np.random.shuffle(sequences)
    # Doing a 9:1 train test split since we don't have a lot of training data
    train_ids = sequences[:int(len(sequences) * 0.9)]
    test_ids = sequences[int(len(sequences) * 0.9):]
    return train_ids, test_ids

def setup_dataset(model, train_ids, test_ids, dataset_src, dataset_dest):
    if os.path.exists(dataset_dest) is False:
        os.makedirs(dataset_dest)
    if model == 'futuregan':
        if os.path.exists(dataset_dest + "/train") is False:
            os.makedirs(dataset_dest + "/train")
        if os.path.exists(dataset_dest + "/test") is False:
            os.makedirs(dataset_dest + "/test")

        numbered_dirs = list(Path(dataset_src).glob('*/'))
        for dir in numbered_dirs:
            if not dir.is_dir():
                continue
            for img_path in list(dir.glob('*.png')):
                img = Image.open(img_path)
                cropped_imgs = divide_and_crop_image(img)
                crop_id = 1
                id = dir.name
                for cropped_img in cropped_imgs:
                    cropped_img = cropped_img.resize(IMG_SIZE)
                    cropped_img = cropped_img.convert('L') # Make sure it's greyscaled
                    new_img_id = f"{id}-{crop_id}"
                    new_img_name = new_img_id + "_" + img_path.name 
                    if new_img_id in train_ids:
                        if os.path.exists(dataset_dest + "/train/" + new_img_id) is False:
                            os.makedirs(dataset_dest + "/train/" + new_img_id)
                        cropped_img.save(f"{dataset_dest}/train/{new_img_id}/{new_img_name}")
                    elif new_img_id in test_ids:
                        if os.path.exists(dataset_dest + "/test/" + new_img_id) is False:
                            os.makedirs(dataset_dest + "/test/" + new_img_id)
                        cropped_img.save(f"{dataset_dest}/test/{new_img_id}/{new_img_name}")
                    crop_id += 1
    else:
        print("Format for {model} is not supported yet")

if __name__ == "__main__":
    setup_dataset_directory()
    #select_usable_images() # This is a helper method that's useful for handpicking nice sequences. It's not needed once the sequences are already picked.
    nice_sequences = None # Comment this and the lines below (that read the usable_imges.txt) if you want to use the global nice_sequences list. Content is the same. Global is there for convenience.
    with open("./usable_images.txt", "r") as f:
        nice_sequences = f.read().splitlines()
    train_ids, test_ids = get_train_test_ids(nice_sequences)
    setup_dataset(model='futuregan', train_ids=train_ids, test_ids=test_ids, dataset_src='../datasets/Registration_US80_WB_MP8-7_H_2x2_Segmentation/', dataset_dest='./processed_images/FutureGAN_format/')
