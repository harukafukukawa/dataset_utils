import pdb
from PIL import Image
import glob, os
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import re
import random
from shutil import copy2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
IMG_SIZE = 128, 128
SIZE_FACTOR = 1
BLACK_THRESHOLD = 1.5

debug = False

# Manually selected sequences (under usable images)
nice_sequences = [
    '1-0-0', '2-0-0', '3-0-0', '3-1-0', '4-0-0', '4-1-0', '5-0-0', '6-0-0', '6-1-0', '11-0-0', '12-0-0', '43-0-0', '44-0-0', '45-1-0', '46-0-0', '47-0-0', '49-0-0', '51-0-0', '56-0-0', '57-0-0',
    '58-0-0', '58-1-0', '59-0-0', '60-0-0', '60-1-0', '61-0-0', '61-1-0', '62-0-0', '63-0-0', '63-1-1', '64-0-0', '65-0-0', '67-0-1', '68-0-0', '69-0-0', '70-0-0', '75-0-0', '84-0-0', '118-0-0', '124-0-0', 
    '128-0-0', '133-0-0', '155-0-0', '163-0-0', '178-0-0', '183-1-0', '186-0-0', '187-0-0', '192-0-0', '193-0-0', '196-0-0', '205-0-0', '208-0-0', '212-0-0', '214-0-0', '216-0-0', '266-0-0', '296-0-0', '299-0-0'
]

# Manually rejected sequences (under usable images)
bad_sequences = [
    '3-1-1', '10-1-1', '11-0-1', '15-1-1', '40-0-0', '129-0-0'
]

def make_binary_thresholded_segmentation_mask(img, threshold=180):
    """Make a mask for the image by thresholding and truncating"""
    img = img.convert('L')
    img = np.array(img)
    # Comment in if we want to try to get rid of black background (current implementation doesn't work that well)
#    img[img < 3] = 255
    img[img < threshold] = 0
    img[img >= threshold] = 255
    img = Image.fromarray(img)
    return img

def make_thresh_trunc_segmentation_mask(img, threshold=185):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img,threshold,255,cv2.THRESH_TRUNC)
    img[img >= threshold] = 255
    img = Image.fromarray(img)
    return img

def make_otsu_thresholded_segmentation_mask(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    img = Image.fromarray(img)
    return img

def make_otsu_thresholded_blurred_segmentation_mask(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    img = Image.fromarray(img)
    return img

def make_adaptive_segmentation_mask(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,20)
    img = Image.fromarray(img)
    return img
    
def make_segmentation_masks(img, new_img_name, method='selected'):
    """
    Normalize then create segmentation masks of image
    method can be 'all' which makes segmentation masks using all available methods, or 'selected' which only does it for the binary threshold and truncated threshold methods
    """
    if method == 'selected':
        sm = make_binary_thresholded_segmentation_mask(img)
        resize_and_save(sm, new_img_name, "binary_thresholded")
        sm = make_thresh_trunc_segmentation_mask(img)
        resize_and_save(sm, new_img_name, "threshold_truncated")
    if method == 'all':
        sm = make_binary_thresholded_segmentation_mask(img)
        resize_and_save(sm, new_img_name, "binary_thresholded")
        sm = make_thresh_trunc_segmentation_mask(img)
        resize_and_save(sm, new_img_name, "threshold_truncated")
        sm = make_otsu_thresholded_segmentation_mask(img)
        resize_and_save(sm, new_img_name, "otsu_thresholded")
        sm = make_otsu_thresholded_blurred_segmentation_mask(img)
        resize_and_save(sm, new_img_name, "otsu_thresholded_blurred")
        sm = make_adaptive_segmentation_mask(img)
        resize_and_save(sm, new_img_name, "adaptive")

def resize_and_save(img, img_name, subdir_name):
    try:
        img = img.resize(IMG_SIZE)
    except Exception as e:
        print(e)
        print(img_name)
    finally:
        img.save("./processed_images/segmentation_masks/" + subdir_name + "/" + img_name )
        get_id = re.search(r'^\d+-\d-\d', img_name)
        id = get_id.group(0)
        if id in nice_sequences:
            img.save("./processed_images/nice_image_sequences/segmentation_masks/" + subdir_name + "/" + img_name )

#TODO: Do this for all directories used in the code. Make it cleaner so it doesn't look like a mess
def setup_dataset_directory():
    # Create directory to store processed images
    if os.path.exists("./processed_images") is False:
        print("The processed images will be stored in the current directory in a folder called processed_images")
        os.makedirs("./processed_images")
    os.makedirs("./processed_images/usable_images") if os.path.exists("./processed_images/usable_images") is False else None
    os.makedirs("./processed_images/unusable_images") if os.path.exists("./processed_images/unusable_images") is False else None
    os.makedirs("./processed_images/segmentation_masks") if os.path.exists("./processed_images/segmentation_masks") is False else None
    
    segmentation_masks = ['binary_thresholded', 'threshold_truncated', 'otsu_thresholded', 'otsu_thresholded_blurred', 'adaptive']
    for sm in segmentation_masks:
        os.makedirs(f"./processed_images/segmentation_masks/{sm}") if os.path.exists(f"./processed_images/segmentation_masks/{sm}") is False else None

    usable_images_file = Path("./usable_images.txt")
    unusable_images_file = Path("./unusable_images.txt")
    
    # Create these files if they don't exist
    if usable_images_file.is_file() is False:
        open("./usable_images.txt", "w").close()
    if unusable_images_file.is_file() is False:
        open("./unusable_images.txt", "w").close()
        
def is_usable(img):
    """
    Check if image is usable for training
    """
    # Convert image to greyscale
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
     
    vals = img.flatten()
    unique, counts = np.unique(vals, return_counts=True)
    if debug:
        print(f"{np.argmax(counts)=}")

    if len(counts) < 2:
        return False
    # If the most common pixel value is black and is BLACK_THRESHOLD times larger than the second most common pixel value, mark as unusable
    # The BLACK_THRESHOLD times threshold was arbitrarily chosen
    if unique[np.argmax(counts)] == 0 and sorted(counts)[-1] > BLACK_THRESHOLD * sorted(counts)[-2]:
        return False
    return True

# Try to register-- if it fails, then keep original
# This does not work very well, so we will not use it. Kept here in case further development is attempted.
def register_sequence(img_set):
    ct=0
    # Define the parameters for the feature detection algorithm
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    prev_img = Image.open(img_set[0])
    width, height = prev_img.size
    left = width / 2 - IMG_SIZE[0] * SIZE_FACTOR
    right = width / 2 + IMG_SIZE[0] * SIZE_FACTOR
    top = height / 2 - IMG_SIZE[1] * SIZE_FACTOR
    bottom = height / 2 + IMG_SIZE[1] * SIZE_FACTOR
    prev_img = prev_img.crop((left, top, right, bottom))

#    prev_img = cv2.imread(str(img_set[0]))
    prev_img = cv2.cvtColor(np.array(prev_img), cv2.COLOR_BGR2GRAY)


    prev_prev_img = prev_img
    registered_img_set = []
    for img in img_set[1:]:
        ct+=1
        img = Image.open(img)
        width, height = img.size
        left = width / 2 - IMG_SIZE[0] * SIZE_FACTOR
        right = width / 2 + IMG_SIZE[0] * SIZE_FACTOR
        top = height / 2 - IMG_SIZE[1] * SIZE_FACTOR
        bottom = height / 2 + IMG_SIZE[1] * SIZE_FACTOR
        img = img.crop((left, top, right, bottom))
#        img = cv2.imread(str(img))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        try:
            # Detect features in the first image
            p0 = cv2.goodFeaturesToTrack(prev_img, mask=None, **feature_params)
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img, img, p0, None)
            # Calculate the homography matrix between the images
            M, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 5.0)
            # Apply the homography matrix to the second image
            registered_image = cv2.warpPerspective(img, M, (prev_img.shape[1], prev_img.shape[0]))
        except:
            print("Error calculating homography matrix")
            try:
                # Try using the previous previous image as reference instead
                # Detect features in the first image
                p0 = cv2.goodFeaturesToTrack(prev_prev_img, mask=None, **feature_params)
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_prev_img, img, p0, None)
                # Calculate the homography matrix between the images
                M, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 5.0)
                # Apply the homography matrix to the second image
                registered_image = cv2.warpPerspective(img, M, (prev_img.shape[1], prev_img.shape[0]))
            except:
                print("Error #2 calculating homography matrix")
                registered_image = img

        registered_image = Image.fromarray(registered_image)
        prev_prev_img = prev_img
        prev_img = img
        resize_and_save(registered_image, f"registered_{ct}.png", "tmp")
        registered_img_set.append(registered_image)

    print(registered_img_set)
    return registered_img_set

# Excepted dataset_path structure:
'''
cracks/SCI VIP - Crack Forecasting - Registration_US80_WB_MP8-7_H_2x2
cracks/SCI VIP - Crack Forecasting - Registration_US80_WB_MP8-7_H_2x2 / 27-1-0
cracks/SCI VIP - Crack Forecasting - Registration_US80_WB_MP8-7_H_2x2 / 27-1-0 / 2014-2015.png
cracks/SCI VIP - Crack Forecasting - Registration_US80_WB_MP8-7_H_2x2 / 27-1-0 / 2015-2016.png
...etc

So, if cracks were in /tmp/cracks, then dataset_path would be /tmp/cracks
'''
def select_usable_images(dataset_path=None):
    p = Path('.')
    if dataset_path is None:
        numbered_dirs = list(p.glob('../datasets/cracks/*/*'))
    else:
        numbered_dirs = list(p.glob(f'{dataset_path}/*/*'))
#    ct = 0
    for dir in numbered_dirs:
        # The ct is used during development to limit how many images we process. Comment ct stuff out when we want to process all images.
#        ct += 1
#        if ct > 50:
#            break
#        if not dir.is_dir():
#            continue
#        registered_imgs = register_sequence(list(dir.glob('*.png')))
#        print(f"{registered_imgs=}")
#        pdb.set_trace()
#        img_paths_and_registered_imgs = zip(list(dir.glob('*.png')), registered_imgs)
        for img_path in list(dir.glob('*.png')):
            if debug:
                print(img_path)
            img = Image.open(img_path)
            
            # Crop, so we're looking at the center of the image only
            width, height = img.size
            left = width / 2 - IMG_SIZE[0] * SIZE_FACTOR + 55 # Shift images because of the 2014 images having a black border on the left
            right = width / 2 + IMG_SIZE[0] * SIZE_FACTOR + 55
            top = height / 2 - IMG_SIZE[1] * SIZE_FACTOR
            bottom = height / 2 + IMG_SIZE[1] * SIZE_FACTOR
            img = img.crop((left, top, right, bottom))
            usable = is_usable(img)
            new_img_name = dir.name + "_" + img_path.name 
            img = img.resize(IMG_SIZE)
            
            if usable:
                make_segmentation_masks(img, new_img_name, method='all')
                img.save("./processed_images/usable_images/" + new_img_name)
                get_id = re.search(r'^\d+-\d-\d', new_img_name)
                id = get_id.group(0)
                if id in nice_sequences:
                    img.save("./processed_images/nice_image_sequences/" + new_img_name)
                with open("usable_images.txt", "a") as file:
                    file.write(new_img_name + "\n")
            else:
                img.save("./processed_images/unusable_images/" + new_img_name)
                with open("unusable_images.txt", "a") as file:
                    file.write(new_img_name + "\n")

def details_about_sequence(nice_sequences, model=None):
    p = Path('./processed_images/nice_image_sequences')
    # Store how many images are in each sequence, which year(s) are missing, the first year of the sequence, and last year of the sequence
    df = pd.DataFrame(columns=['sequence', 'num_images', 'missing_years', 'first_year', 'last_year'])
    sequence_paths = list(p.glob('./*.png'))
    sequence_paths.sort()
    sequence_info = []
    curr_img_id = None
    num_images = 0
    missing_years = []
    first_year = None
    last_year = None
    list_of_images = []
    for img_path in sequence_paths:
        filename = os.path.basename(img_path)
        regex = re.match(r"(?P<id>\d+-\d-\d)_(?P<year>\d+)", filename)
        if curr_img_id != regex['id']:
            if len(sequence_info) > 0:
                if len(missing_years) == 0:
                    missing_years = None
                sequence_info.extend([num_images, missing_years, first_year, last_year])
                row = pd.DataFrame([sequence_info], columns=['sequence', 'num_images', 'missing_years', 'first_year', 'last_year'])
                df = pd.concat([df, row], ignore_index=True)
            sequence_info = []
            curr_img_id = regex['id']
            sequence_info.append(curr_img_id)
            num_images = 1
            missing_years = []
            first_year = regex['year']
            last_year = regex['year']
        else:
            num_images += 1
            if int(regex['year']) - int(last_year) > 1:
                missing_years.append(int(last_year) + 1)
            last_year = regex['year']
    print("Details about sequences:")
    print(df)
    print()
    print("These are the sequences where all images (including 2014) were deemed usable:")
    full_sequences = df[df['num_images'] == 6].values
    print(full_sequences)
    print(f"There are {len(df[df['num_images'] == 6])} total images that fit this criteria")
    
    with open("usable_sequences.txt", "w") as file:
                    file.write(str(full_sequences)[1:-1])
    ids_of_full_sequences = full_sequences[:, 0]
    create_train_test_datasets(full_sequences)
    return ids_of_full_sequences
    print("Splitting train and test data")

        
def create_train_test_datasets(full_sequences, model="FutureGAN"):
    np.random.shuffle(full_sequences)
    train_ids = full_sequences[:int(len(full_sequences) * 0.9)]
    test_ids = full_sequences[int(len(full_sequences) * 0.9):]
    print()
    print(f"Training Set IDs ({len(train_ids)} Total):\n{train_ids[:, 0]}\n")
    print(f"Testing Set IDs ({len(test_ids)}Total):\n{test_ids[:,0]}")
    if model == "FutureGAN":
        nice_img_seq_path = Path('./processed_images/nice_image_sequences')
        seg_mask_path = Path('/Users/haruka/Documents/code/crack_image_forecasting/processed_images/nice_image_sequences/segmentation_masks/threshold_truncated')
        
        '''
        For FutureGAN:
        Data is assumed to be arranged in this way:
            data_root/video/frame.ext -> dataset/train/video1/frame1.ext
                                                    -> dataset/train/video1/frame2.ext
                                                    -> dataset/train/video2/frame1.ext
                                                    -> ...
        '''

        if os.path.exists("./processed_images/FutureGAN_format") is False:
            os.makedirs("./processed_images/FutureGAN_format")

        for img_path in nice_img_seq_path.glob('./*.png'):
            filename = os.path.basename(img_path)
            regex = re.match(r"(?P<id>\d+-\d-\d)_(?P<year>\d+)", filename)
            futuregan_path_train = f"./processed_images/FutureGAN_format/train/{regex['id']}" 
            futuregan_path_test = f"./processed_images/FutureGAN_format/test/{regex['id']}" 
            if regex['id'] in train_ids:
                if os.path.exists(futuregan_path_train) is False:
                    os.makedirs(futuregan_path_train)
                copy2(seg_mask_path / img_path.name, futuregan_path_train)
                    
            elif regex['id'] in test_ids:
                if os.path.exists(futuregan_path_test) is False:
                    os.makedirs(futuregan_path_test)
                copy2(seg_mask_path / img_path.name, futuregan_path_test)
    return train_ids, test_ids

     
# This is not completed or checked yet
def format_for_convLSTM(sequences, dataset_path=None):
    if dataset_path is None:
        path = Path('/Users/haruka/Documents/code/crack_image_forecasting/processed_images/nice_image_sequences/segmentation_masks/threshold_truncated')
    else:
        path = Path(dataset_path)
    np.random.shuffle(sequences)
    train_ids = sequences[:int(len(sequences) * 0.9)]
    test_ids = sequences[int(len(sequences) * 0.9):]
    train_seq = []
    train_target = []
    test_seq = []
    test_target = []
    for seq_id in sequences:
        image_sequence = []
        target = []
        for i in range(1, 7):
            image = Image.open(path / f"{seq_id}_201{i}-201{i+1}.png")
            image_tensor = transforms.ToTensor()(image)
            if i == 6:
                target.append(image_tensor)
            else:
                image_sequence.append(image_tensor)
        image_sequence = torch.stack(image_sequence, dim=0)
        target = torch.stack(target, dim=0)
        if seq_id in train_ids:
            train_seq.append(image_sequence)
            train_target.append(target)
        elif seq_id in test_ids:
            test_seq.append(image_sequence)
            test_target.append(target)
    # Stack the image sequences along the batch dimension to create a tensor of size (number_of_sequences, 6, 1, 128, 128)
    train_seq = torch.stack(train_seq, dim=0)
    test_seq = torch.stack(test_seq, dim=0)

    train_target = torch.stack(train_target, dim=0)
    test_target = torch.stack(test_target, dim=0)
    print(f"{test_target.shape=}, {test_seq.shape=}")
    return train_seq, train_target, test_seq, test_target

if __name__ == '__main__':
    np.random.seed(seed=7)
    setup_dataset_directory()
    select_usable_images()
#    print(f"{len(nice_sequences)=}")
    full_sequences = details_about_sequence(nice_sequences, model=None)
    print()
    print("These are the manually selected usable sequences:")
    print(full_sequences)
#    train_seq, train_target, test_seq, test_target = format_for_convLSTM(full_sequences)
#    print(train_seq.shape)
#    print(test_seq.shape)