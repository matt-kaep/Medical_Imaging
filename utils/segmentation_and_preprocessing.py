import matplotlib.pyplot as plt  # Library for visualization
import numpy as np  # Library for numerical data processing
import pandas as pd  # Library for data manipulation
import io
import imageio as io  # Pour lire les images
import cv2 as cv  # OpenCV library for image processing
import os
import DarkArtefactRemoval as dca  # Library for dark artifact removal
import dullrazor as dr  # Library for hair artifact removal

from sklearn.cluster import KMeans  # Library for KMeans clustering
from skimage.morphology import remove_small_objects  # Library for removing small objects
from tqdm import tqdm  # Library for progress bars

def five_segmentation(image):
    """
    This function performs thresholding on different color channels of the input image and returns the thresholded masks.
    The color channels used are:
    - Blue channel from RGB
    - b channel from CIE-Lab
    - x, y, z channels from CIE-XYZ (after 3D color clustering)

    Parameters:
    image (numpy array): Input RGB image

    Returns:
    tuple: A tuple containing five thresholded masks as numpy arrays (thresholded_blue, thresholded_b, thresholded_x, thresholded_y, thresholded_z)
    """

    # Thresholding with the blue channel (RGB)
    blue_channel = image[:, :, 0]
    _, thresholded_blue = cv.threshold(blue_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresholded_blue = cv.bitwise_not(thresholded_blue)/255

    # Thresholding with the b channel (CIE-Lab)
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    b_channel = lab_image[:, :, 2]
    _, thresholded_b = cv.threshold(b_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresholded_b = cv.bitwise_not(thresholded_b)/255

    # 3D color clustering with CIE-XYZ
    xyz_image = cv.cvtColor(image, cv.COLOR_BGR2XYZ)
    xyz_features = xyz_image.reshape((-1, 3))
    x, y, z = cv.split(xyz_image)
    # Reshape x, y, z back to their original 2D shapes
    x = x.reshape(image.shape[:2])
    y = y.reshape(image.shape[:2])
    z = z.reshape(image.shape[:2])

    # Thresholding on x, y, z
    _, thresholded_x = cv.threshold(x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresholded_x = cv.bitwise_not(thresholded_x)/255
    _, thresholded_y = cv.threshold(y, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresholded_y = cv.bitwise_not(thresholded_y)/255
    _, thresholded_z = cv.threshold(z, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresholded_z = cv.bitwise_not(thresholded_z)/255

    return thresholded_blue, thresholded_b, thresholded_x, thresholded_y, thresholded_z



# Remove small elements of the segmentation mask
def remove_small_parts_and_fill(five_masks, min_size=1000):
    """
    This function removes small elements from the segmentation masks and fills holes in the masks.
    :param five_masks: List of five binary segmentation masks
    :param min_size: Minimum size of an object to be considered for removal
    :return: List of processed binary segmentation masks
    """
    five_masks_list = list(five_masks)
    # Opening
    radius = 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

    for i in range(len(five_masks_list)):
        five_masks_list[i] = five_masks_list[i].astype(np.uint8)
        # Opening
        five_masks_list[i] = cv.morphologyEx(five_masks_list[i], cv.MORPH_CLOSE, kernel)
        five_masks_list[i] = five_masks_list[i].astype(bool)
        five_masks_list[i] = remove_small_objects(five_masks_list[i], min_size=min_size)
    return five_masks_list

def union_mask(five_masks):
    """
    This function applies rules to select and combine the segmentation masks.
    :param five_masks: List of five binary segmentation masks
    :return: Binary mask resulting from the combination of the selected segmentation masks
    """
    mask_state = [True, True, True, True, True]

    # Rule 1: Segmentation results with the lesion mask growing into the image border are rejected
    # Check if the lesion mask grows into the image border
    for i in range(5):
        if np.any(five_masks[i][0, :]) or np.any(five_masks[i][-1, :]) or np.any(five_masks[i][:, 0]) or np.any(five_masks[i][:, -1]):
            mask_state[i] = False

    # Rule 2: Segmentation results without any detected region are rejected
    for i in range(0, len(five_masks)):
        if np.sum(five_masks[i]) == 0:
            mask_state[i] = False

    # Rule 4: The segmentation result with the smallest mask is rejected
    smallest_mask_index = np.argmin([np.sum(mask) for mask in five_masks])
    # Reject the segmentation result with the smallest mask
    mask_state[smallest_mask_index] = False

    # Rule 5: Segmentation results, whose mask areas differ too much from the other segmentation results, are rejected
    mask_areas = [np.sum(mask) for mask in five_masks]
    mean_mask_area = np.mean(mask_areas)
    for i in range(0, len(mask_areas)):
        if mask_areas[i] < 0.5 * mean_mask_area or mask_areas[i] > 1.5 * mean_mask_area:
            mask_state[i] = False

    # If all masks are rejected, accept the mask closest to the mean mask area
    if mask_state == [False, False, False, False, False]:
        i = np.argmin([mask_areas - mean_mask_area])
        mask_state[i] = True

    # Build the final mask
    mask_true = [five_masks[i] for i in range(len(mask_state)) if mask_state[i] == True]
    united_mask = np.logical_or.reduce(mask_true)

    return united_mask

def postprocessing(image, mask_dca):
    """
    This function applies the DCA mask to the input image.
    :param image: Input RGB image
    :param mask_dca: Binary DCA mask
    :return: Processed RGB image
    """
    mask_dca = cv.bitwise_not(mask_dca)
    masked_image = cv.bitwise_and(image, image, mask=mask_dca)
    return masked_image

def inpainting_dca(image):
    """
    This function performs dark artifact removal and inpainting on the input image.
    :param image: Input RGB image
    :return: Inpainted RGB image and binary DCA mask
    """
    # Perform DCA
    dca_mask = dca.get_mask(image)
    if dca_mask is None:
        return image, None
    dca_mask_np = np.array(dca_mask, dtype=np.uint8)
    inpainted_image = cv.inpaint(image, dca_mask_np, 3, cv.INPAINT_TELEA)
    return inpainted_image, dca_mask_np

def compute_segmentation(image_path):
    """
    This function computes the segmentation mask for an input image.
    :param image_path: Path to the input image
    :return: Binary segmentation mask
    """
    image = io.imread(image_path)
    image_cleaned = dr.dullrazor(image)
    image_cleaned_rgb = cv.cvtColor(image_cleaned, cv.COLOR_BGR2RGB)  # Convert to RGB before processing
    inpainted_image, dca_mask = inpainting_dca(image_cleaned_rgb)
    five_masks = five_segmentation(inpainted_image)
    five_masks_cleaned = remove_small_parts_and_fill(five_masks)
    united_mask = union_mask(five_masks_cleaned)
    united_mask = np.array(united_mask, dtype=np.uint8)
    if dca_mask is not None:
        final_mask = postprocessing(united_mask, dca_mask)
    else:
        final_mask = united_mask
    final_mask_cleaned = remove_small_parts_and_fill([final_mask])[0]
    return final_mask_cleaned

def dice_score(mask1, mask2):
    """
    This function calculates the Dice score between two binary masks.
    :param mask1: First binary mask
    :param mask2: Second binary mask
    :return: Dice score
    """
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    result = 2 * intersection / union
    return result

# Create dataset
def create_dataset(images_path, masks, output_dir):
    """
    This function creates a dataset with resized images and masks.
    :param images_path: List of paths to input images
    :param masks: List of binary masks
    :param output_dir: Path to the output directory
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Iterate through images and masks
    for i, (image_path, mask) in enumerate(zip(images_path, masks)):
        mask = mask.astype(np.uint8)
        # Save the image and final mask
        mask_name = os.path.basename(image_path) + '_pred_mask.png'
        print(mask_name)
        io.imsave(os.path.join(output_dir, mask_name), mask)
        # Display progress

def resize_with_padding_binary_mask(mask, target_size):
    """
    This function resizes a binary mask with padding to the target size.
    :param mask: Binary mask
    :param target_size: Target size (width, height)
    :return: Resized binary mask with padding
    """
    height, width = mask.shape[:2]

    # Calculate the scaling ratio while maintaining the aspect ratio
    if height > width:
        ratio = target_size[0] / height
    else:
        ratio = target_size[1] / width

    # Resize the mask while maintaining the aspect ratio using the INTER_NEAREST method
    resized_mask = cv.resize(mask, None, fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)

    # Apply padding to obtain a size of 256x256
    pad_width = (target_size[1] - resized_mask.shape[1]) // 2
    pad_height = (target_size[0] - resized_mask.shape[0]) // 2
    padded_mask = cv.copyMakeBorder(resized_mask, pad_height, pad_height, pad_width, pad_width, cv.BORDER_CONSTANT, value=0)

    if padded_mask.shape[0] != target_size[0] or padded_mask.shape[1] != target_size[1]:
        padded_mask = cv.resize(padded_mask, target_size, interpolation=cv.INTER_NEAREST)
    return padded_mask

def compute_and_save_segmented_lesions(liste_chemins_images, output_dir_lesions, output_dir_masks):
    """
    This function computes and saves segmented lesions and their corresponding masks.
    :param liste_chemins_images: List of paths to input images
    :param output_dir_lesions: Path to the output directory for segmented lesions
    :param output_dir_masks: Path to the output directory for masks
    """
    masks_pred_resized = []
    segmented_lesions = []
    segmented_lesions_square = []

    # Use tqdm to display a progress bar
    for i in tqdm(range(0, len(liste_chemins_images))):
        image = io.imread(liste_chemins_images[i])
        image_cleaned = dr.dullrazor(image)
        image_cleaned = cv.cvtColor(image_cleaned, cv.COLOR_BGR2RGB)
        mask_pred = compute_segmentation(liste_chemins_images[i])
        # Normalize masks
        mask_pred_normalized = np.array(mask_pred.astype(float) / mask_pred.max()).astype(int)

        mask_pred = mask_pred_normalized
        lesions_r = image_cleaned[:,:,0] * mask_pred
        lesions_g = image_cleaned[:,:,1] * mask_pred
        lesions_b = image_cleaned[:,:,2] * mask_pred
        lesions = np.stack([lesions_r, lesions_g, lesions_b], axis=2)
        resized_padded_image = resize_with_padding_binary_mask(lesions, (256, 256))
        segmented_lesions.append(lesions)
        masks_pred_resized.append(mask_pred_normalized)
        segmented_lesions_square.append(resized_padded_image)
    create_dataset(liste_chemins_images, segmented_lesions_square, output_dir_lesions)
    create_dataset(liste_chemins_images, masks_pred_resized, output_dir_masks)

def compute_segmentation_for_given_mask(images_path, masks_path, output_dir):
    """
    This function computes and saves segmented lesions based on given masks.
    :param images_path: List of paths to input images
    :param masks_path: List of paths to input masks
    :param output_dir: Path to the output directory
    """
    masks = []
    for i in tqdm(range(0, len(images_path))):
        image = io.imread(images_path[i])
        mask_pred = io.imread(masks_path[i], as_gray=True)
        image_cleaned = dr.dullrazor(image)
        image_cleaned_rgb = cv.cvtColor(image_cleaned, cv.COLOR_BGR2RGB)  # Convert to RGB before processing
        mask_pred = np.array(mask_pred.astype(float) / mask_pred.max()).astype(int)
        lesions_r = image_cleaned_rgb[:,:,0] * mask_pred
        lesions_g = image_cleaned_rgb[:,:,1] * mask_pred
        lesions_b = image_cleaned_rgb[:,:,2] * mask_pred
        lesions = np.stack([lesions_r, lesions_g, lesions_b], axis=2)
        resized_padded_image = resize_with_padding_binary_mask(lesions, (256, 256))

        masks.append(resized_padded_image)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Iterate through images and masks
    for i, (image_path, mask) in enumerate(zip(images_path, masks)):
        mask = mask.astype(np.uint8)
        # Save the image and final mask
        mask_name = os.path.basename(image_path) + '_true_mask.png'
        io.imsave(os.path.join(output_dir, mask_name), mask)
        # Display progress
    return resized_padded_image
