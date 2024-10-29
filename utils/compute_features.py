import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage import feature, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, \
                            recall_score, accuracy_score, classification_report
import cv2 as cv
import os
import DarkArtefactRemoval as dca
import dullrazor as dr
import utils.segmentation_and_preprocessing as sp
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops



def calculate_features(images, masks, lesions, plot_limit=200, affichage=False):
    """
    This function calculates and extracts various features of lesions from images, masks, and the lesions themselves.

    Parameters
    ----------
    images : list
        List of paths to the images to analyze.
    masks : list
        List of paths to the masks corresponding to the images.
    lesions : list
        List of paths to the lesions corresponding to the images.
    plot_limit : int, optional
        Maximum number of images for which to display the calculated features. The default is 200.
    affichage : bool, optional
        If True, displays the calculated features for each image. The default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the calculated features for each image.

    """
    feature_list = []  # List to store the features of each image

    for idx, (image, mask, lesion) in enumerate(tqdm(zip(images, masks, lesions))):
        # Loading the image, mask, and lesion
        image_path = image
        image = cv2.imread(image)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        lesion = cv2.imread(lesion)

        # Calculating the total area of the lesion
        area_total = np.sum(mask)

        # Calculating the perimeter and compactness index
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            index = np.argmax([len(c) for c in contours])
            lesion_region = contours[index]
            perimeter = cv2.arcLength(lesion_region, True)
            compact_index = (perimeter ** 2) / (4 * np.pi * area_total)

            # Calculating the diameter of the lesion
            x, y, w, h = cv2.boundingRect(lesion_region)
            diameter = max(w, h)

            if affichage:
                print('\n-- DIAMETER --')
                print('Diameter: {:.3f}'.format(diameter))

        except ValueError:
            #print(f"No contours found for image {image_path}. Setting compact index & diameter to NaN.")
            compact_index = np.nan
            diameter = np.nan

        if affichage:
            if idx < plot_limit:
                print('\n-- BORDER IRREGULARITY --')
                print('Compact Index: {:.3f}'.format(compact_index))

        # Calculating color variegation
        lesion_r = lesion[:, :, 0]
        lesion_g = lesion[:, :, 1]
        lesion_b = lesion[:, :, 2]

        C_r = np.std(lesion_r) / np.max(lesion_r)
        C_g = np.std(lesion_g) / np.max(lesion_g)
        C_b = np.std(lesion_b) / np.max(lesion_b)

        if affichage:
            print('\n-- COLOR VARIEGATION --')
            print('Red Std Deviation: {:.3f}'.format(C_r))
            print('Green Std Deviation: {:.3f}'.format(C_g))
            print('Blue Std Deviation: {:.3f}'.format(C_b))

        # Converting the lesion region to grayscale
        gray_lesion = rgb2gray(lesion)

        # Computing the grey-level co-occurrence matrix
        glcm = feature.graycomatrix(image=img_as_ubyte(gray_lesion), distances=[1],
                                    angles=[0, np.pi/4, np.pi/2, np.pi * 3/2],
                                    symmetric=True, normed=True)

        # Computing texture features
        correlation = np.mean(feature.graycoprops(glcm, prop='correlation'))
        homogeneity = np.mean(feature.graycoprops(glcm, prop='homogeneity'))
        energy = np.mean(feature.graycoprops(glcm, prop='energy'))
        contrast = np.mean(feature.graycoprops(glcm, prop='contrast'))

        if affichage:
            if idx < plot_limit:
                print('\n-- TEXTURE --')
                print('Correlation: {:.3f}'.format(correlation))
                print('Homogeneity: {:.3f}'.format(homogeneity))
                print('Energy: {:.3f}'.format(energy))
                print('Contrast: {:.3f}'.format(contrast))

        if affichage:
            if compact_index == np.nan:
                # Draw the contour on the image and display it
                cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Draw all contours in green
                plt.imshow(image)
                plt.axis('off')
                plt.show()

        # Storing features in a dictionary
        features = {
            'ID': os.path.basename(image_path).split('.')[0],
            'compact_index': compact_index,
            'C_r': C_r,
            'C_g': C_g,
            'C_b': C_b,
            'diameter': diameter,
            'correlation': correlation,
            'homogeneity': homogeneity,
            'energy': energy,
            'contrast': contrast
        }

        feature_list.append(features)

    # Converting the list of dictionaries to a DataFrame
    df = pd.DataFrame(feature_list)

    return df
