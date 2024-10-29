# Skin Lesion Segmentation and Classification  
### Matthieu Kaeppelin, Télécom 2024  

### You can have access to a detailed and illustrated report in the final_report.ipynb notebook !
---  
   
## Table of Contents  
1. [Introduction](#introduction)  
2. [Project Overview](#project-overview)  
3. [Methodology](#methodology)  
   - [Data Preprocessing](#data-preprocessing)  
   - [Segmentation](#segmentation)  
   - [Feature Extraction](#feature-extraction)  
   - [Classification](#classification)  
4. [Results](#results)  
5. [Conclusion](#conclusion)  
6. [References](#references)  
   
---  
   
## Introduction  
Skin cancer is a significant public health issue, affecting millions globally. Early detection is crucial for effective treatment and improved survival rates. This project leverages Deep Learning and ABCD (Asymmetry, Border irregularity, Color variation, Diameter) feature extraction methods to classify and segment skin lesions. The goal is to develop a robust automated system for early skin cancer detection, comparing traditional ABCD analysis with advanced CNN-based methods.  
   
---  
   
## Project Overview  
This project is structured into several key steps, each focusing on a crucial aspect of the skin lesion analysis pipeline. The main phases include data preprocessing, segmentation, feature extraction, and classification.  
   
---  
   
## Methodology  
   
### Data Preprocessing  
The preprocessing phase aims to clean and standardize the image data to enhance the performance of the segmentation and classification models. The key steps involved are:  
   
1. **Hair Removal:** The DullRazor algorithm is used to detect and remove hair from the images. This involves:  
   - Converting the image to grayscale.  
   - Applying a black hat filter.  
   - Using Gaussian Blur and binary thresholding.  
   - Inpainting the image to fill in the removed hair regions.  
   
2. **Artifact Removal:** The Dark Artefact Removal Algorithm targets black corners and other unwanted artifacts in the images. This process includes:  
   - Identifying and masking dark artifacts.  
   - Inpainting the masked regions to ensure smooth and artifact-free images.  
   
3. **Resizing and Padding:** Images are resized to a uniform size (256x256 pixels) using zero-padding to maintain aspect ratios and prepare them for further analysis.  
   
### Segmentation  
Segmentation aims to isolate the lesion from the surrounding skin. The methodology includes:  
   
1. **Multiple Segmentation Masks:** Using a combination of global thresholding, dynamic thresholding, and color clustering techniques to generate multiple segmentation masks.  
2. **Mask Selection Rules:** Implementing rules to select the most accurate mask:  
   - Reject masks growing into image borders or without detected regions.  
   - Discard fragments at image borders and the smallest masks.  
   - Choose masks with consistent areas compared to others.  
3. **Union Mask:** Combining the selected masks to form a final segmentation mask.  
4. **Post-Processing:** Applying the Dark Artefact Removal mask to clean the segmented regions.  
   
### Feature Extraction  
Feature extraction focuses on deriving meaningful attributes from the segmented lesions using the ABCD methodology:  
   
1. **Total Area:** Calculating the total area of the lesion.  
2. **Perimeter and Compactness Index:** Measuring the perimeter and calculating the compactness index (ratio of squared perimeter to area).  
3. **Diameter:** Determining the maximum dimension of the lesion's bounding box.  
4. **Color Variegation:** Assessing color variation within the lesion using the standard deviation of RGB channels.  
5. **Texture Features:** Extracting texture features (correlation, homogeneity, energy, contrast) using the Gray-Level Co-occurrence Matrix (GLCM).  
   
### Classification  
Two classification approaches are compared: Random Forest using ABCD features and Convolutional Neural Networks (CNN):  
   
1. **Random Forest Classifier:**  
   - **Data Preparation:** One-hot encoding of categorical metadata and balancing the dataset through oversampling.  
   - **Model Training:** Training a Random Forest classifier with the extracted features.  
   - **Evaluation:** Assessing model
