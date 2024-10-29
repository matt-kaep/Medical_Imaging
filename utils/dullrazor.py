# Import the necessary libraries
import cv2
import matplotlib.pyplot as plt

# Define the dullrazor function
def dullrazor(image):
    """
    This function takes an input image and processes it to remove small, dark details such as hair.

    Parameters:
    image (numpy array): The input image in RGB format.

    Returns:
    dst (numpy array): The processed image in RGB format.
    """

    # Convert the image to grayscale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply a black hat filter to the grayscale image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # Apply a Gaussian blur to the black hat image
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)

    # Apply binary thresholding to the blurred image to create a mask
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)

    # Use the mask to replace pixels in the original image
    dst = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)

    # Convert the image back to RGB format
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    # Return the processed image
    return dst


