import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import cv2
import numpy as np
import os
import re

# Regular expressions for email and address patterns
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
address_pattern = r'Address\s*:\s*\d+.*'

import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Regular expressions for email and address patterns
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
address_pattern = r'Address\s*:\s*\d+.*'

def clean_text(text, label):
    # If the text is from the 'personal' class, extract emails and addresses
    if label == 'personal':
        emails = re.findall(email_pattern, text)
        addresses = re.findall(address_pattern, text)

        text = re.sub(email_pattern, '', text)
        text = re.sub(address_pattern, '', text)

    # Lowercase the text
    text = text.lower()

    # Handle special cases: replace en-dashes, em-dashes, and curly quotes
    text = text.replace('–', ' ')  # Replace en-dash with space
    text = text.replace('—', ' ')  # Replace em-dash with space
    text = text.replace('’', "'")  # Normalize curly apostrophe
    text = text.replace('“', '"').replace('”', '"')  # Normalize curly quotes

    # Define custom punctuation, excluding apostrophes if you want to keep them
    custom_punctuation = string.punctuation.replace("'", "")

    # Remove punctuation (excluding apostrophes)
    text = re.sub(r"[{}]".format(custom_punctuation), '', text)

    # Remove isolated single letters (except 'a' and 'i')
    text = re.sub(r'\b[b-hj-z]\b', '', text)  # Matches single letters except 'a' and 'i'

    # Example: Standardize the GPAX format if needed
    # text = re.sub(r"(GPAX\s*:\s*)(\d+\.\d+)", r"GPAX: \2", text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Rejoin tokens into cleaned text
    cleaned_text = ' '.join(tokens)

    # Restore emails and addresses if 'personal'
    if label == 'personal':
        if emails:
            cleaned_text = ' '.join(emails) + ' ' + cleaned_text
        if addresses:
            cleaned_text = cleaned_text + ' ' + ' '.join(addresses)

    return cleaned_text.strip()

import re

def clean_text2(text):
    # Remove HTML-like symbols (e.g., "<", ">")
    text = re.sub(r"[<>]", "", text)

    # Remove special characters but keep important ones like periods, commas, parentheses, etc.
    text = re.sub(r"[^a-zA-Z0-9\s.,:()–-]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Optional: Normalize the dash symbol
    text = text.replace("–", "-")

    # Example: Standardize the GPAX format if needed
    text = re.sub(r"(GPAX\s*:\s*)(\d+\.\d+)", r"GPAX: \2", text)

    return text


def scale_and_preprocess(img, min_dim=800, max_dim=1600, scale_percent=150):
    """
    Scale the image up or down based on thresholds and apply preprocessing steps.

    Parameters:
    img (ndarray): Input image to be scaled and preprocessed.
    min_dim (int): Minimum dimension to trigger scaling up.
    max_dim (int): Maximum dimension to trigger scaling down.
    scale_percent (int): Percentage to scale up if below minimum threshold.

    Returns:
    ndarray: Preprocessed image.
    """
    # Get dimensions of the input image
    height, width = img.shape[:2]

    # Step 1: Scale the image
    if max(height, width) > max_dim:
        scaling_factor = max_dim / float(max(height, width))
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        # logger.info(f"Image scaled down to {img.shape[1]}x{img.shape[0]} for optimization.")
        print(f"Image scaled down to {img.shape[1]}x{img.shape[0]} for optimization.")
    elif min(height, width) < min_dim:
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        # logger.info(f"Image scaled up to {img.shape[1]}x{img.shape[0]} for better processing.")
        print(f"Image scaled up to {img.shape[1]}x{img.shape[0]} for better processing.")

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Invert the grayscale image
    inverted = cv2.bitwise_not(gray)

    # Step 4: Sharpen the image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(inverted, -1, sharpen_kernel)

    # Step 5: Dilation
    dilation_kernel = np.ones((2, 2), np.uint8)  # Adjust kernel size if needed
    dilated = cv2.dilate(sharpened, dilation_kernel, iterations=1)

    # Step 6: Erosion
    erosion_kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(dilated, erosion_kernel, iterations=1)

    # Step 7: Blur the image
    blurred = cv2.GaussianBlur(eroded, (1, 1), 0)

    # Step 8: Reconnect the image (bitwise not of blurred image)
    reconnected = cv2.bitwise_not(blurred)

    # logger.info(f"Preprocessing completed. Final image size: {reconnected.shape[1]}x{reconnected.shape[0]}")
    return reconnected



if __name__ == '__main__':
    inputs = "e Education and activities Nakhon Sawan Rajabhat University (NSRU), Nakhon Sawan < May 2019 – June 2023 > (Bachelor’s Degree in Political Science program in Politics and Government) GPAX : 3.56 (2nd class honor)"
    cleaned_text = clean_text(inputs, label=None)
    print("Before cleaned : ", inputs)
    print("After cleaned : ", cleaned_text)