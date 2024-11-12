import pytesseract
import cv2
import numpy as np
import os
import re
from utils import scale_and_preprocess
pytesseract.pytesseract.tesseract_cmd = r'D:\Users\paramate.p\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


def run_ocr(image, lang):
    
    config = f'--oem 3 --psm 6 --dpi 300 -l {lang}'
    text = pytesseract.image_to_string(image, config=config)
    
    return text


if __name__ == "__main__":

    img_path = './test/5_page_1.jpg'
    img = cv2.imread(img_path)
    res = run_ocr(img, 'eng+th')
    print(res)