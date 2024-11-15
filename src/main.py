from obj_inference import obj_det_inference
from text_cls_inference import cls_inference
from ocr import run_ocr
from ultralytics import YOLO
from pdf2image import convert_from_path
import numpy as np
# import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text
import cv2
# from ingest.obs_ingest import  download_from_obs


class Pipeline:

    def __init__(self, file_path, job_description):
        self.file_path = file_path
        self.job_description = job_description

    ###### Retrieve CV as pdf file from Object Storage Service ########
        # file path or file
        # self.file = ....

    ###################################################################
        
    @staticmethod
    def sim_score(jd_text, exp_text):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([jd_text, exp_text])
        
        # Calculate cosine similarity
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return score[0][0]
        

    # def main(self):
    #     # Object detection
    #     model = YOLO('./model/best.pt')

    #     # convert pdf to images
    #     images = convert_from_path(self.file_path)
    #     exp_texts = []
    #     for i, image in enumerate(images):
    #         print(f'Start Object Detection on Page {i+1}')
    #         image = np.array(image)
    #         crop_images = obj_det_inference(model, image)

    #         x1, y1, x2, y2 = crop_images['boxes'][idx]
    #         # perform ocr
    #         print('Perform OCR')
    #         for idx,crop in enumerate(crop_images['result']):
    #             # print(crop)
    #             # print(type(crop))
    #             # cv2.imwrite('result.jpg', crop)
    #             text = run_ocr(crop, lang='eng')


    #             '''
    #             class list:
    #                 'certification',
    #                 'education',
    #                 'experience',
    #                 'others',
    #                 'personal',
    #                 'preface',
    #                 'skill'
    #             '''

    #             res_class = cls_inference(text)

    #             if res_class == 'experience':
    #                 exp_texts.append(text)
    #                 cv2.putText(image, res_class, )

    #     res_text = ' '.join(exp_texts)
                
    #     return res_text
    
    def _main(self):
        # Object detection
        model = YOLO('./model/best.pt')

        # Convert PDF to images
        images = convert_from_path(self.file_path)
        exp_texts = []
        for i, image in enumerate(images):
            print(f'Start Object Detection on Page {i+1}')
            image = np.array(image)
            crop_images = obj_det_inference(model, image)
            
            # Perform OCR and draw boxes
            print('Perform OCR')
            for idx, crop in enumerate(crop_images['result']):
                # Get the bounding box coordinates
                x1, y1, x2, y2 = crop_images['boxes'][idx]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Perform OCR on the cropped image
                text = run_ocr(crop, lang='eng')

                # Classify the text
                res_class = cls_inference(text)

                # Draw the bounding box on the original image
                color = (0, 255, 0)  # Green color in BGR
                thickness = 2
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                # Put the classification text at the top-left corner of the box
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                print('class result: ',res_class)
                cv2.putText(image, res_class[0], (x1, y1 - 10), font, font_scale, color, thickness)

                if res_class == 'experience':
                    exp_texts.append(text)
            
            # Optionally, save or display the image with drawn boxes
            cv2.imwrite(f'output2_page_{i+1}.jpg', image)

        res_text = ' '.join(exp_texts)
        return res_text
    
    # Function for push data to Database
    # def push_data(self):
    #   pass 




if __name__ == "__main__":

    file_path = './test/cv/21.pdf'
    
    get_text = Pipeline(file_path)
    res = get_text._main()
    print('-----------Process OCR Done---------')
    # # print(res)

    # jd = ""

    # score = get_text.sim_score(res)

    with open('finance.txt', 'r', encoding='utf-8') as file:
        content = file.read()  # Reads the entire content of the file

    jd_text = clean_text(content, label=None)
    print('Job Description: ', jd_text)
    print()
    print('-------------------------------------')
    print()
    cleaned_res = clean_text(res, label=None)
    print('CV : ', cleaned_res)

    print("---------Perform Similarity score calculation---------")
    score = get_text.sim_score(cleaned_res, jd_text)
    print("Similarity Score: ", score)

