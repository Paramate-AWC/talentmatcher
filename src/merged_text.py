# from obj_inference import obj_det_inference
# from ocr_huawei import HuaweiOCRClient
# import cv2
# from ultralytics import YOLO
# from huawei_boxes import draw_huawei_box

# model = YOLO('./model/best.pt')

# image_path = './test/5_page_1.jpg'
# img = cv2.imread(image_path)
# layout = obj_det_inference(model, image=img, show=False)
# layout = layout['boxes']
# client = HuaweiOCRClient()
# word_container = client.perform_ocr("./test/5_page_1.jpg")





# def get_bounding_box(location):
#     x_coords = [point[0] for point in location]
#     y_coords = [point[1] for point in location]
#     xmin = min(x_coords)
#     ymin = min(y_coords)
#     xmax = max(x_coords)
#     ymax = max(y_coords)
#     return xmin, ymin, xmax, ymax


# def overlaps(boxA, boxB, wordbox_area, threshold=70):
#     xmin1, ymin1, xmax1, ymax1 = boxA
#     xmin2, ymin2, xmax2, ymax2 = boxB

#     # หาค่าพิกัดของพื้นที่ที่ซ้อนทับกัน
#     x_left = max(xmin1, xmin2)
#     y_top = max(ymin1, ymin2)
#     x_right = min(xmax1, xmax2)
#     y_bottom = min(ymax1, ymax2)

#     # ตรวจสอบว่ามีการซ้อนทับกันหรือไม่
#     if x_right < x_left or y_bottom < y_top:
#         print("Check overlap")
#         return False, 0.0

#     # คำนวณพื้นที่ที่ซ้อนทับกัน
#     overlap_area = (x_right - x_left) * (y_bottom - y_top)
    

#     # คำนวณพื้นที่ของกล่องที่สอง (boxB)
#     # boxB_area = (xmax2 - xmin2) * (ymax2 - ymin2)
#     # wordbox_area

#     if wordbox_area == 0:
#         return False, 0.0

#     # คำนวณเปอร์เซ็นต์ของพื้นที่ที่ซ้อนทับ
#     percentage_overlap = (overlap_area / wordbox_area) * 100
#     print("Percent Overlap",percentage_overlap)

#     if percentage_overlap < threshold:
#         print("False")
#         return False, percentage_overlap 
    

#     # if xmax1 < xmin2 or xmax2 < xmin1:
#     #     return False
#     # if ymax1 < ymin2 or ymax2 < ymin1:
#     #     return False
    

#     return True, percentage_overlap 





# # Initialize yolo_texts
# layout_texts = ['' for _ in range(len(layout))]

# # For each word_box in word_container
# percent_overlap_list = []
# for word_box in word_container:
#     location = word_box['location']
#     words = word_box['words']
#     word_bbox = get_bounding_box(location)
#     print("WORD BOX: ",word_bbox)

#     xmin2, ymin2, xmax2, ymax2 = word_bbox
#     wordbox_area = (xmax2 - xmin2) * (ymax2 - ymin2)
#     check_in_layout = False
#     for idx, yolo_box in enumerate(layout):
#         print("YOLO BOX: ",yolo_box) 

#         res_overlap = overlaps(yolo_box,word_bbox, wordbox_area)
#         percent_overlap_list.append(res_overlap[1])
#         x1, y1, x2, y2 = yolo_box
#         if res_overlap[0]:
#             # Append the words to yolo_texts[idx]
           
#             if layout_texts[idx]:
#                 layout_texts[idx] += ' ' + words
#             else:
#                 layout_texts[idx] = words
            
#             check_in_layout = True
#             break

         

#         if  res_overlap[1] > 5 and res_overlap[1] < 70:
#             cv2.putText(img, f'{res_overlap}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#         # else:
#         #     percent_overlap_list.append(res_overlap[1])
#     print("check in layout: ",check_in_layout)
#     if check_in_layout==False:
#         print(words)
# # Now, yolo_texts contains the merged text per yolo_box
# # result text after layout
# texts = []
# for idx, text in enumerate(layout_texts):
#     print(f"YOLO Box {idx}:")
#     print(text)
#     texts.append(text)
#     print()

# # print(percent_overlap_list)

# # result raw text from ocr
# try:
#     word_container = client.perform_ocr("./test/5_page_1.jpg")
    
#     # Convert the word blocks to a plain text format
#     raw_text = "\n".join(block["words"] for block in word_container if "words" in block)
    
#     # Print the result
#     # print(result_text)
    
# except Exception as e:
#     # logging.error(e)
#     print(e)

# # final result to construct API
# res = {
#     'result':
#         {'text_layout':texts,
#          'boxes':layout,
#          'raw_text':raw_text
#          }
#          }

# # print(word_container)
# # Draw boxes
# # if draw
# img = draw_huawei_box(word_container, img)

# for box in layout:
#     # Get bounding box coordinates and convert to integer
#     x1, y1, x2, y2 = box
#     # conf = box.conf[0].item()  # Confidence score

#     # Draw rectangle and confidence score on the image
#     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     # cv2.putText(img, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# cv2.imwrite('detected_percent_overlap.jpg', img)
# cv2.imshow('detected_percent_overlap', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from obj_inference import obj_det_inference
from ocr_huawei import HuaweiOCRClient
import cv2
from ultralytics import YOLO
from huawei_boxes import draw_huawei_box
from text_cls_inference import cls_inference

class OcrProcessor:
    def __init__(self, model_path, image_path):
        self.model = YOLO(model_path)
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.client = HuaweiOCRClient()
        self.layout = None
        self.layout_texts = []
        self.word_container = None
        self.raw_text = ''
        self.result = {}

    def get_bounding_box(self, location):
        x_coords = [point[0] for point in location]
        y_coords = [point[1] for point in location]
        xmin = min(x_coords)
        ymin = min(y_coords)
        xmax = max(x_coords)
        ymax = max(y_coords)
        return xmin, ymin, xmax, ymax

    def overlaps(self, boxA, boxB, wordbox_area, threshold=70):
        xmin1, ymin1, xmax1, ymax1 = boxA
        xmin2, ymin2, xmax2, ymax2 = boxB

        # Find the coordinates of the intersection rectangle
        x_left = max(xmin1, xmin2)
        y_top = max(ymin1, ymin2)
        x_right = min(xmax1, xmax2)
        y_bottom = min(ymax1, ymax2)

        # Check if there is an overlap
        if x_right < x_left or y_bottom < y_top:
            print("Check overlap")
            return False, 0.0

        # Calculate the area of overlap
        overlap_area = (x_right - x_left) * (y_bottom - y_top)

        if wordbox_area == 0:
            return False, 0.0

        # Calculate the percentage of overlap
        percentage_overlap = (overlap_area / wordbox_area) * 100
        print("Percent Overlap", percentage_overlap)

        if percentage_overlap < threshold:
            print("False")
            return False, percentage_overlap

        return True, percentage_overlap

    def process_image(self):
        # Perform object detection
        layout = obj_det_inference(self.model, image=self.img, show=False)
        self.layout = layout['boxes']
        self.layout_texts = ['' for _ in range(len(self.layout))]

        # Perform OCR
        self.word_container = self.client.perform_ocr(self.image_path)

        # Initialize list to store overlap percentages
        percent_overlap_list = []
        for word_box in self.word_container:
            location = word_box['location']
            words = word_box['words']
            word_bbox = self.get_bounding_box(location)
            print("WORD BOX: ", word_bbox)

            xmin2, ymin2, xmax2, ymax2 = word_bbox
            wordbox_area = (xmax2 - xmin2) * (ymax2 - ymin2)
            check_in_layout = False

            for idx, yolo_box in enumerate(self.layout):
                print("YOLO BOX: ", yolo_box)

                res_overlap = self.overlaps(yolo_box, word_bbox, wordbox_area)
                percent_overlap_list.append(res_overlap[1])
                x1, y1, x2, y2 = yolo_box

                if res_overlap[0]:
                    # Append the words to the corresponding text layout
                    if self.layout_texts[idx]:
                        self.layout_texts[idx] += ' ' + words
                    else:
                        self.layout_texts[idx] = words

                    check_in_layout = True
                    break

                if 5 < res_overlap[1] < 70:
                    cv2.putText(
                        self.img,
                        f'{res_overlap}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )
            print("Check in layout: ", check_in_layout)
            if not check_in_layout:
                print(words)

        # Collect texts for each YOLO box
        texts = []
        classes = []
        for idx, text in enumerate(self.layout_texts):
            # print(f"YOLO Box {idx}:")
            # print(text)
            # perform text classification
            text_class = cls_inference(text) # result as array
            text_class = text_class[0]
            classes.append(text_class)
            texts.append(text)
            print()

        # Get raw text from OCR
        try:
            word_container = self.client.perform_ocr(self.image_path)
            self.raw_text = "\n".join(
                block["words"] for block in word_container if "words" in block
            )
        except Exception as e:
            print(e)

        # Construct the result dictionary
        self.result = {
            'result': {
                'text_layout': texts,
                'layout_classes': classes,
                'boxes': self.layout,
                'raw_text': self.raw_text
            }
        }

    def draw_boxes(self, output_image_path='detected_percent_overlap.jpg'):
        # Draw OCR boxes on the image
        img_with_boxes = draw_huawei_box(self.word_container, self.img.copy())

        # Draw YOLO boxes
        for box in self.layout:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imwrite(output_image_path, img_with_boxes)
        cv2.imshow('Detected Percent Overlap', img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    processor = OcrProcessor('./model/best.pt', './test/5_page_1.jpg')
    processor.process_image()
    processor.draw_boxes()
    print(processor.result)

