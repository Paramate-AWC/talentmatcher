import os
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO



def obj_det_inference(model, image, show=None):
    results = model.predict(image, conf=0.75)

    # Load the image using cv2
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

    # Prepare to store cropped images
    cropped_images = []
    box_list = []
    # Draw the bounding boxes on the image
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates and convert to integer
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()  # Confidence score

            # Draw rectangle and confidence score on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cropped_img = img[y1:y2, x1:x2]
            cropped_images.append(cropped_img)
            box_coor = (x1,y1,x2,y2)
            box_list.append(box_coor)

    if show:
        cv2.imwrite('detected_im.jpg', img)
        cv2.imshow('detected', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return dict({'result':cropped_images,
                 'boxes':box_list})

if __name__ == "__main__":

    model = YOLO('./model/best.pt')

    image_path = '../test/example_im.png'
    img = cv2.imread(image_path)

    print('start inference')
    result = obj_det_inference(model, image=img, show=True)
    print(result['boxes'])

