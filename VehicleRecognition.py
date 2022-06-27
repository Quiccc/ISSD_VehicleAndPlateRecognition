import glob
import os
import numpy as np
import cv2

net = cv2.dnn.readNet('yolov5s.onnx') # Insert trained YOLOv5 model


# Convert the image to a square format
def format_yolov5(frame):

    row, col, _ = frame.shape 
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

for image_path in glob.glob('dataset/images/*.jpg'):
    image = cv2.imread(image_path)  
    input_image = format_yolov5(image)
    blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True) 
    net.setInput(blob)
    predictions = net.forward()


    class_ids = []
    confidences = []
    boxes = []

    output_data = predictions[0]

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(25200):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    class_list = []
    with open("dataset/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    for i in range(len(result_class_ids)):

        box = result_boxes[i] # [left, top, width, height] edges of the bounding box
        class_id = result_class_ids[i] # class id of the object

        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1) #Surround the identified object in a rectangle
        cv2.putText(image, class_list[0], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0)) #Add the class name to the rectangle

    image = cv2.resize(image, (1600,900))
    cv2.imshow("output", image)
    cv2.waitKey()
    cv2.destroyAllWindows()