from flask import Flask, request, jsonify
import numpy as np
import time
import cv2
import os
import classifier

app = Flask(__name__)

yolo_path = "yolo-coco"
labelsPath = os.path.join(yolo_path, "coco.names")
weightsPath = 'yolov3.weights'
# weightsPath = os.path.join(yolo_path, "yolov3.weights")
configPath = os.path.join(yolo_path, "yolov3.cfg")
conf = 0.5
thr = 0.6
car_color_classifier = classifier.Classifier()

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


@app.route('/classify_car_image', methods=['POST'])
def classify_car_image():
    data = request.get_json()
    image_path = data['image_path']
    image = cv2.imread(image_path)
    results = process_image(image)
    
    return jsonify(results)

def process_image(image):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    results = []  # List to store results
    
    boxes = []
    confidences = []
    classIDs = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, thr)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            result = {}
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in np.random.randint(0, 255, size=(3,))]

            if classIDs[i] == 2:
                car_image = image[max(y,0):y + h, max(x,0):x + w]
                car_result = car_color_classifier.predict(car_image)
                result['make'] = car_result[0]['make']
                result['model'] = car_result[0]['model']
            else:
                result['make'] = 'Unknown'
                result['model'] = 'Unknown'

            results.append(result)

    return results

if __name__ == '__main__':
    app.run(debug=True)
