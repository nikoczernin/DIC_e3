# class for Yolo

import cv2
import numpy as np

class Yolo:

    def __init__(self):
        # Load YOLO network
        # The function cv2.dnn.readNet() loads the YOLO model weights and configuration file
        # "yolov3-tiny.weights" contains the trained weights for the YOLOv3-tiny model
        # "yolov3-tiny.cfg" contains the configuration parameters for the YOLOv3-tiny model
        yolo_dir = "yolo_tiny_configs"
        self.net = cv2.dnn.readNet(f"{yolo_dir}/yolov3-tiny.weights", f"{yolo_dir}/yolov3-tiny.cfg")

        # Load the COCO class labels
        # The file "coco.names" contains the names of the classes YOLO is trained to detect (e.g., person, bicycle, car, etc.)
        with open(f"{yolo_dir}/coco.names", "r") as f:
            # Read each line in the file and strip any whitespace characters
            self.classes = [line.strip() for line in f.readlines()]

        # Get the names of the output layers
        # net.getLayerNames() returns the names of all layers in the YOLO network
        self.layer_names = self.net.getLayerNames()

        # net.getUnconnectedOutLayers() returns the indexes of the output layers
        # We subtract 1 because OpenCV's layer indexing starts at 1, but Python's list indexing starts at 0
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def transform(self, img_path):
        # Load the input image
        # cv2.imread() reads the image from the file and returns it as a NumPy array
        image = cv2.imread(img_path)
        if image is None:
            raise Exception("No image found!")

        # Get the dimensions of the image
        height, width, channels = image.shape

        # Prepare the image for YOLO
        # cv2.dnn.blobFromImage() creates a 4D blob from the image
        # The blob is used as input to the YOLO network
        # Parameters:
        # - image: the input image
        # - 0.00392: scale factor to normalize pixel values (1/255)
        # - (416, 416): spatial size for the network input (YOLO expects 416x416 images)
        # - (0, 0, 0): mean values to subtract from each channel (no mean subtraction here)
        # - True: swap the red and blue channels (OpenCV uses BGR, YOLO uses RGB)
        # - crop: don't crop the image
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # Set the input blob for the YOLO network
        self.net.setInput(blob)

        # Perform the forward pass to get the network output
        # The output is a list of 2 np arrays, each corresponding to an output layer
        # each array has shape (x, y), where x is the number of detections
        # the other dimension are the detection values:
        # values 1-4 are the bounding box coordinates (center x, center y, width, height)
        # value 5 is the confidence score
        # the following 80 values are the probabilities of the 80 classes
        outs = self.net.forward(self.output_layers)
        # yolo has two output layers. each is responsible for detecting objects at different scales,
        # which helps to detect large and small objects within the same pic

        # Initialize lists to hold detection data
        class_ids = []  # List to hold the class IDs of detected objects
        confidences = []  # List to hold the confidence scores of detected objects
        boxes = []  # List to hold the bounding box coordinates of detected objects

        # Process the detections
        # Iterate over each output layer
        for out in outs:
            # Iterate over each detection in the output layer
            for detection in out:
                # The first 5 elements are the bounding box coordinates and confidence score
                # The remaining elements are the class probabilities
                scores = detection[5:]
                # Get the class ID with the highest probability
                # it is the location index of the class/object that has most probably been detected
                class_id = np.argmax(scores)
                # Get the highest probability (confidence)
                confidence = scores[class_id]
                # Filter out weak detections by ensuring the confidence is greater than a threshold (e.g., 0.5)
                if confidence > 0.5:
                    # Scale the bounding box coordinates back to the size of the original image
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Calculate the top-left corner of the bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # Append the bounding box coordinates, confidence score, and class ID to the respective lists
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression to remove redundant overlapping boxes with lower confidences
        # Non-max suppression ensures that only the most confident box for each object is kept
        # cv2.dnn.NMSBoxes() returns the indices of the kept boxes
        # Non-Maximum Suppression (NMS)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # return the pruned results
        class_ids_pruned = [class_ids[i] for i,_ in enumerate(class_ids) if i in indexes]
        confidences_pruned = [confidences[i] for i,_ in enumerate(confidences) if i in indexes]
        boxes_pruned = [boxes[i] for i,_ in enumerate(boxes) if i in indexes]

        return class_ids_pruned, confidences_pruned, boxes_pruned

    def transform_draw(self, img_path_in, display=False):
        image = cv2.imread(img_path_in)
        class_ids, confidences, boxes = self.transform(img_path_in)
        print(class_ids)

        # Draw bounding boxes on the image
        # Define the font for the text
        font = cv2.FONT_HERSHEY_PLAIN
        # Iterate over the kept boxes
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            # Get the class label for the detected object
            label = str(self.classes[class_ids[i]])
            # Define the color for the bounding box (green in this case)
            color = (0, 255, 0)
            # Draw the bounding box on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # Put the class label text near the top-left corner of the bounding box
            cv2.putText(image, label, (x, y + 30), font, 3, color, 3)

        if display:
            # Optionally display the output image (useful for local testing)
            # cv2.imshow() displays the image in a window
            cv2.imshow("Image", image)

        return image


    def save_image(self, image, img_path_out):
        # does not work on AWS i think
        # Save the output image with the drawn bounding boxes
        cv2.imwrite(img_path_out, image)




def main():
    print("Testing Yolo on an image!")
    yolo = Yolo()
    yolo.transform_draw("input_folder/000000000674.jpg", "test_output.jpg")

if __name__ == '__main__':
    main()