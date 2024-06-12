import cv2
import numpy as np

class Yolo:

    # Initialize Yolo object with configuration directory, confidence threshold and NMS threshold
    def __init__(self, config_dir="./Yolo/yolo_tiny_configs", confidence_threshold=0.0, nms_threshold=0.0):
        self.config_dir = config_dir
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        # Load YOLO model
        self.net = self._load_model()
        # Load class names
        self.classes = self._load_classes()
        # Get output layer names
        self.output_layers = self._get_output_layers()

    # Load  YOLO model from configuration files
    def _load_model(self):
        # Load model weights
        weights_path = f"{self.config_dir}/yolov3-tiny.weights"
        # Load model config
        config_path = f"{self.config_dir}/yolov3-tiny.cfg"
        # Load model
        net = cv2.dnn.readNet(weights_path, config_path)
        return net

    # Load class names from coco.names file
    def _load_classes(self):
        classes_path = f"{self.config_dir}/coco.names"
        with open(classes_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]  # Read and strip each line
        return classes

    # Get the output layers of the YOLO model
    def _get_output_layers(self):
        # Get all layer names
        layer_names = self.net.getLayerNames()
        # Get unconnected layers
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    # Perform object detection on an image
    def transform(self, img_path):
        # Load image
        image = self._load_image(img_path)
        # Prepare image
        blob = self._prepare_image(image)
        # Set the input
        self.net.setInput(blob)
        # Perform forward pass
        outs = self.net.forward(self.output_layers)
        # Process detections
        class_ids, confidences, boxes = self._process_detections(outs, image.shape[:2])
        # Apply non-max suppression and return results
        return self._apply_nms(boxes, confidences, class_ids)

    # Perform object detection on an image and draw bounding boxes on the detected objects.
    def transform_draw(self, img_path_in, display=False):

        # Read input image
        image = cv2.imread(img_path_in)
        # Perform detection
        class_ids, confidences, boxes = self.transform(img_path_in)

        # Draw bounding boxes on the image
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            # Get class label
            label = str(self.classes[class_ids[i]])
            # Set bounding box color
            color = (0, 255, 0)
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # Draw label
            cv2.putText(image, label, (x, y + 30), font, 3, color, 3)

        # In case you want to display the image
        if display:
            # Display image
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image with bounding boxes
        cv2.imwrite(img_path_in, image)

        # Return results
        return class_ids, confidences, boxes

    # Load image from specified path
    def _load_image(self, img_path):

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            # Raise error if image not found
            raise FileNotFoundError(f"No image found at {img_path}")
        return image

    # Prepare image for the model
    def _prepare_image(self, image):
        # Create blob
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        return blob

    # Process  detections from the YOLO model
    def _process_detections(self, outs, image_shape):
        # Get image dimensions
        height, width = image_shape
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                # Get scores for all classes
                scores = detection[5:]
                # Get class ID with the highest score
                class_id = np.argmax(scores)
                # Get the confidence for the highest score
                confidence = scores[class_id]
                if confidence >= self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # Append bounding box
                    boxes.append([x, y, w, h])
                    # Append confidence
                    confidences.append(float(confidence))
                    # Append class ID
                    class_ids.append(class_id)

        return class_ids, confidences, boxes

    # Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes with low confidence
    def _apply_nms(self, boxes, confidences, class_ids):

        # Apply NMS
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        # Prune class IDs
        class_ids_pruned = [class_ids[i] for i in indexes]
        # Prune confidences
        confidences_pruned = [confidences[i] for i in indexes]
        # Prune boxes
        boxes_pruned = [boxes[i] for i in indexes]

        # Sort the pruned results by confidence in descending order
        # Get sorted indices
        sorted_indices = np.argsort(confidences_pruned)[::-1]
        # Sort class IDs
        class_ids_sorted = [class_ids_pruned[i] for i in sorted_indices]
        # Sort confidences
        confidences_sorted = [confidences_pruned[i] for i in sorted_indices]
        # Sort boxes
        boxes_sorted = [boxes_pruned[i] for i in sorted_indices]

        # Return sorted results
        return class_ids_sorted, confidences_sorted, boxes_sorted

if __name__ == "__main__":
    # Initialize YOLO object
    yolo = Yolo(config_dir="./yolo_tiny_configs")
