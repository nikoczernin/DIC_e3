import cv2
import numpy as np

class Yolo:
    def __init__(self, config_dir="yolo_tiny_configs", confidence_threshold=0.2, nms_threshold=0.2):
        """
        Initialize the Yolo object with configuration directory, confidence threshold, and NMS threshold.
        Args:
            config_dir (str): Directory containing YOLO configuration files.
            confidence_threshold (float): Confidence threshold for detections.
            nms_threshold (float): Non-max suppression threshold.
        """
        self.config_dir = config_dir
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = self._load_model()  # Load the YOLO model
        self.classes = self._load_classes()  # Load class names
        self.output_layers = self._get_output_layers()  # Get output layer names

    def _load_model(self):
        """
        Load the YOLO model from the configuration files.
        Returns:
            net (cv2.dnn_Net): Loaded YOLO model.
        """
        weights_path = f"{self.config_dir}/yolov3-tiny.weights"
        config_path = f"{self.config_dir}/yolov3-tiny.cfg"
        net = cv2.dnn.readNet(weights_path, config_path)
        return net

    def _load_classes(self):
        """
        Load the class names from the coco.names file.
        Returns:
            classes (list): List of class names.
        """
        classes_path = f"{self.config_dir}/coco.names"
        with open(classes_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def _get_output_layers(self):
        """
        Get the output layers of the YOLO model.
        Returns:
            output_layers (list): List of output layer names.
        """
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def transform(self, img_path):
        """
        Perform object detection on an image.
        Args:
            img_path (str): Path to the input image.
        Returns:
            tuple: Sorted class IDs and confidences of detected objects.
        """
        image = self._load_image(img_path)  # Load the image
        blob = self._prepare_image(image)  # Prepare the image for the YOLO model
        self.net.setInput(blob)  # Set the input to the model
        outs = self.net.forward(self.output_layers)  # Perform the forward pass
        class_ids, confidences, boxes = self._process_detections(outs, image.shape[:2])  # Process the detections
        return self._apply_nms(boxes, confidences, class_ids)  # Apply non-max suppression

    def _load_image(self, img_path):
        """
        Load an image from the specified path.
        Args:
            img_path (str): Path to the image file.
        Returns:
            image (numpy.ndarray): Loaded image.
        """
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"No image found at {img_path}")
        return image

    def _prepare_image(self, image):
        """
        Prepare the image for the YOLO model.
        Args:
            image (numpy.ndarray): Input image.
        Returns:
            blob (numpy.ndarray): Prepared image blob.
        """
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        return blob

    def _process_detections(self, outs, image_shape):
        """
        Process the detections from the YOLO model.
        Args:
            outs (list): Output from the YOLO model.
            image_shape (tuple): Shape of the input image.
        Returns:
            tuple: Class IDs, confidences, and bounding boxes of detected objects.
        """
        height, width = image_shape
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence >= self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return class_ids, confidences, boxes

    def _apply_nms(self, boxes, confidences, class_ids):
        """
        Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes.
        Args:
            boxes (list): List of bounding boxes.
            confidences (list): List of confidences.
            class_ids (list): List of class IDs.
        Returns:
            tuple: Sorted class IDs and confidences after NMS.
        """
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        class_ids_pruned = [class_ids[i] for i in indexes]
        confidences_pruned = [confidences[i] for i in indexes]

        # Sort the pruned results by confidence in descending order
        sorted_indices = np.argsort(confidences_pruned)[::-1]
        class_ids_sorted = [class_ids_pruned[i] for i in sorted_indices]
        confidences_sorted = [confidences_pruned[i] for i in sorted_indices]

        return class_ids_sorted, confidences_sorted
