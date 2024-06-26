import cv2
import numpy as np
import os
import time
class Yolo:
    def __init__(self, config_dir= "yolo_tiny_configs", confidence_threshold=0.5, nms_threshold=0.0):
        """
        Initialize the Yolo object with configuration directory, confidence threshold and NMS threshold.
        Args:
            config_dir (str): Directory containing YOLO configuration files.
                Path starts from location of this python class file
            confidence_threshold (float): Confidence threshold for detections.
            nms_threshold (float): Non-max suppression threshold.
        """
        self.config_dir = os.path.dirname(__file__)
        self.config_dir = os.path.join(self.config_dir, config_dir)

        print("Loading YOLO model from:", self.config_dir)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = self._load_model()
        self.classes = self._load_classes()
        self.output_layers = self._get_output_layers()

    def _load_model(self):
        """
        Load the YOLO model from the configuration files.
        Returns:
            net: Loaded YOLO model cv2.dnn_Net.
        """
        weights_path = f"{self.config_dir}/yolov3-tiny.weights"
        config_path = f"{self.config_dir}/yolov3-tiny.cfg"
        try:
            net = cv2.dnn.readNet(weights_path, config_path)
        except cv2.error as e:
            print("Error loading YOLO model:", e)
            print("Please make sure the YOLO configuration files are in the correct directory.")
            raise Exception("""Error loading YOLO model. Please make sure the YOLO configuration files are in the correct directory.""")
        return net


    def _load_classes(self):
        """
        Load the class names from the coco.names file.
        Returns:
            classes: List of class names.
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
        # get the names of the output layers in the network
        unconnected_out_layers = self.net.getUnconnectedOutLayers()
        # it happens that they might be a numpy.ndarray of numpy.ndarrays of indices, rather than a numpy.ndarray of indices
        # somehow that's not always the case
        # if it is, flatten it here
        unconnected_out_layers = self.flatten_list_of_iterables(unconnected_out_layers)
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        return output_layers


    def flatten_list_of_iterables(self, list_of_iterables):
        """
        Flatten a list of lists, if the first element is a list or numpy array.
        Otherwise, return the list as is.
        Args:
            list_of_lists: List of lists or numpy arrays.
        """
        if isinstance(list_of_iterables[0], list) or isinstance(list_of_iterables[0], np.ndarray):
            return [item for sublist in list_of_iterables for item in sublist]
        else:
            return list_of_iterables


    def transform(self, img_path, verbose=False):
        """
        Perform object detection on an image.
        Args:
            img_path: Path to the input image.
        Returns:
            tuple: Sorted class IDs and confidences of detected objects.
        """
        image = self._load_image(img_path)  # Load the image
        blob = self._prepare_image(image)  # Prepare the image for the model
        self.net.setInput(blob)  # Set the input to the model
        outs = self.net.forward(self.output_layers)  # Perform the forward pass
        class_ids, confidences, boxes = self._process_detections(outs, image.shape[:2])  # Process the detections

        # Apply non-max suppression
        class_ids_pruned, confidences_pruned, boxes_pruned = self._apply_nms(boxes, confidences, class_ids)

        # print results
        if verbose:
            print("We found")
            for i, conf in zip(class_ids_pruned, confidences_pruned):
                print(f"--> a {self.classes[i]} with a confidence of {conf}")

        return class_ids_pruned, confidences_pruned, boxes_pruned

    def transform_and_time(self, img_path, verbose=False):
        """
        Perform object detection on an image.
        Args:
            img_path: Path to the input image.
        Returns:
            tuple: Sorted class IDs and confidences of detected objects.
        """
        start_time = time.time()
        image = self._load_image(img_path)  # Load the image
        blob = self._prepare_image(image)  # Prepare the image for the model
        self.net.setInput(blob)  # Set the input to the model
        outs = self.net.forward(self.output_layers)  # Perform the forward pass
        class_ids, confidences, boxes = self._process_detections(outs, image.shape[:2])  # Process the detections

        # Apply non-max suppression
        class_ids_pruned, confidences_pruned, boxes_pruned = self._apply_nms(boxes, confidences, class_ids)
        end_time = time.time()
        object_detection_time = end_time - start_time
        # print results
        if verbose:
            print("We found")
            for i, conf in zip(class_ids_pruned, confidences_pruned):
                print(f"--> a {self.classes[i]} with a confidence of {conf}")

        return object_detection_time, class_ids_pruned, confidences_pruned, boxes_pruned


    def draw(self, image, class_ids, confidences, boxes, color=(0, 255, 0)):
        # Draw bounding boxes on the image
        font = cv2.FONT_HERSHEY_PLAIN
        image_drawn = image.copy()

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            cv2.rectangle(image_drawn, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_drawn, label, (x, y + 30), font, 3, color, 3)

        return image_drawn



    def transform_draw(self, img_path_in, display=False, verbose=False, color=(0, 255, 0)):
        """
        Perform object detection on an image and draw bounding boxes on the detected objects.
        Args:
            img_path_in: Path to the input image.
            display: Whether to display the image with bounding boxes.
        Returns:
            image: Image with bounding boxes drawn.
        """
        image = cv2.imread(img_path_in)
        class_ids, confidences, boxes = self.transform(img_path_in, verbose=verbose)

        # Draw bounding boxes on the image
        image = self.draw(image, class_ids, confidences, boxes, color=color)

        if display:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image, (class_ids, confidences, boxes)



    def _load_image(self, img_path):
        """
        Load an image from the specified path.
        Args:
            img_path: Path to the image file.
        Returns:
            image: Loaded image.
        """
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"No image found at {img_path}")
        return image

    def _prepare_image(self, image):
        """
        Prepare the image for the YOLO model.
        Args:
            image: Input image.
        Returns:
            blob: Prepared image blob.
        """
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        return blob

    def _process_detections(self, outs, image_shape):
        """
        Process the detections from the YOLO model.
        Args:
            outs: Output from the YOLO model.
            image_shape: Shape of the input image.
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
            boxes: List of bounding boxes.
            confidences: List of confidences.
            class_ids: List of class IDs.
        Returns:
            tuple: Sorted class IDs and confidences after NMS.
        """
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        # if indexes is a list of iterables instead of a list of indices, flatten them
        if len(indexes) > 0:
            indexes = self.flatten_list_of_iterables(indexes)

            class_ids_pruned = [class_ids[i] for i in indexes]
            confidences_pruned = [confidences[i] for i in indexes]
            boxes_pruned = [boxes[i] for i in indexes]

            # Sort the pruned results by confidence in descending order
            sorted_indices = np.argsort(confidences_pruned)[::-1]
            class_ids_sorted = [class_ids_pruned[i] for i in sorted_indices]
            confidences_sorted = [confidences_pruned[i] for i in sorted_indices]
            boxes_sorted = [boxes_pruned[i] for i in sorted_indices]

            return class_ids_sorted, confidences_sorted, boxes_sorted
        else:
            return [],[],[]

    def save_image(self, image, img_path_out):
        # does not work on AWS i think
        # Save the output image with the drawn bounding boxes
        cv2.imwrite(img_path_out, image)


if __name__ == "__main__":
    yolo = Yolo()
    print("Testing Yolo on an image!")
    image, _ = yolo.transform_draw("../../input_folder/000000000674.jpg", verbose=True)
    yolo.save_image(image, "example_output_img.jpg")
    print("Great success!")
