import cv2
import json

# GroundTruth manifest file which contains information needed to map segmentation masks to classes.
GROUND_TRUTH_MANIFEST_FILE = "ground_truth_labels/output.manifest"

# YOLOv5 labels (output folder).
YOLOV5_LABELS_FOLDER = "yolov5_labels"


def main():
    # Read in the manifest file.
    # This file is JSON-formatted, but has to be read line-by-line.
    with open(GROUND_TRUTH_MANIFEST_FILE, 'r') as manifest_file:
        lines = manifest_file.readlines()
        for line in lines:
            # Convert line to manifest entry.
            manifest_entry = json.loads(line)
            label_metadata = manifest_entry['coding-challenge-data-engineer-ref-metadata']

            # Map class names to YOLOv5 IDs.
            class_name_to_id_map = {}
            label_defs = label_metadata['internal-color-map']
            for key in label_defs:
                label_def = label_defs[key]
                label_name = label_def['class-name']
                class_id = int(key) - 1
                if class_id < 0:
                    # Ignore the BACKGROUND class.
                    continue
                class_name_to_id_map[int(key) - 1] = label_name

            # Load the original image.
            image_path = manifest_entry["source-ref"]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            height, width, _ = image.shape

            # Expected location of the YOLOv5 annotation.
            annotations_path = f'{YOLOV5_LABELS_FOLDER}/{image_path.split("/")[-1].replace(".jpg", ".txt")}'
            with open(annotations_path, 'r') as annotations_file:
                annotations = annotations_file.readlines()

            # Visualize the bounding boxes.
            for annotation in annotations:
                # Convert YOLOv5 annotation to a bounding box rectangle.
                components = annotation.split(" ")
                class_id = int(components[0])
                x_center = float(components[1]) * width
                y_center = float(components[2]) * height
                w = int(float(components[3]) * width)
                h = int(float(components[4]) * height)
                x = int(x_center - w/2)
                y = int(y_center - h/2)
                top_left = (x, y)
                bottom_right = (x + w, y + h)

                # Draw the bounding box and text on the original image.
                color = (0, 255, 0)
                label_name = class_name_to_id_map[class_id]
                cv2.rectangle(image, top_left, bottom_right, color, 5)
                cv2.putText(image, label_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Write out the original image with bounding box overlays.
            cv2.imwrite(image_path.split("/")[-1], image)


if __name__ == '__main__':
    main()
