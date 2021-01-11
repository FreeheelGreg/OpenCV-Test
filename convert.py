import cv2
import numpy
import json

# GroundTruth manifest file which contains information needed to map segmentation masks to classes.
GROUND_TRUTH_MANIFEST_FILE = "ground_truth_labels/output.manifest"

# YOLOv5 labels (output folder).
YOLOV5_LABELS_FOLDER = "yolov5_labels"


def main():

    # Read in the manifest file.
    # This file is JSON-formatted, but has to be read line-by-line.
    with open(GROUND_TRUTH_MANIFEST_FILE, 'r') as manifest_file:

        # Read all lines, and convert each line to a JSON object.
        lines = manifest_file.readlines()
        for line in lines:

            # Convert text line to GroundTruth manifest entry.
            manifest_entry = json.loads(line)

            ###################################################
            #####            ADD YOUR CODE BELOW          #####
            ###################################################
            pass


if __name__ == '__main__':
    main()
