# test.py
import cv2
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

def start_camera_and_recognition():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)
    classifier = Classifier("Model/Keras_model.h5", "Model/labels.txt")
    labels = ["A", "B", "C"]
    offset = 20
    imgsize = 300

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        imgotp = img.copy()
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Your existing code for hand gesture recognition

        # Display the processed image
        cv2.imshow("Camera Feed", imgotp)
        key = cv2.waitKey(1)
        if key == 27:  # Press Esc to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera_and_recognition()
