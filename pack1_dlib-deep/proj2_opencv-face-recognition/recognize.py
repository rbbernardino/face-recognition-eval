# USAGE
# python recognize.py --detector face_detection_model \
#    --embedding-model openface_nn4.small2.v1.t7 \
#    --recognizer output/recognizer.pickle \
#    --le output/le.pickle --image images/adrian.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
from imutils.video import FPS
import pickle
import cv2
import os
from pathlib import Path

DEBUG = False

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagedir", required=True,
    help="path to input image")
ap.add_argument("-d", "--detector", required=True,
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
    help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
    help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
    help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

im_list = [f for f in Path(args["imagedir"]).glob("*.png")]
im_list.sort()
right_face_match = 0
print("[INFO] processing images...")

# start the FPS throughput estimator
fps = FPS().start()
for (i, im_path) in enumerate(im_list):
    fps.stop()
    right_match_percent_str = "{:.2f}%".format(100 * right_face_match/len(im_list))
    if i % 10 == 0:
        print("Processing image {:<4} | r-match {:>5} | {:.4} FPS"
              .format(i,right_match_percent_str,fps.fps()), end='\r', flush=True)
    
    # get image information
    correct_class = im_path.stem.split('_')[0]
    if correct_class.startswith("unkown"):
        correct_nfaces = 0
        correct_class = "unkown"
    else:
        correct_nfaces = 1

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    image = cv2.imread(str(im_path))
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        name = "unkown"
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

    if name == correct_class:
        right_face_match += 1
        counted_right_match = True

            # draw the bounding box of the face along with the associated
            # probability
            # text = "{}: {:.2f}%".format(name, proba * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(image, (startX, startY), (endX, endY),
            #     (0, 0, 255), 2)
            # cv2.putText(image, text, (startX, y),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # update the FPS counter
    fps.update()


right_match_percent_str = "{:.2f}%".format(100 * right_face_match/len(im_list))
print("Total correct recognition match: {} ({})".format(right_face_match, right_match_percent_str))

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)
