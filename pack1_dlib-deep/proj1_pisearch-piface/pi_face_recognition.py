# USAGE
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
import face_recognition
import argparse
import imutils
from imutils.video import FPS
import pickle
import cv2
from pathlib import Path

DEBUG = False

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-i", "--imagedir", required=True, help="path to the input images directory")
args = vars(ap.parse_args())

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

im_dir = Path(args["imagedir"])
im_list = [f for f in im_dir.glob("*.png")]
if len(im_list) == 0:
    print("empty directory"); exit(1)
im_list.sort()
right_face_match = 0

print("[INFO] processing images...")
fps = FPS().start()
for (i, im_path) in enumerate(im_list):
    fps.stop()
    right_match_percent_str = "{:.2f}%".format(100 * right_face_match/len(im_list))
    if i % 10 == 0:
        print("Processing image {:<4} | r-match {:>5} | {:02.4} FPS"
              .format(i,right_match_percent_str,fps.fps()), end='\r', flush=True)
    
    # get image information
    correct_class = im_path.stem.split('_')[0]
    if correct_class.startswith("unkown"):
        correct_nfaces = 0
        correct_class = "unkown"
    else:
        correct_nfaces = 1

    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    # frame = vs.read()
    frame = cv2.imread(str(im_path))
    frame = imutils.resize(frame, width=500)

    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
        
        # update the list of names
        names.append(name)

    # loop over the recognized faces
    counted_right_match = False
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)
        if name == correct_class and not counted_right_match:
            right_face_match += 1
            counted_right_match = True
    fps.update()


print("")
right_match_percent_str = "{:.2f}%".format(100 * right_face_match/len(im_list))
print("Total correct recognition match: {} ({})".format(right_face_match, right_match_percent_str))

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cv2.imshow("Frame", frame)
# cv2.waitKey()
