import time
import numpy as np
import tensorflow.keras.models
import tensorflow as tf
from cv2 import cv2
from imutils.object_detection import non_max_suppression
from img_utilities import image_pyramid, image_reverse_pyramid, sliding_window

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

model = tensorflow.keras.models.load_model('model1.h5')

PYR_SCALE = 1.25
WIN_STEP = 4
ROI_SIZE = (128, 128)
INPUT_SIZE = (128, 128)
MIN_PROB = 0.85
MAX_PROB = 1


def find_plane(image=None, zoom=False):
    global WIN_STEP
    original = image
    rois = []
    locs = []

    start = time.time()

    (H, W) = original.shape[:2]

    # pyramid = image_pyramid(original, scale=PYR_SCALE, minSize=ROI_SIZE)
    if zoom :
        pyramid = image_reverse_pyramid(original, scale=PYR_SCALE, maxSize=(1.26*W,1.26*H))
        WIN_STEP = 8
    else:
        pyramid = [original]

    for image in pyramid:
        scale = W / float(image.shape[1])

        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)

            roi = cv2.resize(roiOrig, INPUT_SIZE)

            rois.append(roi)
            locs.append((x, y, x + w, y + h))

    end = time.time()
    print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
        end - start))


    rois = np.array(rois)

    prediction = model.predict(rois)

    predicted_boxes = []

    for (i, p) in enumerate(prediction):
        prob = p[0]

        if prob >= MIN_PROB and prob <= MAX_PROB:
            predicted_boxes.append((locs[i], prob))


    boxes = []
    probs = []

    for (box, prob) in predicted_boxes:
        startX, startY, endX, endY = box
        boxes.append((startX, startY, endX, endY))
        probs.append(prob)

    clone = original.copy()

    boxes = np.array(boxes)
    probs = np.array(probs)
    boxes = non_max_suppression(boxes, probs)

    for startX, startY, endX, endY in boxes:
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 0, 255), 2)

    #clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)

    return clone
