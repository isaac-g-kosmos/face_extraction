from yolo_utils import scale_boxes
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
import wandb
from keras.utils import plot_model
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pre_process import augmented_cut

@tf.function
def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4)
        box_confidence -- tensor of shape (19, 19, 5, 1)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    x = 10
    y = tf.constant(100)

    # YOUR CODE STARTS HERE

    # Step 1: Compute box scores
    ##(≈ 1 line)
    box_scores = box_class_probs * box_confidence

    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    ##(≈ 2 lines)
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ## (≈ 1 line)
    filtering_mask = (box_class_scores >= threshold)

    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    ## (≈ 3 lines)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    # YOUR CODE ENDS HERE

    return scores, boxes, classes

@tf.function
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # YOUR CODE STARTS HERE

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ##(≈ 7 lines)
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(0, yi2 - yi1)
    inter_height = max(0, xi2 - xi1)
    inter_area = inter_width * inter_height

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ## (≈ 3 lines)
    box1_area = (box1_x2 - box1_x1) * ((box1_y2 - box1_y1))
    box2_area = (box2_x2 - box2_x1) * ((box2_y2 - box2_y1))
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    ## (≈ 1 line)
    iou = inter_area / union_area

    # YOUR CODE ENDS HERE

    return iou

@tf.function
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    ##(≈ 1 line)
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)

    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    ##(≈ 3 lines)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    return scores, boxes, classes
@tf.function
def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model (≈1 line)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, score_threshold)

    # Scale boxes back to original image shape (720, 1280 or whatever)
    boxes = scale_boxes(boxes, image_shape)  # Network was trained to run on 608x608 images

    # Use one of the functions you've implemented to perform Non-max suppression with
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    return scores, boxes, classes

inputs = Input(shape=(256, 256, 3), name='main_input')
inputs = Input(shape=(256, 256, 3), name='main_input')












# inputs = Input(shape=(shape[0], shape[1], 3), name='main_input')
#
# main_branch = Conv2D(8, kernel_size=(3, 3), padding="same")(inputs)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
#
# main_branch = Conv2D(16, kernel_size=(3, 3), padding="same")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
# main_branch = Conv2D(8, kernel_size=(3, 3), padding="same")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
# main_branch = Flatten()(main_branch)
# main_branch = Dense(32)(main_branch)
# people_branch=Dense(16)(main_branch)
# people_branch=Activation("relu")(people_branch)
# people_branch=Dropout(dr)(people_branch)
# main_branch = Activation('relu')(main_branch)
# main_branch = Dropout(dr)(main_branch)
# main_branch = Dense(16)(main_branch)
#
# land_marks = Dense(4, activation='relu', name='landmarks')(main_branch)
# people_count = Dense(3, activation='softmax', name='people')(people_branch)
















# inputs = Input(shape=(shape[0], shape[1], 3), name='main_input')
#
# main_branch = Conv2D(8, kernel_size=(3, 3), padding="same")(inputs)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
#
# main_branch = Conv2D(16, kernel_size=(3, 3), padding="same")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
# main_branch = Conv2D(16, kernel_size=(3, 3), padding="same")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
# main_branch = Flatten()(main_branch)
# main_branch = Dense(32)(main_branch)
# people_branch=Dense(16)(main_branch)
# people_branch=Activation("relu")(people_branch)
# people_branch=Dropout(dr)(people_branch)
# main_branch = Activation('relu')(main_branch)
# main_branch = Dropout(dr)(main_branch)
# main_branch = Dense(16)(main_branch)
#
# land_marks = Dense(4, activation='relu', name='landmarks')(main_branch)
# people_count = Dense(3, activation='softmax', name='people')(people_branch)

# filter_size = (5, 5)
# maxpool_size = (2, 2)
# dr = 0.3
#
# inputs = Input(shape=(shape[0], shape[1], 3), name='main_input')
#
# main_branch = Conv2D(8, kernel_size=(5, 5), padding="same")(inputs)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
#
# main_branch = Conv2D(16, kernel_size=(3, 3), padding="same")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
# main_branch = Conv2D(16, kernel_size=(3, 3), padding="same")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
# main_branch = Flatten()(main_branch)
# main_branch = Dense(32)(main_branch)
# people_branch=Dense(16)(main_branch)
# people_branch=Activation("relu")(people_branch)
# people_branch=Dropout(dr)(people_branch)
# main_branch = Activation('relu')(main_branch)
# main_branch = Dropout(dr)(main_branch)
# main_branch = Dense(16)(main_branch)
# filter_size = (5, 5)
# maxpool_size = (2, 2)
# dr = 0.3
#
# inputs = Input(shape=(shape[0], shape[1], 3), name='main_input')
#
# main_branch = Conv2D(8, kernel_size=(5, 5), padding="same")(inputs)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
#
#
# main_branch = Conv2D(16, kernel_size=(3, 3), padding="same")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
# main_branch = Flatten()(main_branch)
# main_branch = Dense(32)(main_branch)
# people_branch=Dense(16)(main_branch)
# people_branch=Activation("relu")(people_branch)
# people_branch=Dropout(dr)(people_branch)
# main_branch = Activation('relu')(main_branch)
# main_branch = Dropout(dr)(main_branch)
# main_branch = Dense(16)(main_branch)
#
# filter_size = (5, 5)
# maxpool_size = (2, 2)
# dr = 0.3
#
# inputs = Input(shape=(shape[0], shape[1], 3), name='main_input')
#
# main_branch = Conv2D(8, kernel_size=(5, 5), padding="same")(inputs)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
#
#
# main_branch = Conv2D(16, kernel_size=(3, 3), padding="same")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
# main_branch = Flatten()(main_branch)
# main_branch = Dense(32)(main_branch)
# people_branch=Dense(16)(main_branch)
# people_branch=Activation("relu")(people_branch)
# people_branch=Dropout(dr)(people_branch)
# main_branch = Activation('relu')(main_branch)
# main_branch = Dropout(dr)(main_branch)
# main_branch = Dense(16)(main_branch)

# filter_size = (5, 5)
# maxpool_size = (2, 2)
# dr = 0.3
#
# inputs = Input(shape=(shape[0], shape[1], 3), name='main_input')
#
# main_branch = Conv2D(8, kernel_size=(5, 5), padding="valid")(inputs)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
#
# main_branch = Conv2D(8, kernel_size=(5, 5), padding="valid")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
# main_branch = Conv2D(4, kernel_size=(5, 5), padding="valid")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
# main_branch = Conv2D(4, kernel_size=(3, 3), padding="valid")(main_branch)
# main_branch = Activation("relu")(main_branch)
# main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
#
# main_branch = Flatten()(main_branch)
# main_branch = Dense(80)(main_branch)
# main_branch = Dense(32)(main_branch)
# main_branch = Dense(16)(main_branch)
#
# people_branch=Dropout(dr)(main_branch)
# main_branch = Activation('relu')(main_branch)
# main_branch = Dropout(dr)(main_branch)
# main_branch = Dense(16)(main_branch)
#
#
# land_marks = Dense(4, activation='relu', name='landmarks')(main_branch)
# people_count = Dense(3, activation='softmax', name='people')(main_branch)