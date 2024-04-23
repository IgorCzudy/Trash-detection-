import cv2
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import json
from create_feature_vectors import Rectangle
import os
import time

from roboflow import Roboflow
import supervision as sv
import pickle
from create_feature_vectors import apply_dog, find_rois, merge_rois, get_rgb_histogram_vector, get_sift_feature_vector


#-------------------- PARAMETERS ------------------
IOU_THRESHOLD = 0.1 # for AP

# for roboflow model
CONFIDENCE = 20
OVERLAP=30

# for svc model
DOG_THRESHOLD = 0.03
FILTERING_THRESHOLD = 1200 # for filtering rois
# feature_vector = "hsv" # not changeable
# --------------------------------------------------


def get_true_rectangles(path: str, filename: str) -> List[Rectangle]:
    # get trash labeled rectangles from json file
    true_rectangles = []

    with open(path + filename + ".json") as json_file:
        data = json.load(json_file)
        shapes = data["shapes"]

        for shape in shapes:
            if shape["shape_type"] != "rectangle":
                raise Exception("Invalid shape type in", filename)
            if shape["label"] != "trash":
                raise Exception("Invalid label in", filename)

            p1, p2 = shape["points"]
            p1 = tuple(map(int, p1))
            p2 = tuple(map(int, p2))
            
            true_rectangles.append(Rectangle(x_l=min(p1[0], p2[0]), y_b=min(p1[1], p2[1]), x_r=max(p1[0], p2[0]), y_t=max(p1[1], p2[1])))

    return true_rectangles


def get_predicted_rectangles_roboflow(result: dict) -> List[Rectangle]:
    # get predicted rectangles from the result
    predicted_rectangles = []
    predicted_confidences = []

    for item in result["predictions"]:
        predicted_rectangles.append(Rectangle(x_l=item["x"], y_b=item["y"], x_r=item["x"]+item["width"], y_t=item["y"]+item["height"]))
        predicted_confidences.append(item["confidence"])
    
    return predicted_rectangles, predicted_confidences


def get_predicted_rectangles_svc(svc_model, image, scaler, filename):
    with open(path + filename + ".json") as json_file:
        data = json.load(json_file)
        altitude = int(os.path.splitext(data["imagePath"])[0].split("_")[-1])

    mask, _ = apply_dog(image, altitude, DOG_THRESHOLD)
    roi_circles, _ = find_rois(image, mask, visualize=False)
    roi_circles = [
        circle for circle in roi_circles if np.pi * circle.r ** 2 > FILTERING_THRESHOLD
    ]  # filter rois by surface

    roi_circles, _ = merge_rois(roi_circles, image, visualize=False)

    roi_rectangles = []
    for circle in roi_circles:
        x, y, r = circle.x, circle.y, circle.r
        roi_rectangles.append(Rectangle(max(x - r, 0), max(y - r, 0), x + r, y + r))

    X = []
    for circle, rectangle in zip(roi_circles, roi_rectangles):
        x_circles, y_circles, r_circles = circle[0], circle[1], circle[2]

        image_part_hsv = cv2.cvtColor(image[rectangle.y_b:rectangle.y_t, rectangle.x_l:rectangle.x_r], cv2.COLOR_RGB2HSV)
        hsv_feature_vector = get_rgb_histogram_vector(
            image_part_hsv,
            plot=False,
        )

        kp = [cv2.KeyPoint(x_circles, y_circles, 2 * r_circles)]
        sift_feature_vector = get_sift_feature_vector(image, kp, plot=False)
        
        feature_vector = np.concatenate(
            (hsv_feature_vector, sift_feature_vector)
        )

        X.append(feature_vector)

    if X == []:
        predicted_trash_rectangles = []
    else:
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = scaler.transform(X)
        Y_pred = svc_model.predict(X_scaled)
        predicted_trash_rectangles = [rectangle for i, rectangle in enumerate(roi_rectangles) if  Y_pred[i]==1]  

    return predicted_trash_rectangles


def rectangles_intersect(
    rect1: Rectangle,
    rect2: Rectangle,
) -> bool:

    x_distance = max(0, min(rect1.x_r, rect2.x_r) - max(rect1.x_l, rect2.x_l))
    y_distance = max(0, min(rect1.y_t, rect2.y_t) - max(rect1.y_b, rect2.y_b))
    return x_distance * y_distance > 0


def iou_rectangles(rect1: Rectangle, rect2: Rectangle):
    x_left = max(rect1.x_l, rect2.x_l)
    x_right = min(rect1.x_r, rect2.x_r)
    y_bottom = max(rect1.y_b, rect2.y_b)
    y_top = min(rect1.y_t, rect2.y_t)

    intersection = max(0, x_right - x_left) * max(0, y_top - y_bottom)

    area1 = (rect1.x_r - rect1.x_l) * (rect1.y_t - rect1.y_b)
    area2 = (rect2.x_r - rect2.x_l) * (rect2.y_t - rect2.y_b)

    union = area1 + area2 - intersection
    return intersection / union


def merge_rectangles(rectangles_in: list[Rectangle], confidences=None) -> List[Rectangle]:
    rectangles = rectangles_in.copy()
    changed = True
    while changed == True:
        changed = False
        i = 0
        while i < len(rectangles):
            j = i + 1
            while j < len(rectangles):
                if rectangles_intersect(rectangles[i], rectangles[j]):
                    new_rectangle = Rectangle(
                        x_l=min(rectangles[i].x_l, rectangles[j].x_l),
                        y_b=min(rectangles[i].y_b, rectangles[j].y_b),
                        x_r=max(rectangles[i].x_r, rectangles[j].x_r),
                        y_t=max(rectangles[i].y_t, rectangles[j].y_t),
                    )
                    rectangles[i] = new_rectangle
                    rectangles.pop(j)
                    
                    confidences[i] = max(confidences[i], confidences[j])
                    confidences.pop(j)
                    changed = True

                else:
                    j += 1
            i += 1
    return rectangles, confidences


def calculate_ap50(true_rectangles, predicted_rectangles, predicted_confidences=None, iou_threshold=0.5):
    """
    Calculate the precision for a given IoU threshold. Rectangles should be from the same class.
    @param true_rectangles: List of true rectangles
    @param predicted_rectangles: List of predicted rectangles
    @param predicted_confidences: List of predicted confidences
    @param iou_threshold: IoU threshold
    """

    # Sort predicted rectangles by confidence score
    if predicted_confidences is not None:
        sorted_indices = np.argsort(predicted_confidences)[::-1]
        predicted_rectangles = [predicted_rectangles[i] for i in sorted_indices]

    tp = 0
    fp = 0

    for pred_rect in predicted_rectangles:
        is_tp = False
        for true_rect in true_rectangles:
            iou = iou_rectangles(pred_rect, true_rect)
            if iou >= iou_threshold:
                is_tp = True
                break
        if is_tp:
            tp += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else None

    return precision, tp, fp


def draw_rectangles(image, rectangles: List[Rectangle], color=(0, 255, 0)):
    for rectangle in rectangles:
        cv2.rectangle(image, (rectangle.x_l, rectangle.y_b), (rectangle.x_r, rectangle.y_t), color, 3)
    return image


def evaluate_model(model, path, model_type, visualize=False, standard_scaler=None):
    all_tp = 0
    all_fp = 0
    avg_time = 0
    n_files = 0

    for filename in os.listdir(path):
        if filename.endswith(".JPG"):
            n_files += 1
            if model_type == "roboflow":
                start = time.time()
                result = model.predict(path+filename, confidence=CONFIDENCE, overlap=OVERLAP).json()
                unmerged_rectangles, unmerged_confidences = get_predicted_rectangles_roboflow(result)
                predicted_rectangles, predicted_confidences = merge_rectangles(unmerged_rectangles, unmerged_confidences)
                end = time.time()

            elif model_type == "svc":
                img = cv2.imread(path + filename)
                predicted_confidences = None

                start = time.time()
                predicted_rectangles = get_predicted_rectangles_svc(model, img, standard_scaler, os.path.splitext(filename)[0])
                end = time.time()
            
            avg_time += end - start

            true_rectangles = get_true_rectangles(path, os.path.splitext(filename)[0])

            if visualize:
                fig = plt.figure(filename)
                fig.canvas.set_window_title(filename)
                img = cv2.imread(path + filename)
                draw_rectangles(img, true_rectangles, (0, 255, 0))
                if model_type == "roboflow": draw_rectangles(img, unmerged_rectangles, (255, 0, 255))
                draw_rectangles(img, predicted_rectangles, (0, 0, 255))
                plt.imshow(img)
                plt.show()

            _, tp, fp = calculate_ap50(true_rectangles, predicted_rectangles, predicted_confidences, iou_threshold=IOU_THRESHOLD)
            all_tp += tp
            all_fp += fp
            
            
    avg_time /= n_files
    ap50 = all_tp / (all_tp + all_fp) if all_tp + all_fp > 0 else None
    
    print(f"{model_type.upper()}")
    print(f"AP: {ap50:.3f}")
    print(f"Avg time of inference and getting rectangles: {avg_time:.3f} s")



if __name__ == "__main__":
    paths = ["data/Dataset/training/", "data/Dataset/validation/", "data/Dataset/test/"]
    visualize = False

    rf = Roboflow(api_key="4gI0eIbggGqzYmHFPYIk")
    project = rf.workspace().project("waste-in-water")
    rf_model = project.version(1).model

    with open('out/svc_model.pkl', 'rb') as f:
        standard_scaler, svc_model = pickle.load(f)

    print(f"\n{IOU_THRESHOLD=}\n")

    for path in paths:
        print(f"----{path.split('/')[-2].upper()}-----\n")
        evaluate_model(rf_model, path, "roboflow", visualize)
        print("-------------------\n")
        evaluate_model(svc_model, path, "svc", visualize, standard_scaler)
        print("-------------------\n")
