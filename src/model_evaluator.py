import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from roboflow import Roboflow
from sklearn.metrics import precision_recall_curve, auc
import supervision as sv
import time
from typing import List

from create_feature_vectors import Rectangle, create_feature_vector, iou_rectangles, get_true_rectangles

#-------------------- PARAMETERS ------------------
IOU_THRESHOLD = 0.5 # for AP

# for roboflow model
CONFIDENCE = 10
OVERLAP=30

# for svc model
DOG_THRESHOLD = 0.03
SVC_IMG_SCALING_FACTOR = 162/1080
# feature_vector = "hsv" # not changeable
# --------------------------------------------------

def get_predicted_rectangles_roboflow(result: dict) -> List[Rectangle]:
    # get predicted rectangles from the result
    predicted_rectangles = []
    predicted_confidences = []

    for item in result["predictions"]:
        predicted_rectangles.append(Rectangle(x_l=item["x"], y_b=item["y"], x_r=item["x"]+item["width"], y_t=item["y"]+item["height"]))
        predicted_confidences.append(item["confidence"])
    
    return predicted_rectangles, predicted_confidences


def get_predicted_rectangles_svc(svc_model, path, scaler, filename):
    X, roi_rectangles = create_feature_vector(filename, path, dog_threshold = DOG_THRESHOLD, visualize=False, no_trash_warning=False, timing=False, with_labels=False) 

    if X == []:
        scaled_predicted_trash_rectangles = []
    else:
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = scaler.transform(X) if scaler is not None else X
        Y_pred = svc_model.predict(X_scaled)
        predicted_trash_rectangles = [rectangle for i, rectangle in enumerate(roi_rectangles) if  Y_pred[i]==1]  

        # rescale to HD image
        scaled_predicted_trash_rectangles = []
        for rectangle in predicted_trash_rectangles:
            scaled_rectangle = Rectangle(x_l=int(rectangle.x_l/SVC_IMG_SCALING_FACTOR) , y_b=int(rectangle.y_b/SVC_IMG_SCALING_FACTOR) , x_r=int(rectangle.x_r/SVC_IMG_SCALING_FACTOR) , y_t=int(rectangle.y_t/SVC_IMG_SCALING_FACTOR))
            scaled_predicted_trash_rectangles.append(scaled_rectangle)

    return scaled_predicted_trash_rectangles


def rectangles_intersect(
    rect1: Rectangle,
    rect2: Rectangle,
) -> bool:

    x_distance = max(0, min(rect1.x_r, rect2.x_r) - max(rect1.x_l, rect2.x_l))
    y_distance = max(0, min(rect1.y_t, rect2.y_t) - max(rect1.y_b, rect2.y_b))
    return x_distance * y_distance > 0


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

    true_labels = []
    predicted_labels = []

    for pred_rect in predicted_rectangles:
        is_tp = False
        for true_rect in true_rectangles:
            iou = iou_rectangles(pred_rect, true_rect)
            if iou >= iou_threshold:
                is_tp = True
                break
        if is_tp:
            true_labels.append(1)
            predicted_labels.append(1)
            tp += 1
        else:
            true_labels.append(0)
            predicted_labels.append(1)
            fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else None

    return precision, tp, fp, predicted_labels, true_labels


def draw_rectangles(image, rectangles: List[Rectangle], color=(0, 255, 0)):
    for rectangle in rectangles:
        cv2.rectangle(image, (rectangle.x_l, rectangle.y_b), (rectangle.x_r, rectangle.y_t), color, 3)
    return image


def evaluate_model(model, path, model_type, visualize=False, standard_scaler=None):
    all_tp = 0
    all_fp = 0
    all_predicted_confidences = []
    all_true_labels = []
    avg_time = 0
    n_files = 0

    for filename in os.listdir(path + "images/"):
        n_files += 1
        if model_type == "roboflow":
            start = time.time()
            result = model.predict(path + "images/" + filename, confidence=CONFIDENCE, overlap=OVERLAP).json()
            unmerged_rectangles, unmerged_confidences = get_predicted_rectangles_roboflow(result)
            predicted_rectangles, predicted_confidences = merge_rectangles(unmerged_rectangles, unmerged_confidences)
            end = time.time()

        elif model_type == "svc":
            predicted_confidences = None

            start = time.time()
            predicted_rectangles = get_predicted_rectangles_svc(model, path, standard_scaler, os.path.splitext(filename)[0])
            end = time.time()
        
        avg_time += end - start

        true_rectangles = get_true_rectangles(path, os.path.splitext(filename)[0])

        if visualize:
            fig = plt.figure(filename)
            img = cv2.imread(path + "images/" + filename)
            draw_rectangles(img, true_rectangles, (0, 0, 255))
            if model_type == "roboflow": draw_rectangles(img, unmerged_rectangles, (255, 0, 255))
            draw_rectangles(img, predicted_rectangles, (0, 255, 0))
            plt.title(f"{filename} - {model_type.upper()} predicted: green, true: blue")
            plt.imshow(img)
            plt.show()

        _, tp, fp, predicted_labels, true_labels = calculate_ap50(true_rectangles, predicted_rectangles, predicted_confidences, iou_threshold=IOU_THRESHOLD)
        all_tp += tp
        all_fp += fp
        all_predicted_confidences += predicted_confidences
        all_true_labels += true_labels

    if predicted_confidences is not None:
        precision, recall, thresholds = precision_recall_curve(all_true_labels, all_predicted_confidences, pos_label=1, drop_intermediate=True)
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.show()
        print("thresholds:", thresholds)
        pr_auc = auc(recall, precision)
        print("precision-recall auc: ", pr_auc)
 
            
    avg_time /= n_files
    if all_tp + all_fp == 0:
        print("No trash found in the images")
    else:
        mAP = all_tp / (all_tp + all_fp) 
        
        print(f"{model_type.upper()}")
        print(f"AP: {mAP:.3f}")
    print(f"Avg time of inference and getting rectangles: {avg_time:.3f} s")



if __name__ == "__main__":
    paths = ["data/Dataset/training/", "data/Dataset/validation/", "data/Dataset/test/"]
    visualize = False

    rf = Roboflow(api_key="4gI0eIbggGqzYmHFPYIk")
    project = rf.workspace().project("waste-in-water")
    rf_model = project.version(1).model

    with open('out/svc_model.pkl', 'rb') as f:
        standard_scaler, svc_model = pickle.load(f)

    print(f"\n{IOU_THRESHOLD=}\nminimal {CONFIDENCE=}\n")

    for path in paths:
        if os.path.exists(path + "images/") == False:
            print("No such directory", path)
            continue
        print(f"----{path.split('/')[-2].upper()}-----\n")
        evaluate_model(rf_model, path, "roboflow", visualize)
        print("-------------------\n")
        evaluate_model(svc_model, path, "svc", visualize, standard_scaler)
        print("-------------------\n")
