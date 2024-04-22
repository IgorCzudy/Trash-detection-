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


def calculate_ap50(true_rectangles, predicted_rectangles, predicted_confidences, iou_threshold=0.5):
    """
    Calculate the precision for a given IoU threshold. Rectangles should be from the same class.
    @param true_rectangles: List of true rectangles
    @param predicted_rectangles: List of predicted rectangles
    @param predicted_confidences: List of predicted confidences
    @param iou_threshold: IoU threshold
    """

    # Sort predicted rectangles by confidence score
    sorted_indices = np.argsort(predicted_confidences)[::-1]
    predicted_rectangles = [predicted_rectangles[i] for i in sorted_indices]

    tp = 0
    fp = 0

    for d, pred_rect in enumerate(predicted_rectangles):
        for t, true_rect in enumerate(true_rectangles):
            iou = iou_rectangles(pred_rect, true_rect)
            if iou >= iou_threshold:
                tp += 1
            else:
                fp += 1
    precision = tp / (tp + fp) if tp + fp > 0 else None

    return precision, tp, fp


def draw_rectangles(image, rectangles: List[Rectangle], color=(0, 255, 0)):
    for rectangle in rectangles:
        cv2.rectangle(
            image,
            (rectangle.x_l, rectangle.y_b),
            (rectangle.x_r, rectangle.y_t),
            color,
            3
        )
    return image


if __name__ == "__main__":
    rf = Roboflow(api_key="FvqF3j07aqqT0BqM0VjB")
    project = rf.workspace().project("waste-in-water")
    model = project.version(1).model
    path = "data/Dataset/training/"
    
    visualize = True

    all_tp = 0
    all_fp = 0
    avg_time = 0
    n_files = 0

    for filename in os.listdir(path):
        if filename.endswith(".JPG"):
            n_files += 1
            start = time.time()
            result = model.predict(path+filename, confidence=20, overlap=30).json()
            end = time.time()
            avg_time += end - start

            true_rectangles = get_true_rectangles(path, os.path.splitext(filename)[0])
            predicted_rectangles, predicted_confidences = get_predicted_rectangles_roboflow(result)
            merged_rectangles, merged_confidences = merge_rectangles(predicted_rectangles, predicted_confidences)

            if visualize:
                fig = plt.figure(filename)
                img = cv2.imread(path + filename)
                draw_rectangles(img, true_rectangles, (0, 255, 0))
                draw_rectangles(img, predicted_rectangles, (0, 0, 255))
                draw_rectangles(img, merged_rectangles, (255, 0, 255))
                plt.imshow(img)
                plt.show()

            _, tp, fp = calculate_ap50(true_rectangles, merged_rectangles, merged_confidences, iou_threshold=0.5)
            all_tp += tp
            all_fp += fp
            
            
    avg_time /= n_files
    ap50 = all_tp / (all_tp + all_fp) if all_tp + all_fp > 0 else None
    
    print(f"AP of roboflow model: {ap50:.3f}")
    print(f"Avg time of inference: {avg_time:.3f} s")