import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from shapely.geometry import Point, Polygon
from skimage import io, color, img_as_float
from skimage.filters import difference_of_gaussians
from typing import List, Tuple
from collections import namedtuple
import time

Rectangle = namedtuple("Rectangle", ["x_l", "y_b", "x_r", "y_t"])
Circle = namedtuple("Circle", ["x", "y", "r"])

IMAGE_SHAPE = (162, 288)


def _get_params(altitude: int) -> int:
    """
    f for choosing the σ value based on the altitude h:
    """
    if altitude <= 5:
        sigma = 6
        max_filtering_threshold = 1800
        min_filtering_threshold = 50
    elif altitude <= 10:
        sigma = 5
        max_filtering_threshold = 1000
        min_filtering_threshold = 40
    elif altitude <= 15:
        sigma = 4
        max_filtering_threshold = 700
        min_filtering_threshold = 10
    elif altitude <= 20:
        sigma = 3
        max_filtering_threshold = 600
        min_filtering_threshold = 2
    else:
        sigma = 2
        max_filtering_threshold = 300
        min_filtering_threshold = 0
    return sigma, min_filtering_threshold, max_filtering_threshold


def apply_dog(image: np.array, sigma: int, threshold=0.03) -> Tuple[np.array, np.array]:
    """
    Apply difference of gaussians
    """

    gray_image = color.rgb2gray(image)
    float_image = img_as_float(gray_image)

    dog_image = difference_of_gaussians(float_image, sigma, 1.6 * sigma)

    mask = dog_image.copy()
    _, mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    return mask, dog_image


def find_rois(
    image: np.array, mask: np.array, visualize=False
) -> Tuple[List[Circle], np.array]:
    """
    Find regions of interest (ROIs) in the image
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = image.copy() if visualize else None

    circles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Get the circle around the ROI
        scaling_factor = 1  # increase the size of the circle for merging the ROIs
        center = (x + w // 2, y + h // 2)
        radius = int(scaling_factor * max(w, h) / 2)
        circle = Circle(x=center[0], y=center[1], r=radius)
        circles.append(circle)

        if visualize:
            cv2.circle(image_with_contours, center, radius, (255, 0, 0), 1)

    return circles, image_with_contours


###################### MERGE ROIS ######################
def draw_circles(
    image: np.array, circles: List[Circle], color: Tuple[int, int, int]
) -> np.array:
    """
    show all circles on the image with numbers and centers
    """
    for i, circle in enumerate(circles):
        cv2.circle(image, (circle.x, circle.y), circle.r, color, 1)
        cv2.circle(image, (circle.x, circle.y), 2, (0, 0, 255), 2)  # Center
        cv2.putText(
            image,
            f"{i}",
            (circle.x, circle.y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return image


def create_circle_mask(circle: Circle, image_shape: Tuple[int, int]) -> np.array:
    """ Create a mask for one circle """
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.circle(mask, (circle.x, circle.y), circle.r, 255, -1)
    return mask


def merge_rois(circles: List[Circle], image: np.array, visualize=False) -> Tuple[List[Circle], np.array]:
    """
    merge intersecting circles
    """

    image_shape = image.shape[:2]
    merged_circles_mask = np.zeros(image_shape, dtype=np.uint8)

    for circle in circles:
        current_circle_mask = create_circle_mask(circle, image_shape)
        merged_circles_mask = cv2.bitwise_or(merged_circles_mask, current_circle_mask)


    contours, _ = cv2.findContours(merged_circles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    merged_circles = []
    for contour in contours:
        (x, y), r = cv2.minEnclosingCircle(contour)
        x, y, r = int(x), int(y), int(r)
        merged_circles.append(Circle(x, y, r))
        
        if visualize: cv2.circle(merged_circles_mask, (x, y), r, 255, 1)

    return merged_circles, merged_circles_mask


##################### GET LABELS ######################
def intersection_over_union(
    circle: List[Circle],
    rectangle: Rectangle,
) -> float:

    x, y, r = circle.x, circle.y, circle.r
    rectangle_corners = [
        (rectangle.x_l, rectangle.y_b),
        (rectangle.x_l, rectangle.y_t),
        (rectangle.x_r, rectangle.y_t),
        (rectangle.x_r, rectangle.y_b),
    ]
    circle = Point((x, y)).buffer(r)
    sh_rectangle = Polygon(rectangle_corners)

    intersection_area = circle.intersection(sh_rectangle).area
    union_area = circle.union(sh_rectangle).area

    iou = intersection_area / union_area

    return 0 if union_area == 0 else iou


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


def get_true_rectangles(path: str, filename: str) -> List[Rectangle]:
    # get trash labeled rectangles from json file
    true_rectangles = []

    with open(path + "jsons/" + filename + ".json") as json_file:
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


def get_labels(
    path: str,
    filename: str,
    roi_circles: List[Circle],
    iou_treshold=0.3,
    scaling_factor = 1,
    visualize=False,
    image=None,
    no_trash_warning=False
):
    """
    get trash labeled rectangles from json file
    @param iou_treshold – how much intersection is needed to label the circle as trash
    @no_trash_warning – if True, print a warning if no trash is detected in the image
    """

    true_rectangles = get_true_rectangles(path, filename)
    for i, rectangle in enumerate(true_rectangles):
        rect = Rectangle(
                          x_l=int(rectangle.x_l*scaling_factor),
                          x_r=int(rectangle.x_r*scaling_factor),
                          y_b=int(rectangle.y_b*scaling_factor),
                          y_t=int(rectangle.y_t*scaling_factor)
                        )
        true_rectangles[i] = rect

    # convert roi circles to rectangles
    roi_rectangles = []
    for circle in roi_circles:
        x, y, r = circle.x, circle.y, circle.r

        rectangle = Rectangle(max(x - r, 0), max(y - r, 0), x + r, y + r)
        roi_rectangles.append(rectangle)

    # # filter circles to get false circles
    # labels = []
    # for rectangle in trash_rectangles:
    #     for i, circle in enumerate(roi_circles):
    #         if intersection_over_union(circle, rectangle) > iou_treshold:
    #             labels.append(1)  # 1 is trash
    #         else:
    #             labels.append(0)  # 0 is not trash


    # filter circles to get false circles
    labels = [0 for _ in roi_rectangles]
    for i, roi_rectangle in enumerate(roi_rectangles):
        for trash_rectangle in true_rectangles:
            iou = iou_rectangles(trash_rectangle, roi_rectangle)
            if iou > iou_treshold:
                labels[i] = 1  # 1 is trash
                break

    if no_trash_warning:
        if sum(labels) == 0:
            print(f"No trash detected in image {filename}")
        elif sum(labels) != len(true_rectangles):
            print(f"{sum(labels)} trash detected in image {filename} but should be {len(true_rectangles)}")
    

    if not visualize:
        image_labels = None
    else:
        if image is None:
            print("You have to provide an image for labels visualization")
        else:
            image_labels = image.copy()
            # for label, circle in zip(labels, roi_circles):
            #     x, y, r = circle.x, circle.y, circle.r

            #     if label==0:
            #         cv2.circle(image_labels, (x, y), r, (255, 0, 0), 1)
            #     else:
            #         cv2.circle(image_labels, (x, y), r, (0, 255, 0), 1)

            for label, rectangle in zip(labels, roi_rectangles):
                if label==0:
                    color =  (255, 0, 0)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(
                    image_labels,
                    (rectangle.x_l, rectangle.y_b),
                    (rectangle.x_r, rectangle.y_t),
                    color,
                    1
                )
            for rectangle in true_rectangles:
                cv2.rectangle(
                    image_labels,
                    (rectangle.x_l, rectangle.y_b),
                    (rectangle.x_r, rectangle.y_t),
                    (0, 0, 255),
                    1
                )
            

    return roi_circles, roi_rectangles, labels, image_labels


################ GET FEATURE VECTORS ###################

def get_rgb_histogram_vector(img: np.array, plot=False) -> np.array:
    image = img.copy()
    hist_r, bins_r = np.histogram(image[:, :, 0], bins=50, range=(0, 256))
    hist_g, bins_g = np.histogram(image[:, :, 1], bins=50, range=(0, 256))
    hist_b, bins_b = np.histogram(image[:, :, 2], bins=50, range=(0, 256))

    if plot:
        plt.plot(bins_r[:-1], hist_r, color="r", label="Red")
        plt.plot(bins_g[:-1], hist_g, color="g", label="Green")
        plt.plot(bins_b[:-1], hist_b, color="b", label="Blue")
        plt.title("Color Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()

    return np.concatenate((hist_r, hist_g, hist_b))


def get_sift_feature_vector(img: np.array, kp, plot=False) -> np.array:
    image = img.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()

    sift_vector = sift.compute(image_gray, kp, image)

    if plot:
        image_with_keypoints = cv2.drawKeypoints(
            image_gray, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        plt.imshow(image_with_keypoints)

    return sift_vector[1][0]


############################ MAIN ##############################
def create_feature_vectors(split, dog_threshold = 0.03, iou_treshold=0.1, visualize=False, no_trash_warning=False, timing=False, with_labels=False):
    path = f"data/Dataset/{split}/"

    all_feature_vectors = []

    for jpg_file in os.listdir(path + "images/"):
        print(jpg_file)
        filename = jpg_file[:-4]
        feature_vectors, _ = create_feature_vector(filename, path, dog_threshold, iou_treshold, visualize, no_trash_warning, timing, with_labels)
        all_feature_vectors += feature_vectors

    df = pd.DataFrame(all_feature_vectors)
    df.to_csv(f'data/feature_vectors_{split}.csv', index=False)


def create_feature_vector(filename, path, dog_threshold = 0.03, iou_treshold=0.1, visualize=False, no_trash_warning=False, timing=False, with_labels=False):  
    if timing: start_cfv = time.time()

    image = io.imread(path + "images/" + filename + ".JPG")
    
    # scale image to 162x288
    orig_shape = image.shape
    new_shape = IMAGE_SHAPE
    image = cv2.resize(image, new_shape[::-1])
    assert new_shape[0]/orig_shape[0] == new_shape[1]/orig_shape[1]
    scaling_factor = new_shape[0]/orig_shape[0]

    start_roi = time.time()
    with open(path + "jsons/" + filename + ".json") as json_file:
        data = json.load(json_file)
        altitude = int(os.path.splitext(data["imagePath"])[0].split("_")[-1])

    sigma, min_filtering_threshold, max_filtering_threshold = _get_params(altitude)

    mask, dog_image = apply_dog(image, sigma, dog_threshold)
    rois, image_rois = find_rois(image, mask, visualize=visualize)

    rois = [
        circle for circle in rois if (area := np.pi * circle.r ** 2) < max_filtering_threshold and area > min_filtering_threshold
    ]  # filter rois by surface

    roi_circles, image_merged_rois = merge_rois(rois, image, visualize=visualize)
    if timing:
        end = time.time()
        print(f"Time finding and merging rois: {end - start_roi}")

    if with_labels:
        roi_circles, roi_rectangles, labels, image_labels = get_labels(
            path, filename, roi_circles, iou_treshold, scaling_factor, visualize, image, no_trash_warning
        )
    else:
        roi_rectangles = []
        for circle in roi_circles:
            x, y, r = circle.x, circle.y, circle.r
            roi_rectangles.append(Rectangle(max(x - r, 0), max(y - r, 0), x + r, y + r))
        labels = [None] * len(roi_circles)

    feature_vectors = []
    if timing: 
        print(f"Number of circles: {len(roi_circles)}")
        i = 0
    for circle, rectangle, label in zip(roi_circles, roi_rectangles, labels):
        if timing: start_vector = time.time()    
        x_circles, y_circles, r_circles = circle[0], circle[1], circle[2]

        image_part_hsv = cv2.cvtColor(image[rectangle.y_b:rectangle.y_t, rectangle.x_l:rectangle.x_r], cv2.COLOR_RGB2HSV)
        hsv_feature_vector = get_rgb_histogram_vector(
            image_part_hsv,
            plot=False,
        )

        # rgb_feature_vector = get_rgb_histogram_vector(
        #     image[rectangle.y_b:rectangle.y_t, rectangle.x_l:rectangle.x_r],
        #     plot=False,
        # )
        kp = [cv2.KeyPoint(x_circles, y_circles, 2 * r_circles)]
        sift_feature_vector = get_sift_feature_vector(image, kp, plot=False)
        
        if with_labels:
            feature_vector = np.concatenate(
                (hsv_feature_vector, sift_feature_vector, np.array([label]))
            )
        else: 
            feature_vector = np.concatenate(
                (hsv_feature_vector, sift_feature_vector)
            )

        feature_vectors.append(feature_vector)
        if timing:
            end = time.time()
            print(f"Time creating feature vector for circle {i}: {end - start_vector}")
            i += 1
    if timing:
        end = time.time()
        print(f"Time for whole processing: {end - start_cfv}")

    # show all pictures
    if visualize:
        fig, axes = plt.subplots(3, 2, figsize=(20, 8))
        ax = axes.ravel()
        ax[0].imshow(image)
        ax[0].title.set_text("Original image")
        ax[1].imshow(dog_image, cmap="gray")
        ax[1].title.set_text("Difference of gaussians")
        ax[2].imshow(image_rois)
        ax[2].title.set_text("ROIs")
        ax[3].imshow(image_merged_rois)
        ax[3].title.set_text("Filtered and merged ROIs")
        if with_labels:
            ax[4].imshow(image_labels)
            ax[4].title.set_text("Labels (green – trash, red – false, blue – provided labels)")
        else: 
            ax[4].imshow(np.zeros_like(image))
        ax[5].imshow(np.zeros_like(image))

        fig.canvas.manager.set_window_title(f"{filename} {altitude}")
        plt.tight_layout()
        plt.show()

    return feature_vectors, roi_rectangles


if __name__ == "__main__":
    create_feature_vectors("training", no_trash_warning=True, visualize=False, timing=False, with_labels=True)
    create_feature_vectors("validation", visualize=False, with_labels=True, no_trash_warning=True)
    create_feature_vectors("test", visualize=False, with_labels=True)
