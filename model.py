import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import Point, Polygon
from skimage import io, color, img_as_float
from skimage.filters import difference_of_gaussians
from sympy import symbols, Eq, solve, solveset
from typing import List, Tuple
from collections import namedtuple

Rectangle = namedtuple("Rectangle", ["x_l", "y_b", "x_r", "y_t"])
Circle = namedtuple("Circle", ["x", "y", "r"])


def _get_sigma(altitude: int) -> int:
    """
    f for choosing the σ value based on the altitude h:
    """
    if altitude <= 5:
        sigma = 6
    elif altitude <= 10:
        sigma = 5
    elif altitude <= 15:
        sigma = 4
    elif altitude <= 20:
        sigma = 3
    else:
        sigma = 2
    return sigma


def apply_dog(image: np.array, altitude: int) -> Tuple[np.array, np.array]:
    """
    Apply difference of gaussians
    """
    sigma = _get_sigma(altitude)

    gray_image = color.rgb2gray(image)
    float_image = img_as_float(gray_image)

    dog_image = difference_of_gaussians(float_image, sigma, 1.6 * sigma)

    threshold = 0.03

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
        scaling_factor = 1.5  # increase the size of the circle for merging the ROIs
        center = (x + w // 2, y + h // 2)
        radius = int(scaling_factor * max(w, h) / 2)
        circle = Circle(x=center[0], y=center[1], r=radius)
        circles.append(circle)

        if visualize:
            cv2.circle(image_with_contours, center, radius, (255, 0, 0), 2)

    return circles, image_with_contours


###################### MERGE ROIS ######################
def draw_circles(
    image: np.array, circles: List[Circle], color: Tuple[int, int, int]
) -> np.array:
    """
    show all circles on the image
    """
    for i, circle in enumerate(circles):
        cv2.circle(image, (circle.x, circle.y), circle.r, color, 2)
        cv2.circle(image, (circle.x, circle.y), 2, (0, 0, 255), 3)  # Center
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


def circles_intersect(circle1: Circle, circle2: Circle) -> Tuple[bool, float]:
    """
    Check if two circles intersect
    """
    dist_centers = np.sqrt((circle2.x - circle1.x) ** 2 + (circle2.y - circle1.y) ** 2)
    return dist_centers <= (circle1.r + circle2.r), dist_centers


def circle_in_circle(
    circle1: Circle, circle2: Circle, dist_centers: float
) -> Tuple[int, int, int]:
    """
    Check if circle1 is inside circle2 or vice versa
    """
    if dist_centers + circle2.r <= circle1.r:
        return circle1
    if dist_centers + circle1.r <= circle2.r:
        return circle2
    else:
        return False


def find_new_center(circle1: Circle, circle2: Circle) -> Tuple[int, int]:
    """
    find a center of a new circle
    """

    x, y = symbols('x y')

    slope = (circle2.y - circle1.y) / (circle2.x - circle1.x)
    intercept = circle1.y - slope * circle1.x
    line_eq = Eq(y, slope * x + intercept)
    circle_eq = Eq((x - circle1.x)**2 + (y - circle1.y)**2, circle1.r**2)
    intersection_points1 = solve((line_eq, circle_eq), (x, y))

    slope = (circle2.y - circle1.y) / (circle2.x - circle1.x)
    intercept = circle1.y - slope * circle1.x
    line_eq = Eq(y, slope * x + intercept)
    circle_eq = Eq((x - circle2.x)**2 + (y - circle2.y)**2, circle2.r**2)
    intersection_points2 = solve((line_eq, circle_eq), (x, y))

    intersection_points = intersection_points1 + intersection_points2
    intersection_points.sort(key=lambda p: p[0] + p[1])
    intersection_points

    x = (intersection_points[0][0] + intersection_points[-1][0]) // 2
    y = (intersection_points[0][1] + intersection_points[-1][1]) // 2

    return (int(x), int(y))


def show_merge(
    image: np.array,
    center1: float,
    r1: float,
    center2: float,
    r2: float,
    new_center: float,
    new_radius: float,
):
    """
    show circles before and after merging
    """
    blank = np.zeros_like(image)

    cv2.circle(blank, center1, r1, (255, 0, 0), 2)
    cv2.circle(blank, center2, r2, (255, 0, 0), 2)
    cv2.circle(blank, new_center, new_radius, (0, 255, 0), 2)
    plt.imshow(blank)
    plt.show()


def merge_rois(circles: List[Circle], image: np.array, visualize=False):
    """
    marge intersecting circles
    """
    roi_image = image.copy() if visualize else None

    # if visualize:
    #     roi_image = draw_circles(roi_image, circles, (0, 255, 0))

    # Greedy grouping algorithm to merge intersecting circles
    changed = True
    while changed == True:
        changed = False
        i = 0
        while i < len(circles):
            j = i + 1
            while j < len(circles):
                intersect, distance = circles_intersect(circles[i], circles[j])
                circle1 = circles[i]
                circle2 = circles[j]

                if intersect:
                    # Merge circles[i] and circles[j]
                    if circle := circle_in_circle(circle1, circle2, distance):
                        new_circle = circle
                    else:
                        new_radius = int((circle1.r + circle2.r + distance) // 2)
                        new_x, new_y = find_new_center(circles[i], circles[j])
                        new_circle = Circle(x=new_x, y=new_y, r=new_radius)

                    # show_merge(roi_image, (x1, y1), r1, (x2, y2), r2, (new_x, new_y), new_radius)
                    circles[i] = new_circle
                    circles.pop(j)
                    changed = True

                else:
                    j += 1
            i += 1

    # Draw circles around the final merged circles
    if visualize:
        roi_image = draw_circles(roi_image, circles, (255, 0, 0))

    return circles, roi_image


###################### MERGE ROIS AS RECTANGLES ######################
def circles_to_squares(circles: List[Circle]) -> List[Rectangle]:
    """
    map circles to squares
    """
    return [
        Rectangle(
            x_l=circle.x - circle.r,
            y_b=circle.y - circle.r,
            x_r=circle.x + circle.r,
            y_t=circle.y + circle.r,
        )
        for circle in circles
    ]


def draw_rectangles(
    image: np.array,
    rectangles: List[Rectangle],
    color=(0, 255, 0),
) -> np.array:
    for rectangle in rectangles:
        cv2.rectangle(
            image,
            (rectangle.x_l, rectangle.y_b),
            (rectangle.x_r, rectangle.y_t),
            color,
            2,
        )
    return image


def rectangles_intersect(
    rect1: Rectangle,
    rect2: Rectangle,
) -> bool:

    # TODO bug was here? Two times x_distance
    x_distance = min(rect1.x_r, rect2.x_r) - max(rect1.x_l, rect2.x_l)
    y_distance = min(rect1.y_t, rect2.y_t) - max(rect1.y_b, rect2.y_b)
    return x_distance * y_distance > 0


def merge_rois_to_rectangle(
    roi_circles: List[Circle], image: np.array, visualize=False
) -> Tuple[List[Rectangle], np.array]:
    image_with_merges = np.zeros_like(image) if visualize else None

    if visualize:
        draw_circles(image_with_merges, roi_circles, (0, 255, 0))

    rectangles = circles_to_squares(roi_circles)

    # Greedy grouping algorithm to merge intersecting circles
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
                    changed = True

                else:
                    j += 1
            i += 1

    if visualize:
        # Draw circles around the final merged circles
        draw_rectangles(image_with_merges, rectangles, (255, 0, 0))

    return rectangles, image_with_merges


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
    rectangle = Polygon(rectangle_corners)

    intersection_area = circle.intersection(rectangle).area
    union_area = circle.union(rectangle).area

    iou = intersection_area / union_area

    return 0 if union_area == 0 else iou


def get_labels(
    path: str,
    filename: str,
    roi_circles: List[Circle],
    visualize=False,
    image=None,
):
    """
    get trash labeled rectangles from json file
    """
    trash_rectangles = []

    with open(path + filename + ".json") as json_file:
        data = json.load(json_file)
        shapes = data["shapes"]

        for shape in shapes:
            if shape["shape_type"] != "rectangle":
                raise Exception("Invalid shape type in", filename)
            if shape["label"] != "trash":
                raise Exception("Invalid label in", filename)

            bl, tr = shape["points"]
            bl = tuple(map(int, bl))
            tr = tuple(map(int, tr))

            rect = Rectangle(x_l=bl[0], y_b=bl[1], x_r=tr[0], y_t=tr[1])
            trash_rectangles.append(rect)

    rectangles = []
    for circle in roi_circles:
        x, y, r = circle.x, circle.y, circle.r

        rectangle = Rectangle(x - r, y - r, x + r, y + r)
        rectangles.append(rectangle)

    # filter circles to get false circles
    labels = []
    for rectangle in trash_rectangles:
        for i, circle in enumerate(roi_circles):
            if intersection_over_union(circle, rectangle) > 0.3:
                labels.append(1)  # 1 is trash
            else:
                labels.append(0)  # 0 is not trash

    if not visualize:
        image_labels = None
    else:
        if image is None:
            print("You have to provide an image for labels visualization")
        else:
            image_labels = image.copy()
            for label, circle in zip(labels, roi_circles):
                x, y, r = circle.x, circle.y, circle.r

                if label==0:
                    cv2.circle(image_labels, (x, y), r, (255, 0, 0), 3)
                else:
                    cv2.circle(image_labels, (x, y), r, (0, 255, 0), 3)

    return roi_circles, rectangles, labels, image_labels


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


def get_sift_feture_vector(img: np.array, kp, plot=False) -> np.array:
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


if __name__ == "__main__":
    path = "data/Dataset/training/"
    visualize = True

    jpg_files = [f for f in os.listdir(path) if f.endswith(".JPG")]

    for jpg_file in jpg_files:
        print(jpg_file)
        filename = jpg_file[:-4]
        image = io.imread(path + filename + ".JPG")

        with open(path + filename + ".json") as json_file:
            data = json.load(json_file)
            altitude = int(os.path.splitext(data["imagePath"])[0].split("_")[-1])

        mask, dog_image = apply_dog(image, altitude)
        rois, image_rois = find_rois(image, mask, visualize=visualize)
        rois = [
            circle for circle in rois if np.pi * circle[2] ** 2 > 1000
        ]  # filter rois by surface

        rois, image_merged_rois = merge_rois(rois, image, visualize=visualize)

        roi_circles, rectangles, labels, image_labels = get_labels(
            path, filename, rois, visualize, image
        )

        for circle, rectangle, label in zip(roi_circles, rectangles, labels):
            x_circles, y_circles, r_circles = circle[0], circle[1], circle[2]

            rgb_feture_vector = get_rgb_histogram_vector(
                image[rectangle.x_l : rectangle.x_r, rectangle.y_b : rectangle.y_t],
                plot=False,
            )

            kp = [cv2.KeyPoint(x_circles, y_circles, 2 * r_circles)]
            sift_feture_vector = get_sift_feture_vector(image, kp, plot=False)
            feture_vector = np.concatenate(
                (rgb_feture_vector, sift_feture_vector, np.array([label]))
            )

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
            ax[4].imshow(image_labels)
            ax[4].title.set_text("Labels (green – trash, red – false)")
            ax[5].imshow(np.zeros_like(image))

            plt.tight_layout()
            plt.show()
