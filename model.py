import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import Point, Polygon
from skimage import io, color, img_as_float
from skimage.filters import difference_of_gaussians
from sympy import symbols, Eq, solve


def _get_sigma(altitude):
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

def apply_dog(image, altitude):
    ### Apply difference of gaussians
    sigma = _get_sigma(altitude)

    gray_image = color.rgb2gray(image)
    float_image = img_as_float(gray_image)

    dog_image = difference_of_gaussians(float_image, sigma, 1.6*sigma)

    threshold = 0.03

    mask = dog_image.copy()
    _, mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    return mask, dog_image

def find_rois(image, mask, visualize=False):
    ### Find ROIs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = image.copy() if visualize else None

    circles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Get the circle around the ROI
        scaling_factor = 1.5 # increase the size of the circle for merging the ROIs
        center = (x + w // 2, y + h // 2)
        radius = int(scaling_factor * max(w, h) / 2)
        circles.append((*center, radius))

        if visualize:
            cv2.circle(image_with_contours, center, radius, (255, 0, 0), 2)

    return circles, image_with_contours

###################### MERGE ROIS ######################
def draw_circles(image, circles, color):
    # show all circles on the image
    for i, circle in enumerate(circles):
        x, y, r = circle
        cv2.circle(image, (x, y), r, color, 2)
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)  # Center
        cv2.putText(image, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def circles_intersect(circle1, circle2):
    # Check if two circles intersect
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    dist_centers = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist_centers <= (r1 + r2), dist_centers


def circle_in_circle(circle1, circle2, dist_centers):
    # Check if circle1 is inside circle2 or vice versa
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    if dist_centers + r2 <= r1:
        return circle1
    if dist_centers + r1 <= r2:
        return circle2
    else:
        return False
    

def find_new_center(circle1, circle2):
    # find a center of a new circle
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    x, y = symbols('x y')

    # Prosta przechodząca przez 2 punkty
    m = (y2 - y1) / (x2 - x1)
    line_eq = Eq(y - y1, m*(x - x1))

    # Równanie okręgu: (x - a)^2 + (y - b)^2 = r^2
    circle_eq1 = Eq((x - x1)**2 + (y - y1)**2, r1**2)
    circle_eq2 = Eq((x - x2)**2 + (y - y2)**2, r2**2)

    # Rozwiązanie równań
    solutions1 = solve((line_eq, circle_eq1), (x, y))
    solutions2 = solve((line_eq, circle_eq2), (x, y))
    solutions = solutions1 + solutions2

    solutions.sort(key=lambda p: p[0] + p[1])

    x = (solutions[0][0] + solutions[-1][0])//2
    y = (solutions[0][1] + solutions[-1][1])//2

    return (int(x), int(y))
    

def show_merge(image, center1, r1, center2, r2, new_center, new_radius):
    # show circles before and after merging
    blank = np.zeros_like(image)

    cv2.circle(blank, center1, r1, (255, 0, 0), 2)
    cv2.circle(blank, center2, r2, (255, 0, 0), 2)
    cv2.circle(blank, new_center, new_radius, (0, 255, 0), 2)
    plt.imshow(blank);
    plt.show()


def merge_rois(circles, image, visualize=False):
    # Create a black image
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
                    _, _, r1 = circles[i]
                    _, _, r2 = circles[j]
                    if intersect:
                        # Merge circles[i] and circles[j]
                        if circle:= circle_in_circle(circles[i], circles[j], distance):
                            new_x, new_y, new_radius = circle
                        else:
                            new_radius = int((r1 + r2 + distance)//2)
                            new_x, new_y = find_new_center(circles[i], circles[j])
                        
                        # show_merge(roi_image, (x1, y1), r1, (x2, y2), r2, (new_x, new_y), new_radius)
                        circles[i] = (new_x, new_y, new_radius)
                        circles.pop(j)
                        changed = True

                    else:
                        j+=1
                i += 1
                        
    # Draw circles around the final merged circles
    if visualize:
        roi_image = draw_circles(roi_image, circles, (255, 0 , 0))

    return circles, roi_image

##################### GET LABELS ######################

def intersection_over_union(circle, rectangle):
    x, y, r = circle
    bl, tr = rectangle
    rectangle_corners = [bl, (bl[0], tr[1]), tr, (tr[0], bl[1])]

    circle = Point((x, y)).buffer(r)
    rectangle = Polygon(rectangle_corners)
    
    intersection_area = circle.intersection(rectangle).area
    union_area = circle.union(rectangle).area
    
    iou = intersection_area / union_area
    
    return 0 if union_area == 0 else iou

def get_labels(path, filename, roi_circles, visualize=False, image=None):

    # get trash labeled rectangles from json file
    trash_rectangles = []

    with open(path + filename + ".json") as json_file:
        data = json.load(json_file)
        shapes = data["shapes"]

        for shape in shapes: 
            if shape["shape_type"] != "rectangle": raise Exception("Invalid shape type in", filename)
            if shape["label"] != "trash": raise Exception("Invalid label in", filename)
            
            bl, tr = shape["points"]
            bl = tuple(map(int, bl))
            tr = tuple(map(int, tr))

            trash_rectangles.append((bl, tr))

    rectangles = []
    for circle in roi_circles:
        x, y, r = circle
        rectangles.append(((x-r, y-r), (x+r, y+r)))

    # filter circles to get false circles
    labels = []

    for rectangle in trash_rectangles:
        for i, circle in enumerate(roi_circles):
            if intersection_over_union(circle, rectangle) > 0.3:
                labels.append(1) # 1 is trash 
            else:
                labels.append(0) # 0 is not trash 

    if not visualize:
        image_labels = None
    else:
        if image is None:
            print("You have to provide an image for labels visualization")
        else:
            image_labels = image.copy()
            for label, circle in zip(labels, roi_circles):
                x, y, r = circle[0], circle[1], circle[2]

                if label:
                    cv2.circle(image, (x, y), r, (255, 0, 0), 3)
                else:
                    cv2.circle(image, (x, y), r, (0, 255, 0), 3)

    return roi_circles, rectangles, labels, image_labels


def get_rgb_histogram_vector(img: np.array, plot=False)->np.array:
    image = img.copy()
    hist_r, bins_r = np.histogram(image[:, :, 0], bins=50, range=(0, 256))
    hist_g, bins_g = np.histogram(image[:, :, 1], bins=50, range=(0, 256))
    hist_b, bins_b = np.histogram(image[:, :, 2], bins=50, range=(0, 256))

    if plot:
        plt.plot(bins_r[:-1], hist_r, color='r', label='Red')
        plt.plot(bins_g[:-1], hist_g, color='g', label='Green')
        plt.plot(bins_b[:-1], hist_b, color='b', label='Blue')
        plt.title('Color Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()


    return np.concatenate((hist_r, hist_g, hist_b))

def get_sift_feture_vector(img: np.array, kp, plot=False)->np.array:
    image = img.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()

    sift_vector = sift.compute(image_gray, kp, image)

    if plot:
        image_with_keypoints = cv2.drawKeypoints(image_gray, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(image_with_keypoints)

    return sift_vector[1][0]


if __name__ == "__main__":
    path = "data/Dataset/training/"
    filename = "train_image1"
    image = io.imread(path + filename + ".JPG")
    visualize = False

    with open(path + filename + ".json") as json_file:
        data = json.load(json_file)
        altitude = int(os.path.splitext(data["imagePath"])[0].split('_')[-1])

    mask, dog_image = apply_dog(image, altitude)
    rois, image_rois = find_rois(image, mask, visualize=visualize)
    rois = [circle for circle in rois if np.pi*circle[2]**2 > 1000] # filter rois by surface
    rois, image_merged_rois = merge_rois(rois, image, visualize=visualize)
    roi_circles, rectangles, labels, image_labels = get_labels(path, filename, rois, visualize, image)


    for circle, rectangle, label in zip(roi_circles, rectangles, labels):
        x_circles, y_circles, r_circles = circle[0], circle[1], circle[2]
        x1_rec, x2_rec, y1_rec, y2_rec = rectangle[0][1], rectangle[1][1], rectangle[0][0], rectangle[1][0]

        rgb_feture_vector = get_rgb_histogram_vector(image[x1_rec: x2_rec, y1_rec: y2_rec], plot=visualize)

        kp = [cv2.KeyPoint(x_circles, y_circles, 2*r_circles)]
        sift_feture_vector = get_sift_feture_vector(image, kp, plot=visualize)
        feture_vector = np.concatenate((rgb_feture_vector, sift_feture_vector, np.array([label])))

    # show all pictures
    if visualize:
        fig, axes = plt.subplots(3, 2, figsize=(20, 8))
        ax = axes.ravel()
        ax[0].imshow(image)
        ax[0].title.set_text("Original image")
        ax[1].imshow(dog_image, cmap='gray')
        ax[1].title.set_text("Difference of gaussians")
        ax[2].imshow(image_rois)
        ax[2].title.set_text("ROIs")
        ax[3].imshow(image_merged_rois)
        ax[3].title.set_text("Merged ROIs")
        ax[4].imshow(image_labels)
        ax[4].title.set_text("Labels (green – trash, red – false)")
        ax[5].imshow(np.zeros_like(image))

        plt.tight_layout()
        plt.show()