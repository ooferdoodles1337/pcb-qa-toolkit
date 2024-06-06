from skimage.metrics import structural_similarity
import cv2
import numpy as np

# Parameters
LEFT_IMAGE_PATH = "images/pcb1-a.jpg"
RIGHT_IMAGE_PATH = "images/pcb1-b.jpg"
TARGET_HEIGHT = 640
TARGET_WIDTH = 480
NOISE_H = 3
NOISE_TEMPLATE_WINDOW_SIZE = 7
NOISE_SEARCH_WINDOW_SIZE = 21
SSIM_SIGMA = 1.5
MIN_CONTOUR_AREA = 300
STRUCTURING_ELEMENT_SIZE = (5, 5)

def load_images(left_path, right_path, target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH):
    # Load images
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    # Resize images
    left_resized = cv2.resize(left, (target_width, target_height))
    right_resized = cv2.resize(right, (target_width, target_height))

    return left_resized, right_resized

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def reduce_noise(image, h=NOISE_H, template_window_size=NOISE_TEMPLATE_WINDOW_SIZE, search_window_size=NOISE_SEARCH_WINDOW_SIZE):
    return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)

def compute_ssim(image1, image2, sigma=SSIM_SIGMA):
    return structural_similarity(
        image1, image2, full=True, gaussian_weights=True, sigma=sigma
    )

def threshold_diff(diff_image):
    thresh = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, STRUCTURING_ELEMENT_SIZE)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

def find_contours(thresh_image):
    return cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

def process_contours(contours, left, right, diff_box, mask, filled_right, min_area=MIN_CONTOUR_AREA):
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(left, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(right, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
            cv2.drawContours(filled_right, [c], 0, (0, 255, 0), -1)

def main():
    # Load images
    left, right = load_images(LEFT_IMAGE_PATH, RIGHT_IMAGE_PATH)

    # Convert images to grayscale
    left_gray = convert_to_gray(left)
    right_gray = convert_to_gray(right)

    # Reduce noise in images
    left_gray = reduce_noise(left_gray)
    right_gray = reduce_noise(right_gray)

    cv2.imshow("left_gray", left_gray)

    # Compute SSIM between the two images
    score, diff = compute_ssim(left_gray, right_gray)
    print("Image Similarity: {:.4f}%".format(score * 100))

    # Convert diff to 8-bit unsigned integers
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image and find contours
    thresh = threshold_diff(diff)
    contours = find_contours(thresh)

    # Process contours
    mask = np.zeros(left.shape, dtype="uint8")
    filled_right = right.copy()
    process_contours(contours, left, right, diff_box, mask, filled_right)

    # Display images
    cv2.imshow("right", right)
    cv2.imshow("diff", diff)
    cv2.imshow("mask", mask)
    cv2.waitKey()

if __name__ == "__main__":
    main()
