from skimage.metrics import structural_similarity
import cv2
import numpy as np

def load_images(left_path, right_path, target_height=640, target_width=480):
    # Load images
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    # Resize images
    left_resized = cv2.resize(left, (target_width, target_height))
    right_resized = cv2.resize(right, (target_width, target_height))

    return left_resized, right_resized

def reduce_noise(image, kernel_size=(5, 5)):
    image = cv2.GaussianBlur(image, kernel_size, 0)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_morphology(image, kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def compute_ssim(image1, image2, sigma=1.5):
    return structural_similarity(image1, image2, full=True, gaussian_weights=True, sigma=sigma)

def threshold_diff(diff_image):
    thresh = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

def find_contours(thresh_image):
    return cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

def process_contours(contours, before, after, diff_box, mask, filled_after, min_area=500):
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)


def main():
    left_path = 'images/pcb3-a.jpg'
    right_path = 'images/pcb3-b.jpg'

    # Load images
    before, after = load_images(left_path, right_path)

    # Reduce noise in images
    before = reduce_noise(before)
    after = reduce_noise(after)

    # Convert images to grayscale
    before_gray = convert_to_gray(before)
    after_gray = convert_to_gray(after)

    # Apply morphology to grayscale images to reduce edge sensitivity
    before_gray = apply_morphology(before_gray)
    after_gray = apply_morphology(after_gray)

    # Compute SSIM between the two images
    score, diff = compute_ssim(before_gray, after_gray, sigma=1.5)
    print("Image Similarity: {:.4f}%".format(score * 100))

    # Convert diff to 8-bit unsigned integers
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image and find contours
    thresh = threshold_diff(diff)
    contours = find_contours(thresh)

    # Process contours
    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    process_contours(contours, before, after, diff_box, mask, filled_after)

    # Display images
    # cv2.imshow('before', before)
    cv2.imshow('after', after)
    cv2.imshow('diff', diff)
    # cv2.imshow('diff_box', diff_box)
    cv2.imshow('mask', mask)
    # cv2.imshow('filled after', filled_after)
    cv2.waitKey()

if __name__ == "__main__":
    main()
