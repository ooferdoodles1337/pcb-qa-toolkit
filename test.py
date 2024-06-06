from skimage.metrics import structural_similarity
import cv2
import numpy as np

def load_images(left_path, right_path):
    # Load images
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    # Resize images to 480p
    width = 640  # Original width
    height = 480  # Original height
    dim = (width, height)

    left_resized = cv2.resize(left, dim, interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right, dim, interpolation=cv2.INTER_AREA)

    return left_resized, right_resized

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compute_ssim(image1, image2):
    return structural_similarity(image1, image2, full=True)

def threshold_diff(diff_image):
    return cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

def find_contours(thresh_image):
    return cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

def process_contours(contours, before, after, diff_box, mask, filled_after):
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

def main():
    left_path = 'images/test6-a.jpg'
    right_path = 'images/test6-b.jpg'

    # Load images
    before, after = load_images(left_path, right_path)

    # Convert images to grayscale
    before_gray = convert_to_gray(before)
    after_gray = convert_to_gray(after)

    # Compute SSIM between the two images
    score, diff = compute_ssim(before_gray, after_gray)
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
    cv2.imshow('before', before)
    cv2.imshow('after', after)
    cv2.imshow('diff', diff)
    cv2.imshow('diff_box', diff_box)
    cv2.imshow('mask', mask)
    cv2.imshow('filled after', filled_after)
    cv2.waitKey()

if __name__ == "__main__":
    main()
