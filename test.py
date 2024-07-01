import cv2
import numpy as np

def align_images(reference_image, current_frame, resize_factor=(1.0 / 4.0)):

    # Convert images to grayscale
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Resize images
    reference_image_rs = cv2.resize(
        reference_image_gray, (0, 0), fx=resize_factor, fy=resize_factor
    )
    current_frame_rs = cv2.resize(
        current_frame_gray, (0, 0), fx=resize_factor, fy=resize_factor
    )

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(current_frame_rs, None)
    keypoints2, descriptors2 = sift.detectAndCompute(reference_image_rs, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    matches = good_matches

    if len(good_matches) < 4:
        print("Not enough matches found - {}/{}".format(len(good_matches), 4))
        return current_frame

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    if H is None:
        print("Homography calculation failed.")
        return current_frame

    # Warp the current frame to match the reference image
    aligned_frame = cv2.warpPerspective(
        current_frame, H, (reference_image.shape[1], reference_image.shape[0])
    )

    return aligned_frame

# Read images
reference_image = cv2.imread("img1.jpg")
current_frame = cv2.imread("img2.jpg")

# Align images
aligned_image = align_images(reference_image, current_frame)

# Display the image
cv2.imwrite("aligned_image.jpg", aligned_image)