import os
import cv2
import numpy as np

from skimage.exposure import match_histograms
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import time


def resize_images(images, resize_factor=1.0):
    """
    Resizes the input and reference images based on the average dimensions of the two images and a resize factor.
    Parameters:
    images (tuple): A tuple containing two images (input_image, reference_image). Both images should be numpy arrays.
    resize_factor (float): A factor by which to resize the images. Default is 1.0, which means the images will be resized to the average dimensions of the two images.
    Returns:
    tuple: A tuple containing the resized input and reference images.
    Example:
    >>> input_image = cv2.imread('input.jpg')
    >>> reference_image = cv2.imread('reference.jpg')
    >>> resized_images = resize_images((input_image, reference_image), resize_factor=0.5)
    """
    input_image, reference_image = images
    average_width = (input_image.shape[1] + reference_image.shape[1]) * 0.5
    average_height = (input_image.shape[0] + reference_image.shape[0]) * 0.5
    new_shape = (
        int(resize_factor * average_width),
        int(resize_factor * average_height),
    )

    input_image = cv2.resize(input_image, new_shape, interpolation=cv2.INTER_AREA)
    reference_image = cv2.resize(
        reference_image, new_shape, interpolation=cv2.INTER_AREA
    )

    return input_image, reference_image


def homography(images, debug=False, output_directory=None):
    """
    Apply homography transformation to align two images.
    Args:
        images (tuple): A tuple containing two images, where the first image is the input image and the second image is the reference image.
        debug (bool, optional): If True, debug images will be generated. Defaults to False.
        output_directory (str, optional): The directory to save the debug images. Defaults to None.
    Returns:
        tuple: A tuple containing the aligned input image and the reference image.
    """
    input_image, reference_image = images
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    input_keypoints, input_descriptors = sift.detectAndCompute(input_image, None)
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        reference_image, None
    )
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(reference_descriptors, input_descriptors, k=2)

    # Apply ratio test
    good_draw = []
    good_without_list = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # 0.8 = a value suggested by David G. Lowe.
            good_draw.append([m])
            good_without_list.append(m)

    # cv.drawMatchesKnn expects list of lists as matches.
    if debug:
        assert output_directory is not None, "Output directory must be provided"
        os.makedirs(output_directory, exist_ok=True)
        cv2.imwrite(
            os.path.join(output_directory, "matching.png"),
            cv2.drawMatchesKnn(
                reference_image,
                reference_keypoints,
                input_image,
                input_keypoints,
                good_draw,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            ),
        )

    # Extract location of good matches
    reference_points = np.zeros((len(good_without_list), 2), dtype=np.float32)
    input_points = reference_points.copy()

    for i, match in enumerate(good_without_list):
        input_points[i, :] = reference_keypoints[match.queryIdx].pt
        reference_points[i, :] = input_keypoints[match.trainIdx].pt

    # Find homography
    h, _ = cv2.findHomography(input_points, reference_points, cv2.RANSAC)

    # Use homography
    height, width = reference_image.shape[:2]
    white_reference_image = 255 - np.zeros(shape=reference_image.shape, dtype=np.uint8)
    white_reg = cv2.warpPerspective(white_reference_image, h, (width, height))
    blank_pixels_mask = np.any(white_reg != [255, 255, 255], axis=-1)
    reference_image_registered = cv2.warpPerspective(
        reference_image, h, (width, height)
    )
    if debug:
        assert output_directory is not None, "Output directory must be provided"
        cv2.imwrite(
            os.path.join(output_directory, "aligned.png"), reference_image_registered
        )

    input_image[blank_pixels_mask] = [0, 0, 0]
    reference_image_registered[blank_pixels_mask] = [0, 0, 0]

    return input_image, reference_image_registered


def histogram_matching(images, debug=False, output_directory=None):
    """
    Perform histogram matching between an input image and a reference image.
    Args:
        images (tuple): A tuple containing the input image and the reference image.
        debug (bool, optional): If True, save the histogram-matched image to the output directory. Defaults to False.
        output_directory (str, optional): The directory to save the histogram-matched image. Defaults to None.
    Returns:
        tuple: A tuple containing the input image and the histogram-matched reference image.
    """

    input_image, reference_image = images

    reference_image_matched = match_histograms(
        reference_image, input_image, channel_axis=-1
    )
    if debug:
        assert output_directory is not None, "Output directory must be provided"
        cv2.imwrite(
            os.path.join(output_directory, "histogram_matched.jpg"),
            reference_image_matched,
        )
    reference_image_matched = np.asarray(reference_image_matched, dtype=np.uint8)
    return input_image, reference_image_matched


def preprocess_images(images, resize_factor=1.0, debug=False, output_directory=None):
    """
    Preprocesses a list of images by performing the following steps:
    1. Resizes the images based on the given resize factor.
    2. Applies homography to align the resized images.
    3. Performs histogram matching on the aligned images.
    Args:
        images (tuple): A tuple containing the input image and the reference image.
        resize_factor (float, optional): The factor by which to resize the images. Defaults to 1.0.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        output_directory (str, optional): The directory to save the output images. Defaults to None.
    Returns:
        tuple: The preprocessed images.
    Example:
        >>> images = (input_image, reference_image)
        >>> preprocess_images(images, resize_factor=0.5, debug=True, output_directory='output/')
    """
    start_time = time.time()
    resized_images = resize_images(images, resize_factor)
    aligned_images = homography(
        resized_images, debug=debug, output_directory=output_directory
    )
    matched_images = histogram_matching(
        aligned_images, debug=debug, output_directory=output_directory
    )
    print("--- Preprocessing time - %s seconds ---" % (time.time() - start_time))
    return matched_images


# The returned vector_set goes later to the PCA algorithm which derives the EVS (Eigen Vector Space).
# Therefore, there is a mean normalization of the data
# jump_size is for iterating non-overlapping windows. This parameter should be eqaul to the window_size of the system
def find_vector_set(descriptors, jump_size, shape):
    """
    Find the vector set from the given descriptors.
    Args:
        descriptors (numpy.ndarray): The input descriptors.
        jump_size (int): The jump size for sampling the descriptors.
        shape (tuple): The shape of the descriptors.
    Returns:
        tuple: A tuple containing the vector set and the mean vector.
    """
    size_0, size_1 = shape
    descriptors_2d = descriptors.reshape((size_0, size_1, descriptors.shape[1]))
    vector_set = descriptors_2d[::jump_size, ::jump_size]
    vector_set = vector_set.reshape(
        (vector_set.shape[0] * vector_set.shape[1], vector_set.shape[2])
    )
    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec  # mean normalization
    return vector_set, mean_vec


# returns the FSV (Feature Vector Space) which then goes directly to clustering (with Kmeans)
# Multiply the data with the EVS to get the entire data in the PCA target space
def find_FVS(descriptors, EVS, mean_vec):
    """
    Calculate the feature vector space (FVS) by performing dot product of descriptors and EVS,
    and subtracting the mean vector from the result.
    Args:
        descriptors (numpy.ndarray): Array of descriptors.
        EVS (numpy.ndarray): Eigenvalue matrix.
        mean_vec (numpy.ndarray): Mean vector.
    Returns:
        numpy.ndarray: The calculated feature vector space (FVS).
    """
    FVS = np.dot(descriptors, EVS)
    FVS = FVS - mean_vec
    # print("\nfeature vector space size", FVS.shape)
    return FVS


# assumes descriptors is already flattened
# returns descriptors after moving them into the PCA vector space
def descriptors_to_pca(descriptors, pca_target_dim, window_size, shape):
    """
    Applies Principal Component Analysis (PCA) to a set of descriptors.
    Args:
        descriptors (list): List of descriptors.
        pca_target_dim (int): Target dimensionality for PCA.
        window_size (int): Size of the sliding window.
        shape (tuple): Shape of the descriptors.
    Returns:
        list: Feature vector set after applying PCA.
    """
    vector_set, mean_vec = find_vector_set(descriptors, window_size, shape)
    pca = PCA(pca_target_dim)
    pca.fit(vector_set)
    EVS = pca.components_
    mean_vec = np.dot(mean_vec, EVS.transpose())
    FVS = find_FVS(descriptors, EVS.transpose(), mean_vec)
    return FVS


def get_descriptors(
    images,
    window_size,
    pca_dim_gray,
    pca_dim_rgb,
    debug=False,
    output_directory=None,
):
    """
    Compute descriptors for input images using sliding window technique and PCA.
    Args:
        images (tuple): A tuple containing the input image and reference image.
        window_size (int): The size of the sliding window.
        pca_dim_gray (int): The number of dimensions to keep for grayscale PCA.
        pca_dim_rgb (int): The number of dimensions to keep for RGB PCA.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        output_directory (str, optional): The directory to save debug images. Required if debug is True.
    Returns:
        numpy.ndarray: The computed descriptors.
    Raises:
        AssertionError: If debug is True but output_directory is not provided.
    """
    input_image, reference_image = images

    diff_image_gray = cv2.cvtColor(
        cv2.absdiff(input_image, reference_image), cv2.COLOR_BGR2GRAY
    )

    if debug:
        assert output_directory is not None, "Output directory must be provided"
        cv2.imwrite(os.path.join(output_directory, "diff.jpg"), diff_image_gray)

    # Padding for windowing
    padded_diff_gray = np.pad(
        diff_image_gray,
        ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
        mode="constant",
    )

    # Sliding window for gray
    shape = (input_image.shape[0], input_image.shape[1], window_size, window_size)
    strides = padded_diff_gray.strides * 2
    windows_gray = np.lib.stride_tricks.as_strided(
        padded_diff_gray, shape=shape, strides=strides
    )
    descriptors_gray_diff = windows_gray.reshape(-1, window_size * window_size)

    # 3-channel RGB differences
    diff_image_r = cv2.absdiff(input_image[:, :, 0], reference_image[:, :, 0])
    diff_image_g = cv2.absdiff(input_image[:, :, 1], reference_image[:, :, 1])
    diff_image_b = cv2.absdiff(input_image[:, :, 2], reference_image[:, :, 2])

    if debug:
        assert output_directory is not None, "Output directory must be provided"
        cv2.imwrite(
            os.path.join(output_directory, "final_diff.jpg"),
            cv2.absdiff(input_image, reference_image),
        )
        cv2.imwrite(os.path.join(output_directory, "final_diff_r.jpg"), diff_image_r)
        cv2.imwrite(os.path.join(output_directory, "final_diff_g.jpg"), diff_image_g)
        cv2.imwrite(os.path.join(output_directory, "final_diff_b.jpg"), diff_image_b)

    # Padding for windowing RGB
    padded_diff_r = np.pad(
        diff_image_r,
        ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
        mode="constant",
    )
    padded_diff_g = np.pad(
        diff_image_g,
        ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
        mode="constant",
    )
    padded_diff_b = np.pad(
        diff_image_b,
        ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
        mode="constant",
    )

    # Sliding window for RGB
    windows_r = np.lib.stride_tricks.as_strided(
        padded_diff_r, shape=shape, strides=strides
    )
    windows_g = np.lib.stride_tricks.as_strided(
        padded_diff_g, shape=shape, strides=strides
    )
    windows_b = np.lib.stride_tricks.as_strided(
        padded_diff_b, shape=shape, strides=strides
    )

    descriptors_rgb_diff = np.concatenate(
        [
            windows_r.reshape(-1, window_size * window_size),
            windows_g.reshape(-1, window_size * window_size),
            windows_b.reshape(-1, window_size * window_size),
        ],
        axis=1,
    )

    # PCA on descriptors
    shape = input_image.shape[::-1][1:]  # shape = (height, width)
    descriptors_gray_diff = descriptors_to_pca(
        descriptors_gray_diff, pca_dim_gray, window_size, shape
    )
    descriptors_rgb_diff = descriptors_to_pca(
        descriptors_rgb_diff, pca_dim_rgb, window_size, shape
    )

    # Concatenate grayscale and RGB PCA results
    descriptors = np.concatenate((descriptors_gray_diff, descriptors_rgb_diff), axis=-1)

    return descriptors


def k_means_clustering(FVS, components, image_shape):
    """
    Perform K-means clustering on the given feature vectors.
    Args:
        FVS (array-like): The feature vectors to be clustered.
        components (int): The number of clusters (components) to create.
        image_shape (tuple): The size of the images used to reshape the change map.
    Returns:
        array-like: The change map obtained from the K-means clustering.
    """
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    flatten_change_map = kmeans.predict(FVS)
    change_map = np.reshape(flatten_change_map, (image_shape[0], image_shape[1]))
    return change_map


def clustering_to_mse_values(change_map, input_image, reference_image, n):
    """
    Compute the normalized mean squared error (MSE) values for each cluster in a change map.
    Args:
        change_map (numpy.ndarray): Array representing the cluster labels for each pixel in the change map.
        input_image (numpy.ndarray): Array representing the input image.
        reference_image (numpy.ndarray): Array representing the reference image.
        n (int): Number of clusters.
    Returns:
        list: Normalized MSE values for each cluster.
    """

    # Ensure the images are in integer format for calculations
    input_image = input_image.astype(int)
    reference_image = reference_image.astype(int)

    # Compute the squared differences
    squared_diff = np.mean((input_image - reference_image) ** 2, axis=-1)

    # Initialize arrays to store MSE and size for each cluster
    mse = np.zeros(n, dtype=float)
    size = np.zeros(n, dtype=int)

    # Compute the MSE and size for each cluster
    for k in range(n):
        mask = change_map == k
        size[k] = np.sum(mask)
        if size[k] > 0:
            mse[k] = np.sum(squared_diff[mask])

    # Normalize MSE values by the number of pixels and the maximum possible MSE (255^2)
    normalized_mse = (mse / size) / (255**2)

    return normalized_mse.tolist()


def compute_change_map(
    images,
    window_size,
    clusters,
    pca_dim_gray,
    pca_dim_rgb,
    debug=False,
    output_directory=None,
):
    """
    Compute the change map and mean squared error (MSE) array for a pair of input and reference images.
    Args:
        images (tuple): A tuple containing the input and reference images.
        window_size (int): The size of the sliding window for feature extraction.
        clusters (int): The number of clusters for k-means clustering.
        pca_dim_gray (int): The number of dimensions to reduce to for grayscale images.
        pca_dim_rgb (int): The number of dimensions to reduce to for RGB images.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        output_directory (str, optional): The directory to save the output files. Required if debug mode is enabled.
    Returns:
        tuple: A tuple containing the change map and MSE array.
    Raises:
        AssertionError: If debug mode is enabled but output_directory is not provided.
    """
    input_image, reference_image = images
    descriptors = get_descriptors(
        images,
        window_size,
        pca_dim_gray,
        pca_dim_rgb,
        debug=debug,
        output_directory=output_directory,
    )
    # Now we are ready for clustering!
    change_map = k_means_clustering(descriptors, clusters, input_image.shape)
    mse_array = clustering_to_mse_values(
        change_map, input_image, reference_image, clusters
    )

    colormap = mcolors.LinearSegmentedColormap.from_list(
        "custom_jet", plt.cm.jet(np.linspace(0, 1, clusters))
    )
    colors_array = (
        colormap(np.linspace(0, 1, clusters))[:, :3] * 255
    )  # Convert to RGB values

    palette = sns.color_palette("Paired", clusters)
    palette = np.array(palette) * 255  # Convert to RGB values

    # Optimized loop
    change_map_flat = change_map.ravel()
    colored_change_map_flat = (
        colors_array[change_map_flat]
        .reshape(change_map.shape[0], change_map.shape[1], 3)
        .astype(np.uint8)
    )
    palette_colored_change_map_flat = (
        palette[change_map_flat]
        .reshape(change_map.shape[0], change_map.shape[1], 3)
        .astype(np.uint8)
    )

    if debug:
        assert output_directory is not None, "Output directory must be provided"
        cv2.imwrite(
            os.path.join(
                output_directory,
                f"window_size_{window_size}_pca_dim_gray{pca_dim_gray}_pca_dim_rgb{pca_dim_rgb}_clusters_{clusters}.jpg",
            ),
            colored_change_map_flat,
        )
        cv2.imwrite(
            os.path.join(
                output_directory,
                f"PALETTE_window_size_{window_size}_pca_dim_gray{pca_dim_gray}_pca_dim_rgb{pca_dim_rgb}_clusters_{clusters}.jpg",
            ),
            palette_colored_change_map_flat,
        )

    if debug:
        assert output_directory is not None, "Output directory must be provided"
        # Saving Output for later evaluation
        np.savetxt(
            os.path.join(output_directory, "clustering_data.csv"),
            change_map,
            delimiter=",",
        )
    return change_map, mse_array


# selects the classes to be shown to the user as 'changes'.
# this selection is done by an MSE heuristic using DBSCAN clustering, to seperate the highest mse-valued classes from the others.
# the eps density parameter of DBSCAN might differ from system to system
def find_group_of_accepted_classes_DBSCAN(
    MSE_array, debug=False, output_directory=None
):
    """
    Finds the group of accepted classes using the DBSCAN algorithm.
    Parameters:
    - MSE_array (list): A list of mean squared error values.
    - debug (bool): Flag indicating whether to enable debug mode or not. Default is False.
    - output_directory (str): The directory where the output files will be saved. Default is None.
    Returns:
    - accepted_classes (list): A list of indices of the accepted classes.
    """

    clustering = DBSCAN(eps=0.02, min_samples=1).fit(np.array(MSE_array).reshape(-1, 1))
    number_of_clusters = len(set(clustering.labels_))
    if number_of_clusters == 1:
        print("No significant changes are detected.")
        
    # print(clustering.labels_)
    classes = [[] for _ in range(number_of_clusters)]
    centers = np.zeros(number_of_clusters)

    np.add.at(centers, clustering.labels_, MSE_array)

    for i in range(len(MSE_array)):
        classes[clustering.labels_[i]].append(i)

    centers /= np.array([len(c) for c in classes])

    min_class = np.argmin(centers)
    accepted_classes = np.where(clustering.labels_ != min_class)[0]

    if debug:
        assert output_directory is not None, "Output directory must be provided"
        plt.figure()
        plt.xlabel("Index")
        plt.ylabel("MSE")
        plt.scatter(range(len(MSE_array)), MSE_array, c="red")
        plt.scatter(
            accepted_classes[:],
            np.array(MSE_array)[np.array(accepted_classes)],
            c="blue",
        )
        plt.title("K Mean Classification")

        plt.savefig(os.path.join(output_directory, "mse.png"))

        # save output for later evaluation
        np.savetxt(
            os.path.join(output_directory, "accepted_classes.csv"),
            accepted_classes,
            delimiter=",",
        )
    return [accepted_classes]


def draw_combination_on_transparent_input_image(
    classes_mse, clustering, combination, transparent_input_image
):
    """
    Draws a combination of classes on a transparent input image based on their mean squared error (MSE) order.
    Args:
        classes_mse (numpy.ndarray): Array of mean squared errors for each class.
        clustering (dict): Dictionary containing the clustering information for each class.
        combination (list): List of classes to be drawn on the image.
        transparent_input_image (numpy.ndarray): Transparent input image.
    Returns:
        numpy.ndarray: Transparent input image with the specified combination of classes drawn on it.
    """

    # HEAT MAP ACCORDING TO MSE ORDER
    sorted_indexes = np.argsort(classes_mse)
    for class_ in combination:
        index = np.argwhere(sorted_indexes == class_).flatten()[0]
        c = plt.cm.jet(float(index) / (len(classes_mse) - 1))
        for [i, j] in clustering[class_]:
            transparent_input_image[i, j] = (
                c[2] * 255,
                c[1] * 255,
                c[0] * 255,
                255,
            )  # BGR
    return transparent_input_image


def detect_changes(
    images,
    output_alpha,
    window_size,
    clusters,
    pca_dim_gray,
    pca_dim_rgb,
    debug=False,
    output_directory=None,
):
    """
    Detects changes between two images using a combination of clustering and image processing techniques.
    Args:
        images (tuple): A tuple containing two input images.
        output_alpha (int): The alpha value for the output image.
        window_size (int): The size of the sliding window used for computing change map.
        clusters (int): The number of clusters used for clustering pixels.
        pca_dim_gray (int): The number of dimensions to reduce the grayscale image to using PCA.
        pca_dim_rgb (int): The number of dimensions to reduce the RGB image to using PCA.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        output_directory (str, optional): The output directory for saving intermediate results. Defaults to None.
    Returns:
        numpy.ndarray: The resulting image with detected changes.
    """
    start_time = time.time()
    input_image, _ = images
    clustering_map, mse_array = compute_change_map(
        images,
        window_size=window_size,
        clusters=clusters,
        pca_dim_gray=pca_dim_gray,
        pca_dim_rgb=pca_dim_rgb,
        debug=debug,
        output_directory=output_directory,
    )

    clustering = [np.empty((0, 2), dtype=int) for _ in range(clusters)]

    # Get the indices of each element in the clustering_map
    indices = np.indices(clustering_map.shape).transpose(1, 2, 0).reshape(-1, 2)
    flattened_map = clustering_map.flatten()

    for cluster_idx in range(clusters):
        clustering[cluster_idx] = indices[flattened_map == cluster_idx]

    b_channel, g_channel, r_channel = cv2.split(input_image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    alpha_channel[:, :] = output_alpha
    groups = find_group_of_accepted_classes_DBSCAN(mse_array, output_directory)

    for group in groups:
        transparent_input_image = cv2.merge(
            (b_channel, g_channel, r_channel, alpha_channel)
        )
        result = draw_combination_on_transparent_input_image(
            mse_array, clustering, group, transparent_input_image
        )

    print("--- Detect Changes time - %s seconds ---" % (time.time() - start_time))
    return result


def pipeline(
    images,
    resize_factor=1.0,
    output_alpha=50,
    window_size=5,
    clusters=16,
    pca_dim_gray=3,
    pca_dim_rgb=9,
    debug=False,
    output_directory=None,
):
    """
    Applies a pipeline of image processing steps to detect changes in a sequence of images.
    Args:
        images (tuple): A list of input images.
        resize_factor (float, optional): The factor by which to resize the images. Defaults to 1.0.
        output_alpha (int, optional): The alpha value for the output images. Defaults to 50.
        window_size (int, optional): The size of the sliding window for change detection. Defaults to 5.
        clusters (int, optional): The number of clusters for color quantization. Defaults to 16.
        pca_dim_gray (int, optional): The number of dimensions to keep for grayscale PCA. Defaults to 3.
        pca_dim_rgb (int, optional): The number of dimensions to keep for RGB PCA. Defaults to 9.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        output_directory (str, optional): The directory to save the output images. Defaults to None.
    Returns:
        numpy.ndarray: The resulting image with detected changes.
    """
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    preprocessed_images = preprocess_images(
        images,
        resize_factor=resize_factor,
        debug=debug,
        output_directory=output_directory,
    )
    result = detect_changes(
        preprocessed_images,
        output_alpha=output_alpha,
        window_size=window_size,
        clusters=clusters,
        pca_dim_gray=pca_dim_gray,
        pca_dim_rgb=pca_dim_rgb,
        debug=debug,
        output_directory=output_directory,
    )

    return result