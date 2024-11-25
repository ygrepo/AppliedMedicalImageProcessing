import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import denseCRF
import matplotlib
matplotlib.use('TkAgg') 

def load_and_show_slice(image_path, slice_index=143):
    """
    Load an MRI image and return the processed volume as a NumPy array.
    """
    # Load the NIfTI image
    image = sitk.ReadImage(image_path)

    # Convert to NumPy array (Shape: [z, y, x])
    image_array = sitk.GetArrayFromImage(image)

    # Rearrange axes to MATLAB equivalent: permute(vol, [2, 1, 3])
    image_array = np.transpose(image_array, (2, 1, 0))  # Shape becomes [y, x, z]

    # Flip along the first axis: flipdim(vol, 1)
    image_array = np.flip(image_array, axis=0)

    return image_array


def perform_clustering(image_array, slice_index=143, num_clusters=4):
    """
    Perform K-means clustering on a specific slice of an image.
    Returns the original slice and the segmented slice.
    """
    # Extract the requested slice
    selected_slice = image_array[:, :, slice_index]

    # Flatten the slice for clustering
    brain_pixels = selected_slice.flatten().reshape(-1, 1)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    print(f"Number of clusters: {num_clusters}")
    labels = kmeans.fit_predict(brain_pixels)

    # Reshape labels to the original slice dimensions
    segmented_slice = labels.reshape(selected_slice.shape)

    return selected_slice, segmented_slice


def refine_segmentation_with_densecrf(original_slice, segmented_slice, num_clusters=4):
    """
    Refine segmentation using Dense CRF.

    Args:
        original_slice (np.ndarray): Original 2D image slice.
        segmented_slice (np.ndarray): Initial segmentation of the slice.
        num_clusters (int): Number of clusters/classes.

    Returns:
        np.ndarray: Refined segmentation with Dense CRF.
    """
    # Normalize the original slice to 0-255 and cast to uint8 for DenseCRF
    original_image = (
        (original_slice - original_slice.min())
        / (original_slice.max() - original_slice.min())
        * 255
    ).astype(np.uint8)

    # Convert segmented slice to probability map
    H, W = segmented_slice.shape
    prob_map = np.zeros((H, W, num_clusters), dtype=np.float32)
    for c in range(num_clusters):
        prob_map[:, :, c] = (segmented_slice == c).astype(np.float32)

    # Define Dense CRF parameters
    w1, alpha, beta = 10.0, 80, 13
    w2, gamma, it = 3.0, 3, 5
    param = (w1, alpha, beta, w2, gamma, it)

    # Apply Dense CRF
    refined_segmentation = denseCRF.densecrf(original_image, prob_map, param)

    return refined_segmentation


def main():
    """
    Main function to load an image, perform clustering, refine segmentation with Dense CRF, and plot results.
    """
    # Parameters
    image_path = "data/sub-13_T1w.nii.gz"  # Path to the MRI image
    slice_index = 143  # Index of the slice to process
    num_clusters = 5  # Number of clusters for K-means

    # Load the image
    image_array = load_and_show_slice(image_path, slice_index)

    # Perform clustering on the selected slice
    original_slice, segmented_slice = perform_clustering(
        image_array, slice_index, num_clusters
    )

    # Refine segmentation with Dense CRF
    refined_segmentation = refine_segmentation_with_densecrf(
        original_slice, segmented_slice, num_clusters
    )

    # Plot results
    plt.ioff()  # Turn off interactive mode

    plt.figure(figsize=(12, 6))

    # Original slice
    plt.subplot(1, 3, 1)
    plt.imshow(original_slice, cmap="gray", origin="lower")
    plt.title("Original Slice")
    plt.colorbar()

    # Initial segmentation
    plt.subplot(1, 3, 2)
    plt.imshow(segmented_slice, cmap="jet", origin="lower")
    plt.title("Initial Segmentation (K-means)")
    plt.colorbar()

    # Refined segmentation
    plt.subplot(1, 3, 3)
    plt.imshow(refined_segmentation, cmap="jet", origin="lower")
    plt.title("Refined Segmentation (Dense CRF)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# Run the main function
if __name__ == "__main__":
    main()
