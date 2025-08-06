import numpy as np

def pad_array(anomaly_array, patch_size=15):
    """
    This function will pad the anomaly score array to shape (64x64) i.e. starmen image shape
    """
    # Compute padding
    pad_size = patch_size // 2

    # Pad with zeros (or change to np.nan if you prefer)
    padded_anomaly_array = np.pad(anomaly_array,
                    ((pad_size, pad_size), (pad_size, pad_size)),
                    mode='constant', constant_values=0.0)

    return padded_anomaly_array

def compute_pixel_counting():
    """
    This function returns a numpy array where the element i,j gives the number of patches containing that pixel
    """
    
    pixel_relevant_count = np.ones((50,50))
    pixel_relevant_count = pad_array(pixel_relevant_count)
    pixel_count = np.zeros((64,64))
    for i in range(64):
        for j in range(64):
            # Build the window to get all the anomaly patches
            top = max(i - 15//2, 0)
            bottom = min(i + 15//2 + 1, 64)
            left = max(j - 15//2, 0)
            right = min(j + 15//2 + 1, 64)

            pixel_count[i,j] += np.sum(pixel_relevant_count[top:bottom , left:right])
    
    return pixel_count

pixel_counting = compute_pixel_counting()

def patch_to_image(patch_array, patch_size=15):
    """
    This function will take an array of patches and return the original image.
    This function assumes that patch_array contains all the possible extracted patches of the image in the right order,
    i.e. for 64x64 image and patch_size=15, 2500 patches from left to right, top to bottom.
    """
    half = patch_size//2
    image = np.zeros((64,64))
    patch_num = 0
    for i in range(half, 64-half):
        for j in range(half, 64-half):
            top = max(i - half, 0)
            bottom = min(i + half + 1, 64)
            left = max(j - half, 0)
            right = min(j + half + 1, 64)

            image[top:bottom, left:right] += patch_array[patch_num]
            patch_num += 1

    return image/pixel_counting


def patch_contour_to_image(patch_array, centers, patch_size=15):
    """
    This function takes the patches and associated centers and will sum the pixel values of the patches on the reconstructed image.
    We assume that pixel that are never contained in a patch have value 0.
    """
    half = patch_size//2
    reconstructed_image = np.zeros((64,64))
    pixel_count_mask = np.zeros((64,64), dtype=int)
    for patch_num in range(centers.shape[0]):
        x, y = centers[patch_num]
        top = max(x - half, 0)
        bottom = min(x + half + 1, 64)
        left = max(y - half, 0)
        right = min(y + half + 1, 64)

        reconstructed_image[top:bottom, left:right] += patch_array[patch_num]
        pixel_count_mask[top:bottom, left:right] += 1

    np.divide(reconstructed_image, pixel_count_mask, out=reconstructed_image, where=pixel_count_mask>0)
    return reconstructed_image


