import chromaprint
import numpy as np
import matplotlib.pyplot as plt

from config import *


def plot_image(array, info=None):
    plt.figure(figsize=(20, 10))
    title = 'Binary representation of a Chromaprint'
    if info is not None:
        title = f"{title} for {info}"
    plt.title(title)
    plt.imshow(array.T, aspect='auto', origin='lower')
    plt.show()
    plt.close()


def get_array_from_fingerprint_encoded(fp, offset=0, debug=False, info=None):

    fp_int = chromaprint.decode_fingerprint(fp)[0]

    fb_bin = [list('{:032b}'.format(abs(x))) for x  in fp_int] # Int to unsigned 32-bit array

    arr = np.zeros([len(fb_bin), len(fb_bin[0])])

    for i in range(arr.shape[0]):
        arr[i,0] = int(fp_int[i] > 0) # The sign is added to the first bit
        for j in range(1, arr.shape[1]):
            arr[i,j] = float(fb_bin[i][j])

    if offset > 0:
        image_array = arr

        # Get the last 10 rows (bottom 10 horizontal lines)
        bottom_rows = image_array[-offset:, :]

        # Shift the entire image 10 pixels up
        shifted_img = np.roll(image_array, offset, axis=0)

        # Replace the first 10 rows with the bottom 10 rows
        shifted_img[:offset, :] = bottom_rows

        # Convert the shifted array back to an image
        arr = shifted_img

    if SHOW_PLOTS:
        plot_image(arr, info=info)
    return arr


def get_array_from_image(image, debug=False, info=None):
    # Convert the image to grayscale (black and white)
    gray_image = image.convert('L')

    # Convert the grayscale image to a NumPy array
    image_array = np.array(gray_image)
    image_array = np.fliplr(image_array)

    if SHOW_PLOTS:
        # Plot the image
        plot_image(image_array, info=info)

    return image_array


def get_fingerprint_encoded_from_array(arr):
    fp_int = []
    arr = arr / 255

    # Step 1: Iterate over each row in the array to extract sign and binary
    for i in range(arr.shape[0]):
        # First bit is the sign (0 for negative, 1 for positive)
        sign = -1 if arr[i, 0] == 0 else 1

        # Remaining 31 bits represent the unsigned integer
        binary_str = ''.join(str(int(x)) for x in arr[i, 1:])  # Convert float to int, then to str
        if not all(c in '01' for c in binary_str):
            raise ValueError(f"Invalid binary string detected: {binary_str}")

        # Convert the binary string to an integer
        integer_value = int(binary_str, 2) * sign

        # Append the integer to the list
        fp_int.append(integer_value)

    # Step 2: Encode the list of integers back into a fingerprint format
    fingerprint_encoded = chromaprint.encode_fingerprint(fp_int, algorithm=ALGORITHM_VERSION)

    return fingerprint_encoded


def get_fingerprint_encoded_from_filename(filename):
    with open(filename, 'rb') as binary_file:
        # Read the content of the file
        binary_content = binary_file.read()
    return binary_content