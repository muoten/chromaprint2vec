import chromaprint
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import time

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


def get_distance_to_ref(vector_i, vector_ref=None):
    cosine_distance=None
    if vector_ref is not None:
        # Normalize vectors
        vector_i = vector_i / np.linalg.norm(vector_i)
        vector_ref = vector_ref / np.linalg.norm(vector_ref)

        # Compute cosine similarity
        cosine_similarity = np.dot(vector_ref, vector_i)

        # Compute cosine distance
        cosine_distance = 1 - cosine_similarity

    return cosine_distance


# Function to compute the best offset using FFT-based cross-correlation
def find_best_offset_fft(arr1, arr2):
    # Step 1: Compute the FFT of both arrays
    f_arr1 = fft(arr1)
    f_arr2 = fft(arr2)

    # Step 2: Compute the cross-correlation using inverse FFT of the product of one FFT
    # and the complex conjugate of the other FFT
    cross_correlation = ifft(f_arr1 * np.conj(f_arr2)).real

    # Step 3: Find the index of the maximum value in the cross-correlation
    best_offset = np.argmax(cross_correlation)

    # Step 4: Handle the wrap-around offset (negative shifts)
    if best_offset > len(arr1) // 2:
        best_offset -= len(arr1)

    # Return the best offset and the maximum correlation value
    return best_offset, cross_correlation[best_offset]

def generate_distance_matrix(array_all_fingerprints):
    num_vectors = len(array_all_fingerprints)

    # Initialize a distance matrix
    distance_matrix = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(i, num_vectors):  # Start from i to avoid redundant calculations
            if i == j:
                distance_matrix[i, j] = 0  # Distance to itself is zero
            else:
                distance = get_distance_to_ref(array_all_fingerprints[i], array_all_fingerprints[j])
                distance = round(distance, 2)
                distance_matrix[i,j] = distance
                distance_matrix[j,i] = distance
    return distance_matrix


def refine_vectors_with_best_offsets(vectors, threshold=0.3):
    adhoc_mapping = {}
    vectors_refined = []
    offsets = np.zeros(len(vectors))
    n_iterations = len(vectors) * len(vectors)
    count = 0

    start_time = time.time()
    for i,arr_i in enumerate(vectors):
        arr_i_best_offset = arr_i
        min_distance_for_arr_i = 1
        best_offset_for_arr_i = 0
        for j,arr_j in enumerate(vectors):
            count = count+1
            if IS_DEBUG and count%1000 == 0:
                print(f"Iteration: {count} of {n_iterations},  i={i}, j={j}")
                end_time = time.time()
                execution_time = end_time - start_time  # Calculate the execution time
                print(f"Execution time per iteration: {execution_time/1000} seconds")

            if j > i:
                # Find the best offset using FFT
                best_offset, max_corr = find_best_offset_fft(arr_i, arr_j)
                best_offset_i, max_corr_i = find_best_offset_fft(1-arr_i, arr_j)
                if max_corr_i > max_corr:
                    best_offset = best_offset_i
                    arr_i = 1-arr_i
                if best_offset != 0 and j not in adhoc_mapping.values():
                    arr_i_offset = np.concatenate((arr_i[best_offset:], arr_i[:best_offset]))
                    best_distance = get_distance_to_ref(arr_i_offset, vector_ref=arr_j)
                    if (best_distance < threshold) & (best_distance < min_distance_for_arr_i):
                        min_distance_for_arr_i = best_distance
                        best_offset_for_arr_i = best_offset
                        print(f"For i={i}, j={j}, offset={best_offset}, distance={min_distance_for_arr_i}")
                        adhoc_mapping[i] = j
                        arr_i_best_offset = arr_i_offset

        vectors_refined.append(arr_i_best_offset)
        if i in adhoc_mapping.keys():
            offsets[i] = best_offset_for_arr_i
        else:
            offsets[i] = 0
    end_time = time.time()
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Execution time: {execution_time} seconds")

    return offsets, vectors_refined, adhoc_mapping