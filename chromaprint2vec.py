from chromaprint_crawl_anomalies import get_distance_to_ref
from config import *
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import subprocess
import random
from chromaprint_utils import get_array_from_fingerprint_encoded, get_fingerprint_encoded_from_filename
from numpy.fft import fft, ifft
import time
random.seed(RANDOM_SEED)

adhoc_mapping = {}

def generate_vectors_from_artist_list(list_artists):
    vectors = []
    for artist_id in list_artists:
        fingerprint_filenames = sorted([f for f in os.listdir(f"data/{artist_id}") if f.endswith('.txt') and f.startswith('fingerprint')])

        for filename in fingerprint_filenames:
            fingerprint_encoded = get_fingerprint_encoded_from_filename(f"data/{artist_id}/{filename}")

            array = get_array_from_fingerprint_encoded(fingerprint_encoded, debug=IS_DEBUG, info=f"{artist_id}/{filename}")
            vector_i = array.reshape(-1)
            if IS_DEBUG:
                print(fingerprint_encoded)
                print(array)

            vectors.append(vector_i)
    max_length = max(len(vector_i) for vector_i in vectors)  # Find the maximum length of vectors

    # Repeat the content of each vector until it reaches the max_length
    vectors = [np.tile(vector_i, int(np.ceil(max_length / len(vector_i))))[:max_length] for vector_i in vectors]

    if IS_DEBUG:
        print(vectors)
    if FIND_BEST_OFFSET:
        # truncated vectors to accelerate refine_vectors_with_best_offsets
        vectors_truncated = [vector[:MINIMAL_LENGTH_VECTORS] for vector in vectors]

        offsets, vectors_refined = refine_vectors_with_best_offsets(vectors_truncated)
        vectors = [np.concatenate((vector[int(offsets[i]):], vector[:int(offsets[i])])) for i, vector in enumerate(vectors)]
    return vectors


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


def refine_vectors_with_best_offsets(vectors, threshold=0.10):
    vectors_refined = []
    offsets = np.zeros(len(vectors))
    n_iterations = len(vectors) * len(vectors)
    count = 0
    min_distance = 1
    start_time = time.time()
    for i,arr_i in enumerate(vectors):
        for j,arr_j in enumerate(vectors):
            count = count+1
            if IS_DEBUG and count%1000 == 0:
                print(f"Iteration: {count} of {n_iterations},  i={i}, j={j}")
                end_time = time.time()
                execution_time = end_time - start_time  # Calculate the execution time
                print(f"Execution time per iteration: {execution_time/1000} seconds")

            if i != j:
                # Find the best offset using FFT
                best_offset, max_corr = find_best_offset_fft(arr_i, arr_j)
                if best_offset > 0:
                    arr_i_offset = np.concatenate((arr_i[best_offset:], arr_i[:best_offset]))
                    best_distance = get_distance_to_ref(arr_i_offset, vector_ref=arr_j)
                    if best_distance < min_distance:
                        min_distance = best_distance
                    if best_distance < threshold:
                        print(f"For i={i}, j={j}, offset={best_offset}, distance={best_distance}")
                        adhoc_mapping[i] = j
                        vectors_refined.append(arr_i_offset)
                    else:
                        vectors_refined.append(arr_i)
        if i in adhoc_mapping.keys():
            offsets[i] = adhoc_mapping[i]
        else:
            offsets[i] = 0
    end_time = time.time()
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Execution time: {execution_time} seconds")

    return offsets, vectors_refined


def reduce_dimensions(vectors):
    # Perform PCA to reduce dimensions
    pca = PCA(random_state=RANDOM_SEED, n_components=OUTPUT_DIMENSIONS)

    vectors_in = np.vstack(vectors)

    n_input_dimensions = None
    if IS_DEBUG:
        print(f"Original dimensions={vectors_in.shape}")

    vectors_out = pca.fit_transform(vectors_in[:, :n_input_dimensions])

    return vectors_out


def reformat_metadata(df):
    df = pd.DataFrame(df, columns=['artist', 'title', 'length'])
    df = df[['title', 'artist', 'length']]
    df['index'] = df.index
    df['__next__'] = df.index
    #adhoc_mapping = {11: 1, 46: 76, 36: 105, 42: 43}
    df['__next__'] = df['__next__'].apply(lambda x: adhoc_mapping[x] if x in adhoc_mapping.keys() else '')
    return df


def collect_metadata():
    metadata = None
    try:
        # Use shell=True to enable wildcard expansion
        metadata = subprocess.check_output('cat data/*/metadata_*.txt', shell=True, text=True)
        print(metadata)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")

    # Split the input string into lines
    lines = metadata.splitlines()

    # Split each line into artist and title
    df = [line.split("\t") for line in lines]
    return df


if __name__ == "__main__":
    vectors_original = generate_vectors_from_artist_list(LIST_ARTIST_ID)
    vectors_reduced = reduce_dimensions(vectors_original)
    df_vectors_reduced = pd.DataFrame(vectors_reduced)
    df_vectors_reduced.to_csv(VECTORS_FILENAME, sep='\t', header=False, index=False)
    df_metadata = collect_metadata()
    df_metadata = reformat_metadata(df_metadata)
    df_metadata.to_csv(METADATA_FILENAME, sep='\t', index=False)
