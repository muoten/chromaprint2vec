from config import *
import chromaprint
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA
import subprocess
import random

random.seed(RANDOM_SEED)

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

    if debug:
        plt.figure(figsize = (20,10))
        plt.imshow(arr.T, aspect='auto', origin='lower')
        title = 'Binary representation of a Chromaprint'
        if info is not None:
            title = f"{title} for {info}"
        plt.title(title)
        plt.show()
        plt.close()
    return arr


def get_fingerprint_encoded_from_filename(filename):
    with open(filename, 'rb') as binary_file:
        # Read the content of the file
        binary_content = binary_file.read()
    return binary_content


def generate_vectors_from_artist_list(list_artists):
    vectors = []
    min_length = TYPICAL_LENGTH_VECTORS
    for artist_id in list_artists:
        fingerprint_filenames = sorted([f for f in os.listdir(f"data/{artist_id}") if f.endswith('.txt') and f.startswith('fingerprint')])

        for filename in fingerprint_filenames:
            fingerprint_encoded = get_fingerprint_encoded_from_filename(f"data/{artist_id}/{filename}")

            array = get_array_from_fingerprint_encoded(fingerprint_encoded, debug=IS_DEBUG, info=f"{artist_id}/{filename}")
            vector_i = array.reshape(-1)
            if IS_DEBUG:
                print(fingerprint_encoded)
                print(array)

            if len(vector_i) < min_length:
                min_length = len(vector_i)

            vectors.append(vector_i)
    vectors = [vector_i[:min_length] for vector_i in vectors]
    if IS_DEBUG:
        print(vectors)
    return vectors


def reduce_dimensions(vectors):
    # Perform PCA to reduce dimensions
    pca = PCA(random_state=RANDOM_SEED, n_components=OUTPUT_DIMENSIONS)

    vectors_in = np.vstack(vectors)

    n_input_dimensions = None
    if IS_DEBUG:
        print(f"Original dimensions={vectors_in.shape}")

    vectors_out = pca.fit_transform(vectors_in[:, :n_input_dimensions])

    return vectors_out



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
    data = [line.split("\t") for line in lines]

    df = pd.DataFrame(data, columns=['artist', 'title','length'])
    return df


if __name__ == "__main__":
    vectors_original = generate_vectors_from_artist_list(LIST_ARTIST_ID)
    vectors_reduced = reduce_dimensions(vectors_original)
    df_vectors_reduced = pd.DataFrame(vectors_reduced)
    df_vectors_reduced.to_csv(VECTORS_FILENAME, sep='\t', header=False, index=False)
    df_metadata = collect_metadata()
    df_metadata.to_csv(METADATA_FILENAME, sep='\t', index=False)
