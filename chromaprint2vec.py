from config import *
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import subprocess
import random
from chromaprint_utils import get_array_from_fingerprint_encoded, get_fingerprint_encoded_from_filename
random.seed(RANDOM_SEED)


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
