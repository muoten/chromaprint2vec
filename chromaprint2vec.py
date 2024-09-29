from chromagram_utils import get_chromagram_from_chromaprint, get_array_from_chromagram
from config import *
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import subprocess
import random
from chromaprint_utils import get_array_from_fingerprint_encoded, get_fingerprint_encoded_from_filename, \
    refine_vectors_with_best_offsets, get_fingerprint_encoded_from_array

random.seed(RANDOM_SEED)


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

        offsets, vectors_refined, adhoc_mapping = refine_vectors_with_best_offsets(vectors_truncated)
        vectors = [np.concatenate((vector[int(offsets[i]):], vector[:int(offsets[i])])) for i, vector in enumerate(vectors)]
    return vectors, adhoc_mapping




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
    sorted_artist_list = sorted(LIST_ARTIST_ID)
    vectors_original, adhoc_mapping = generate_vectors_from_artist_list(sorted_artist_list)
    vectors_chromagram = []
    for i, vector in enumerate(vectors_original):
        array = vector.reshape(-1,32)
        my_fingerprint = get_fingerprint_encoded_from_array(array)
        chromagram = get_chromagram_from_chromaprint(my_fingerprint)
        array_chromagram = get_array_from_chromagram(chromagram)
        vectors_chromagram.append(array_chromagram.reshape(-1))
    vectors_reduced = reduce_dimensions(vectors_chromagram)
    df_vectors_reduced = pd.DataFrame(vectors_reduced)
    df_vectors_reduced.to_csv(VECTORS_FILENAME, sep='\t', header=False, index=False)
    df_metadata = collect_metadata()
    df_metadata = reformat_metadata(df_metadata)
    df_metadata.to_csv(METADATA_FILENAME, sep='\t', index=False)
