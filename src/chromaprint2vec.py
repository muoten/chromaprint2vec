from chromagram_utils import get_chromagram_from_chromaprint, get_array_from_chromagram
from config import *
import json
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import subprocess
import random
from chromaprint_utils import get_array_from_fingerprint_encoded, get_fingerprint_encoded_from_filename, \
    refine_vectors_with_best_offsets, get_fingerprint_encoded_from_array, get_distance_to_ref

random.seed(RANDOM_SEED)


def generate_vectors_from_artist_list(list_artists, use_chromagrams=USE_CHROMAGRAMS):
    vectors = []
    for artist_id in list_artists:
        fingerprint_filenames = []
        try:
            fingerprint_filenames = sorted([f for f in os.listdir(f"data/{artist_id}") if f.endswith('.txt') and f.startswith('fingerprint')])
        except FileNotFoundError as e:
            print(e)
        assert len(fingerprint_filenames) > 0, f"No data for {artist_id}. Run chromaprint_crawler.py or remove it from LIST_ARTIST_ID"
        for filename in fingerprint_filenames:
            fingerprint_encoded = get_fingerprint_encoded_from_filename(f"data/{artist_id}/{filename}")
            vector_i_fingerprint = np.array([])
            vector_i_chromagram = np.array([])
            if USE_FINGERPRINTS:
                array = get_array_from_fingerprint_encoded(fingerprint_encoded, debug=IS_DEBUG,
                                                       info=f"{artist_id}/{filename}")
                vector_i_fingerprint = array.reshape(-1)
            if use_chromagrams:
                chromagram = get_chromagram_from_chromaprint(fingerprint_encoded)
                array_chromagram = get_array_from_chromagram(chromagram)
                vector_i_chromagram = array_chromagram.reshape(-1)
            vector_i = np.concatenate([vector_i_chromagram, vector_i_fingerprint])
            assert USE_FINGERPRINTS or USE_CHROMAGRAMS, "USE_FINGERPRINTS or USE_CHROMAGRAMS needs to be enabled"
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
        # regenerate vectors after truncation
        vectors = [np.concatenate((vector[int(offsets[i]):], vector[:int(offsets[i])])) for i, vector in enumerate(vectors)]
    return vectors, adhoc_mapping


def regenerate_mapping(adhoc_mapping, vectors):
    # regenerate adhoc_mapping
    adhoc_mapping = {}
    # Compute distances for all pairs
    for i, arr1 in enumerate(vectors):
        for j, arr2 in enumerate(vectors):
            if i < j:  # Avoid duplicate pairs and self-pairs
                distance = get_distance_to_ref(arr1, vector_ref=arr2)
                if distance < FIND_MAPPING_THRESHOLD:
                    print(f"For i={i}, j={j}, distance={distance}")
                    adhoc_mapping[i] = j
    return adhoc_mapping



def reduce_dimensions(vectors):
    # Perform PCA to reduce dimensions
    pca = PCA(random_state=RANDOM_SEED, n_components=OUTPUT_DIMENSIONS)

    vectors_in = np.vstack(vectors)

    n_input_dimensions = None
    if IS_DEBUG:
        print(f"Original dimensions={vectors_in.shape}")

    vectors_out = pca.fit_transform(vectors_in[:, :n_input_dimensions])

    return vectors_out


def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # If s1 is empty, insert all characters of s2
            elif j == 0:
                dp[i][j] = i  # If s2 is empty, remove all characters of s1
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # If last characters match, ignore them
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # Remove
                                   dp[i][j - 1],    # Insert
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][n]


def refine_mapping(df, adhoc_mapping):
    keys = adhoc_mapping.copy().keys()
    for key in keys:
        title_src = df[df.index==key]['title'].iloc[0]
        title_dst = df[df.index==adhoc_mapping[key]]['title'].iloc[0]
        # Calculate Levenshtein distance
        lev_distance = levenshtein_distance(title_src, title_dst)
        lev_distance_norm = lev_distance/(len(title_src)+len(title_dst))
        print(f"Edit distance '{title_src}' vs '{title_dst}':{lev_distance_norm:.2}")
        self_contained = title_src.startswith(title_dst) or title_dst.startswith(title_src)
        if lev_distance_norm > LEV_DISTANCE_THRESHOLD and not self_contained:
            adhoc_mapping.pop(key, None)

    precision = len(adhoc_mapping.keys())/len(keys)
    print(f"Estimated precision based on metadata: {precision:.3}")


def reformat_metadata(df, adhoc_mapping):
    df = pd.DataFrame(df, columns=['artist', 'title', 'length'])
    df = df[['title', 'artist', 'length']]
    df['index'] = df.index
    df['__next__'] = df.index
    refine_mapping(df, adhoc_mapping)
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


def convert_input_to_vectors_csv(function, argument):
    if not os.path.exists(VECTORS_ORIGINAL_FILENAME) or not RELOAD_VECTORS:
        vectors_original, adhoc_mapping = function(argument)
        np.save(VECTORS_ORIGINAL_FILENAME, vectors_original)
        with open(MAPPING_FILENAME, "w") as f:
            json.dump(adhoc_mapping, f)
    else:
        print(f"Loading {VECTORS_ORIGINAL_FILENAME}...")
        vectors_original = np.load(VECTORS_ORIGINAL_FILENAME)
        with open(MAPPING_FILENAME, "r") as f:
            adhoc_mapping = json.load(f)
        # Convert string keys to integers
        adhoc_mapping = {int(k): v for k, v in adhoc_mapping.items()}

    vectors_reduced = reduce_dimensions(vectors_original)
    if REDO_MAPPING_AFTER_PCA:
        adhoc_mapping = regenerate_mapping(adhoc_mapping, vectors_reduced)
    df_vectors_reduced = pd.DataFrame(vectors_reduced)
    df_vectors_reduced.to_csv(VECTORS_FILENAME, sep='\t', header=False, index=False)
    df_metadata = collect_metadata()
    df_metadata = reformat_metadata(df_metadata, adhoc_mapping)
    df_metadata.to_csv(METADATA_FILENAME, sep='\t', index=False)


if __name__ == "__main__":
    sorted_artist_list = sorted(LIST_ARTIST_ID)
    convert_input_to_vectors_csv(generate_vectors_from_artist_list, sorted_artist_list)
