import chromaprint

from config import *
import eyed3
import numpy as np
import os
import random
import time
from chromaprint_utils import refine_vectors_with_best_offsets, get_array_from_fingerprint_encoded
from src.chromaprint2vec import convert_input_to_vectors_csv
from src.harmony_utils import audio_file_to_chords, chords_to_vector
from src.pychromagram import load_audio_file, get_chromagram
if USE_FINGERPRINTS:
    from src.pychromaprint import FingerprinterConfigurationTest2, FingerprintCalculator
    # Create the configuration
    config = FingerprinterConfigurationTest2()
    # Create the FingerprintCalculator using the configuration
    calculator = FingerprintCalculator(config)

random.seed(RANDOM_SEED)


def audiofile_to_chromagram(audiofile):
    data = load_audio_file(audiofile, convert_to_mono_11025hz=True)
    chromagram = get_chromagram(data)
    return chromagram


def get_sorted_audio_filenames(folder):
    sorted_audio_filenames = sorted([f for f in os.listdir(folder)])
    return sorted_audio_filenames


def chromagram_to_chromaprint(chromagram):
    calculator.consume(chromagram.data)
    fingerprint = calculator.get_fingerprint()

    decoded_fp_actual = fingerprint
    encoded_fp_actual = chromaprint.encode_fingerprint(decoded_fp_actual, algorithm=2)
    return encoded_fp_actual


def generate_vectors_from_audio_files(folder):
    vectors = []
    sorted_audio_filenames = get_sorted_audio_filenames(folder)
    start_time = time.time()
    for i,audio_filename in enumerate(sorted_audio_filenames):
        filename_path = f"{folder}/{audio_filename}"

        vector_i_fingerprint = np.array([])
        vector_i_chromagram = np.array([])
        vector_i_harmony = np.array([])

        if USE_CHROMAGRAMS or USE_FINGERPRINTS:
            chromagram = audiofile_to_chromagram(filename_path)
            if USE_CHROMAGRAMS:
                vector_i_chromagram = np.array(chromagram.data).reshape(-1)

        if USE_FINGERPRINTS:
            fingerprint_encoded = chromagram_to_chromaprint(chromagram)
            print(fingerprint_encoded)
            array = get_array_from_fingerprint_encoded(fingerprint_encoded, debug=IS_DEBUG)
            vector_i_fingerprint = array.reshape(-1)

        if USE_HARMONY:
            chords = audio_file_to_chords(filename_path)
            array_chords = chords_to_vector(chords)
            vector_i_harmony = np.array(array_chords).reshape(-1)
        vector_i = np.concatenate([vector_i_chromagram, vector_i_fingerprint, vector_i_harmony])
        vectors.append(vector_i)
        if i % 10 == 0:
            end_time = time.time()
            execution_time = end_time - start_time  # Calculate the execution time
            elapsed_time_msg = (
                f"{i + 1}/{len(sorted_audio_filenames)} files processed. "
                f"Elapsed time: {execution_time:.1f}s ({(execution_time) / (i+1):.1f}s per file)"
            )
            print(elapsed_time_msg)

    max_length = max(len(vector_i) for vector_i in vectors)  # Find the maximum length of vectors

    assert len(vector_i) > 0, f"{audio_filename} gets empty chromagram"
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


def collect_metadata_from_audio_files():
    folder = os.path.expanduser(FOLDER_AUDIO_FILES)
    sorted_audio_filenames = get_sorted_audio_filenames(folder)
    lines = []
    for audio_filename in sorted_audio_filenames:
        audio = eyed3.load(f"{folder}/{audio_filename}")

        # Extract metadata
        print(f"Title: {audio.tag.title}")
        assert audio.tag.title is not None, f"No title metadata in {audio_filename}"
        print(f"Artist: {audio.tag.artist}")
        print(f"Album: {audio.tag.album}")
        print(f"Genre: {audio.tag.genre}")
        print(f"Duration: {audio.info.time_secs:.2f} seconds")
        line = f"{audio.tag.artist}\t{audio.tag.title}\t{audio.info.time_secs}"
        lines.append(line)

    # Split each line into artist and title
    df = [line.split("\t") for line in lines]
    return df


if __name__ == "__main__":
    myfolder = os.path.expanduser(FOLDER_AUDIO_FILES)
    function_to_collect_metadata = collect_metadata_from_audio_files
    convert_input_to_vectors_csv(generate_vectors_from_audio_files, myfolder, function_to_collect_metadata)

