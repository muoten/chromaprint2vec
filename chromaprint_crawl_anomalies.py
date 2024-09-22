from config import *
import requests
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from chromaprint_utils import get_array_from_image, get_distance_to_ref, refine_vectors_with_best_offsets, \
    generate_distance_matrix
from chromaprint_crawler import get_image_by_fingerprint_id, get_fingerprint_and_metadata_by_track_id
import random
import numpy as np

random.seed(RANDOM_SEED)

#LIST_OF_RECORDINGS_TO_REVIEW = ['ca761827-c9d4-4a44-9dea-0eec9df14ed4']
#LIST_OF_RECORDINGS_TO_REVIEW = ['f28d7df9-56bb-4045-9c8c-f341dba9dca3']
#LIST_OF_RECORDINGS_TO_REVIEW = ['51884593-f3e3-4b72-bfbe-a201953fefa0']
LIST_OF_RECORDINGS_TO_REVIEW = ['b301b40b-534c-4ce3-8f0f-81d674ba5444']
LIST_OF_RECORDINGS_TO_REVIEW = ['ca761827-c9d4-4a44-9dea-0eec9df14ed4']
LIST_OF_RECORDINGS_TO_REVIEW = ['980a426e-623e-4ea5-98c7-008d037a0508'] # No surprises, Radiohead
LIST_OF_RECORDINGS_TO_REVIEW = ['9e2ad5bc-c6f9-40d2-a36f-3122ee2072a3'] # Karma Police, Radiohead
LIST_OF_RECORDINGS_TO_REVIEW = ['f3bea96c-5ebf-4d37-990d-14f1bc281459'] # Nude, Radiohead


def get_acoustid_track_id_list_by_mbid(mbid):
    base_url = "https://api.acoustid.org/v2/track/list_by_mbid"
    response = requests.get(f"{base_url}?mbid={mbid}")

    data = response.json()

    list_tracks = data['tracks']

    track_id_list = []
    for track in list_tracks:
        track_id = track['id']
        track_id_list.append(track_id)
    return track_id_list


def get_anomalies_by_distance_to_centroid(array_all_fingerprints, fingerprint_list_id, threshold=0.2):
    centroid = np.mean(array_all_fingerprints, axis=0)
    anomalies = []
    for i, arr_i in enumerate(array_all_fingerprints):
        distance = get_distance_to_ref(arr_i, vector_ref=centroid)
        if distance > threshold:
            anomalies.append(fingerprint_list_id[i])
    return anomalies


def get_anomalies_by_average_distance_to_others(array_all_fingerprints, fingerprint_list_id, threshold=0.3):
    distance_matrix = generate_distance_matrix(array_all_fingerprints)
    print(distance_matrix)
    mean_distance_matrix = np.mean(distance_matrix, axis=0)
    anomalies = []
    for i in range(0,len(mean_distance_matrix)):
        distance = mean_distance_matrix[i]
        if distance > threshold:
            anomalies.append(fingerprint_list_id[i])
    return anomalies

def main_crawler():

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    for mbid in LIST_OF_RECORDINGS_TO_REVIEW:
        track_id_list = get_acoustid_track_id_list_by_mbid(mbid)
        print(track_id_list)
        array_all_fingerprints = []
        fingerprint_id_list = []
        for track_id in track_id_list:
            print(f"acoustid track_id: {track_id}")
            fingerprint_id, metadata = get_fingerprint_and_metadata_by_track_id(track_id)
            print(f"fingerprint_id: {fingerprint_id}")

            # let's skip all arrays with length < MINIMAL_LENGTH_VECTORS


            image = get_image_by_fingerprint_id(fingerprint_id, driver)
            array = get_array_from_image(image, debug=False)
            if len(array.reshape(-1)) > MINIMAL_LENGTH_VECTORS:
                fingerprint_id_list.append(fingerprint_id)
                array_all_fingerprints.append(array.reshape(-1))

        if FIND_BEST_OFFSET:
            # truncated vectors to accelerate refine_vectors_with_best_offsets
            vectors_truncated = [vector[:MINIMAL_LENGTH_VECTORS] for vector in array_all_fingerprints]
            offsets, vectors_refined, adhoc_mapping = refine_vectors_with_best_offsets(vectors_truncated)
            array_all_fingerprints = vectors_refined
        list_distances = [get_distance_to_ref(array_all_fingerprints[i], vector_ref=array_all_fingerprints[value]) for i,value in adhoc_mapping.items()]

        anomalies_method1 = get_anomalies_by_distance_to_centroid(array_all_fingerprints, fingerprint_id_list)
        anomalies_method2 = get_anomalies_by_average_distance_to_others(array_all_fingerprints, fingerprint_id_list)

        print(
            f"This list of Acoustid fingerprints {fingerprint_id_list} ",
            f"is associated to the same MusicBrainz recording_id: {mbid}"
        )
        print("Linked to the following Acoustid tracks:")
        for i, fingerprint_id in enumerate(fingerprint_id_list):
            print(f"{track_id_list[i]} -> {fingerprint_id}")

        print("And we detect this subset as outliers:")
        print(f"Method 1: {anomalies_method1}")
        print(f"Method 2: {anomalies_method2}")

    # Finally: Close the browser
    driver.quit()


if __name__ == "__main__":

   main_crawler()




