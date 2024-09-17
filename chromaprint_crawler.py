from config import *
import requests
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from PIL import Image
import time
import io
import base64
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
from chromaprint_utils import get_fingerprint_encoded_from_array, get_array_from_image
import pandas as pd
import pandas.errors
import random

random.seed(RANDOM_SEED)

def get_singles_by_artist_id(artist_id, max_singles=MAX_SINGLES):

    # Base URL for querying release groups (singles)
    base_url = f"https://musicbrainz.org/ws/2/release-group?artist={artist_id}&type=single&fmt=json&limit=100"

    # List to store the singles with their MBIDs and titles
    singles = []
    offset = 0

    # Step 2: Paginate through the release groups to get all singles
    while True:
        # Query the MusicBrainz API for singles with pagination
        response = requests.get(f"{base_url}&offset={offset}")
        data = response.json()
        if 'release-groups' in data.keys():
    
        # Add titles and MBIDs to the list
            for release_group in data['release-groups']:
                title = release_group['title']
                mbid = release_group['id']
                singles.append({"title": title, "mbid": mbid})
                #print(f"Single: {title}")
                if len(singles) >= max_singles:
                    print(f"We reached max number of singles: {max_singles}")
                    break
    
        # Break the loop if there are no more results
        if len(data['release-groups']) < 100:
            break

        # Increment the offset for the next request
        offset += 100

    return singles


def get_release_id_by_single(release_group_id, title=None):
    # Query the releases under this release group (single)
    release_url = f"https://musicbrainz.org/ws/2/release?release-group={release_group_id}&fmt=json"
    release_response = requests.get(release_url)
    release_data = release_response.json()
    release_id = release_data['releases'][0]['id']
    return release_id

def get_recordings_by_release_id(release_id, max_recordings=MAX_RECORDINGS_PER_RELEASE):
    # Query recordings for this release
    recording_ids = []
    release_url = f"https://musicbrainz.org/ws/2/release/{release_id}?inc=recordings&fmt=json"
    release_response = requests.get(release_url)
    release_data = release_response.json()

    for medium in release_data['media']:
        for track in medium['tracks']:
            recording_ids.append(track['recording']['id'])
            if len(recording_ids) >= max_recordings:
                break
    return recording_ids

def get_recordings_by_single(release_group_id, title=None, max_recordings=2):
    print(f"Single title: {title}")
    release_id = get_release_id_by_single(release_group_id)
    list_mbids = get_recordings_by_release_id(release_id, max_recordings=max_recordings)
    return list_mbids



def get_image_by_fingerprint_id(fingerprint_id, driver, debug=IS_DEBUG):
    # Open the AcoustID fingerprint page
    fingerprint_url = f"https://acoustid.org/fingerprint/{fingerprint_id}"
    driver.get(fingerprint_url)

    # Wait for the page and canvas to load
    time.sleep(SECONDS_SLEEP_ACOUSTID)

    # Execute JavaScript to extract the canvas image as a data URL
    canvas_data_url = driver.execute_script("""
        var canvas = document.getElementById('fp-img');
        return canvas.toDataURL('image/png');
    """)

    # Remove the base64 prefix from the Data URL
    image_data = canvas_data_url.split(",")[1]

    # Decode the base64 data into bytes and open it as an image using Pillow
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    if debug:
    # Display the image using matplotlib
        plt.imshow(image)
        plt.axis('off')  # Turn off axis
        plt.show()
    return image


def store_fingerprint_encoded(fingeprint_bytearray, filename):
    # Open a file in write mode ('w')
    with open(filename, "wb") as file:
        # Write the string to the file
        file.write(fingeprint_bytearray)


def get_acoustid_track_id_by_mbid(mbid):
    track_id = None
    base_url = "https://api.acoustid.org/v2/track/list_by_mbid"
    response = requests.get(f"{base_url}?mbid={mbid}")

    data = response.json()

    list_tracks = data['tracks']

    if len(list_tracks)>0:
        track_id = list_tracks[0]['id']
    return track_id


def get_fingerprint_and_metadata_by_track_id(track_id, html_content=None):
    first_fingerprint_id = None
    first_metadata = None
    
    if html_content is None:
        url = f"https://acoustid.org/track/{track_id}"  # Replace with actual URL
        # Send a GET request to the webpage
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            html_content = response.content
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
            return None, None

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the Fingerprints table and get all rows (skip header row)
    fingerprint_table = soup.find('h3', text='Fingerprints').find_next('table')
    fingerprint_rows = fingerprint_table.find_all('tr')[1:]  # Skip header row

    # Get the first non-empty fingerprint

    for row in fingerprint_rows:
        columns = row.find_all('td')
        if len(columns) >= 3:
            first_fingerprint_id = columns[0].text.strip()
            break  # Exit after finding the first relevant fingerprint
    if first_fingerprint_id is None:
        return None, None

    # Find the Linked MusicBrainz recordings table
    musicbrainz_table = soup.find('h3', text='Linked MusicBrainz recordings').find_next('table')

    # Get all the rows in the Linked MusicBrainz recordings table (skip header row)
    musicbrainz_rows = musicbrainz_table.find_all('tr')[1:]

    # Extract the first valid title, artist, and length
    for row in musicbrainz_rows:
        columns = row.find_all('td')
        if len(columns) >= 3:  # Ensure the row contains the required columns
            title = columns[0].text.strip()
            artist = columns[1].text.strip()
            length = columns[2].text.strip()
            first_metadata = {
                "artist": [artist],
                "title": [title],
                "length": [length]
            }
            if artist.isdigit():
                continue
            else:
                break  # Exit after finding the first valid metadata

    return first_fingerprint_id, first_metadata


def store_metadata(metadata, metadata_filename):
    print(metadata)
    metadata = pd.DataFrame(metadata)
    metadata.to_csv(metadata_filename, sep='\t', header=None, index=False)

def flatten_dict_values(input_dict):
    flat_list = []
    for key, value_list in input_dict.items():
        if isinstance(value_list, list):  # Check if the value is a list
            flat_list.extend(value_list)  # Extend the flat list with elements from the value list
    return flat_list


def main_crawler():
    dict_artists = {}
    try:
        df_mbids = pd.read_csv(MBIDS_HISTORY_FILENAME)
    except (FileNotFoundError, pandas.errors.EmptyDataError):
        df_mbids = None

    for artist_id in LIST_ARTIST_ID:
        if df_mbids is not None and artist_id in df_mbids['artist'].unique():
            list_mbids = df_mbids[df_mbids['artist'] == artist_id].mbid.values.tolist()
        else:
            list_mbids = []
            print(f"Singles and Recording IDs for artist_id: {artist_id}")
            singles = get_singles_by_artist_id(artist_id)

            for single in singles:
                time.sleep(SECONDS_SLEEP_MUSICBRAINZ)
                list_mbids_for_single = get_recordings_by_single(single['mbid'], title=single['title'])
                if list_mbids_for_single is not None:
                    list_mbids.extend(list_mbids_for_single)

        dict_artists[artist_id] = list_mbids

    dict_artists_str = {key: ','.join(value) for key, value in dict_artists.items()}

    processed_data = [(key, item) for key, value in dict_artists_str.items() for item in value.split(',')]

    # Create a DataFrame from the processed data
    df_mbids = pd.DataFrame(processed_data, columns=['artist', 'mbid'])

    df_mbids.to_csv(MBIDS_HISTORY_FILENAME, index=False)
    # Setup: Use WebDriverManager to automatically download and set up ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    for artist_id in LIST_ARTIST_ID:
        print(f"artist_id: {artist_id}")
        list_mbids = dict_artists[artist_id]
        for mbid in list_mbids:
            track_id = get_acoustid_track_id_by_mbid(mbid)
            print(f"acoustid track_id: {track_id}")
            if track_id is not None:
                fingerprint_id, metadata = get_fingerprint_and_metadata_by_track_id(track_id)
                filename = f"data/{artist_id}/fingerprint_{fingerprint_id}.txt"
                if not os.path.exists(filename):
                    image = get_image_by_fingerprint_id(fingerprint_id, driver)
                    array = get_array_from_image(image, debug=IS_DEBUG, info=f"{artist_id}/{filename}")
                    fingerprint = get_fingerprint_encoded_from_array(array)

                    os.system(f"mkdir -p data/{artist_id}")
                    store_fingerprint_encoded(fingerprint, filename)
                    metadata_filename = f"data/{artist_id}/metadata_{fingerprint_id}.txt"
                    store_metadata(metadata, metadata_filename)

    # Finally: Close the browser
    driver.quit()

if __name__ == "__main__":

   main_crawler()




