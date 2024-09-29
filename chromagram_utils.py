import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import chromaprint
from chromaprint_utils import get_fingerprint_encoded_from_filename, get_array_from_fingerprint_encoded, \
    get_fingerprint_encoded_from_array


# Function to convert a 32-bit integer into a chroma vector (12 bins)
def int_to_binary_chroma_vector(fingerprint_value):
    # Extract bits from the 32-bit integer
    bit_string = format(fingerprint_value, '032b')  # Get the 32-bit binary string

    # We will re-interpret the bit patterns rather than assume direct chroma mapping
    # Try extracting in smaller chunks for finer control
    # Try to understand which parts of the bit string correspond to meaningful chroma features
    chroma_vector = []
    for i in range(12):
        # Instead of using all bits as they are, focus on bits that show the most relevant intensity
        bin_value = int(bit_string[i * 2:(i * 2) + 2], 2)  # Extracting two bits per bin
        chroma_vector.append(bin_value)
        # chroma_vector.append(int(bit_string[i]))  # This is just to analyze the bit mapping

    return chroma_vector


def get_chromagram_from_chromaprint(fingerprint):
    chromaprint_fingerprint, version = chromaprint.decode_fingerprint(fingerprint, base64=True)
    return chromaprint_fingerprint


def plot_chromagram(chromaprint_fingerprint):
    # Convert the Chromaprint fingerprint into a coarse chromagram (12 chroma bins per time window)
    chromagram = [int_to_binary_chroma_vector(fp_value) for fp_value in chromaprint_fingerprint]

    # Convert the list of chroma vectors into a NumPy array for easier manipulation and visualization
    chromagram = np.array(chromagram).T  # Transpose to match time vs chroma

    # Plot the coarse chromagram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.colorbar()
    plt.title('Coarse Chromagram from Chromaprint Fingerprint (Array of Integers)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    filename = "data/fingerprint_chromatic.txt"
    my_fingerprint1 = get_fingerprint_encoded_from_filename(filename)
    vector = get_array_from_fingerprint_encoded(my_fingerprint1)
    my_fingerprint2 = get_fingerprint_encoded_from_array(vector)
    chromagram = get_chromagram_from_chromaprint(my_fingerprint2)
    plot_chromagram(chromagram)
