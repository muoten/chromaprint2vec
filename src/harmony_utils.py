import essentia.standard as es
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

FRAME_SIZE = 2048
HOP_SIZE = 512*2
VOLUME_THRESHOLD = 0.01
REMOVE_REPEATED_CHORDS = True
MIN_CONSECUTIVE_CHORDS_THRESHOLD = 16
STRENGTH_THRESHOLD = 0.1


def filename_to_audio(filename):
    loader = es.MonoLoader(filename=filename)
    audio = loader()
    sample_rate = loader.paramValue("sampleRate")
    print(f"Sample rate: {sample_rate}")
    return audio, sample_rate


def remove_silence(audio, sample_rate):

    # Compute loudness
    rms = es.RMS()  # Root Mean Square energy calculator
    frame_cutter = es.FrameCutter(frameSize=FRAME_SIZE, hopSize=HOP_SIZE)

    # Extract frames and calculate RMS
    frames = []
    loudness = []
    while True:
        frame = frame_cutter(audio)
        if len(frame) == 0:
            break
        loudness.append(rms(frame))  # Calculate RMS energy
        frames.append(frame)

    # Apply volume threshold
    frames = np.array(frames)
    loudness = np.array(loudness)
    frames_filtered = frames[loudness > VOLUME_THRESHOLD]

    print(f"Number of frames before filtering: {len(frames)}")
    print(f"Number of frames after filtering: {len(frames_filtered)}")

    # Reconstruct audio after filtering
    if len(frames_filtered) > 0:
        audio_filtered = np.zeros(len(audio), dtype=np.float32)  # Initialize an empty array for the reconstructed signal
        for i, frame in enumerate(frames_filtered):
            start = i * HOP_SIZE
            end = start + FRAME_SIZE
            if end <= len(audio_filtered):
                audio_filtered[start:end] += frame  # Overlap-add reconstruction

        # Normalize the reconstructed signal
        audio_filtered = audio_filtered / np.max(np.abs(audio_filtered))

    es.MonoWriter(filename="filtered_audio2.wav", sampleRate=sample_rate)(audio_filtered)
    return audio_filtered


def get_hpcp_features(audio, sample_rate):
    # Load audio

    # Frame-based analysis
    frame_cutter = es.FrameCutter(frameSize=FRAME_SIZE, hopSize=HOP_SIZE)
    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(minFrequency=20, maxFrequency=20000, magnitudeThreshold=0.0001)
    hpcp = es.HPCP(sampleRate=sample_rate, referenceFrequency=440.0, harmonics=1, bandPreset=True)

    # Compute HPCP over time
    hpcp_frames = []
    while True:
        frame = frame_cutter(audio)
        if len(frame) == 0:
            break
        windowed_frame = window(frame)
        spectrum_frame = spectrum(windowed_frame)
        frequencies, magnitudes = spectral_peaks(spectrum_frame)
        hpcp_features = hpcp(frequencies, magnitudes)
        hpcp_frames.append(hpcp_features)

    # Convert HPCP frames to a 2D array
    hpcp_frames = np.array(hpcp_frames)
    return hpcp_frames


def plot_hpcp_features(hpcp_frames):
    # Visualize HPCP features over time
    plt.imshow(hpcp_frames.T, aspect='auto', origin='lower', interpolation='none')
    plt.title('HPCP Features Over Time')
    plt.xlabel('Time Frames')
    plt.ylabel('HPCP Bins')
    plt.colorbar()
    plt.show()


def extract_chords(hpcp_frames):
    # Perform Chord Detection
    chords_extractor = es.ChordsDetection()
    chords, strengths = chords_extractor(hpcp_frames)
    filtered_chords = []
    for chord, group in groupby(chords):
        group_list = list(group)
        if len(group_list) >= MIN_CONSECUTIVE_CHORDS_THRESHOLD:
            filtered_chords.extend(group_list)

    chords = filtered_chords
    if REMOVE_REPEATED_CHORDS:
        chords = [chords[i] for i in range(len(chords)) if i == 0 or chords[i] != chords[i-1]]
    chords = [chords[i] for i in range(len(chords)) if strengths[i] > STRENGTH_THRESHOLD]
    return chords, strengths


# Function to find most repeated pattern of a given length
def most_repeated_pattern(sequence, pattern_length):
    patterns = [
        tuple(sequence[i:i+pattern_length])
        for i in range(len(sequence) - pattern_length + 1)
    ]

    pattern_counts = Counter(patterns)
    # Exclude patterns with fewer than 3 unique chords
    filtered_patterns = {
        pattern: count
        for pattern, count in pattern_counts.items()
        if len(set(pattern)) >= 3
    }
    most_common = Counter(filtered_patterns).most_common(1)
    return most_common[0] if most_common else None


def circle_of_fifths_embedding(chord):
    """
    Compute a 2D embedding for a chord based on its position on the Circle of Fifths.
    """
    circle_of_fifths = {
        "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5,
        "F#": 6, "Gb": 6,  # F# and Gb are the same
        "Db": 7, "C#": 7,  # Db and C# are the same
        "Ab": 8, "G#": 8,  # Ab and G# are the same
        "Eb": 9, "D#": 9,  # Eb and D# are the same
        "Bb": 10, "A#": 10,  # Bb and A# are the same
        "F": 11, "E#": 11  # F and E# are the same
    }
    tonic = chord.rstrip("m")  # Remove minor 'm' suffix
    is_minor = chord.endswith("m")
    angle = 2 * np.pi * circle_of_fifths[tonic] / 12
    radius = 0.9 if is_minor else 1.0  # Minor chords closer to the center
    return np.array([radius * np.cos(angle), radius * np.sin(angle)])


def chord_embedding(chord):
    circle_embedding = circle_of_fifths_embedding(chord)
    quality_vector = {
        "M": [1, 0, 0],   # Major
        "m": [0, 1, 0],   # Minor
        "dim": [0, 0, 1]  # Diminished
    }
    quality = "m" if chord.endswith("m") else "M"
    return np.concatenate([circle_embedding, quality_vector[quality]])


def audio_file_to_chords(audiofile):
    audio, sample_rate = filename_to_audio(audiofile)
    audio = remove_silence(audio, sample_rate)
    hpcp_frames = get_hpcp_features(audio, sample_rate)
    filtered_chords, strengths = extract_chords(hpcp_frames)
    print("Detected Chords:", filtered_chords)
    print("Chords Strengths:", strengths)
    return filtered_chords


def chords_to_vector(chords):
    array = [chord_embedding(chord) for chord in chords]
    return array


if __name__ == "__main__":
    FILENAME='data/videoplayback_10s.wav'

    chords = audio_file_to_chords(FILENAME)
    print(chords)

    # Find the most repeated pattern for lengths 2 to 5
    for length in range(2, 6):
        result = most_repeated_pattern(chords, length)
        if result:
            pattern, count = result
            print(f"Most repeated pattern of length {length}: {pattern} (repeated {count} times)")

    array = chords_to_vector(chords)


