# data from 'https://raw.githubusercontent.com/jesse-a-l-hamer/chord_progression_assistant/refs/heads/master/three_chord_songs.csv'
import numpy as np
import pandas as pd
from harmony_utils import chords_to_vector

EXCLUDE_ALL_MINOR = False
EXCLUDE_MORE_THAN_4CHORDS_PER_SONG = True
USE_SET_NOT_SEQUENCE = False
USE_TEMPO = True


# Function to generate chord progression based on pitch class and mode
def get_chord_progression(sequence, key, mode):

    pitch_class_to_note = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    roman_numerals = sequence.split(',')
    key = int(key)  # Convert key to integer for pitch-class indexing

    chords = []
    for numeral in roman_numerals:
        degree = int(numeral) - 1  # Convert Roman numeral to index (1-based to 0-based)
        assert degree <= 6
        assert degree <= len ([0, 2, 4, 5, 7, 9, 11])
        root_pitch = (key + [0, 2, 4, 5, 7, 9, 11][degree]) % 12  # Calculate pitch class

        # Determine chord type (major/minor/diminished)
        chord_root = pitch_class_to_note[root_pitch]
        chord_type = "m"
        if mode == 1 and degree+1 in [1,4,5] or mode == 0 and degree+1 in [3,6,7]:
            chord_type = ""
        elif mode == 1 and degree+1 in [7] or mode == 0 and degree+1 in [2]:
            chord_type = "d"

        chords.append(f"{chord_root}{chord_type}")

    return chords


def get_chord_progression_per_row(row):
    qualified_chords = get_chord_progression(row['cp'], row['key'], row['mode'])
    return qualified_chords


def is_valid_list(value):
    try:
        # Split the string into parts, convert to integers, and check conditions
        integers = [int(x) for x in value.split(',')]
        if all(i < 8 for i in integers):
            return True
    except (ValueError, AttributeError):
        # ValueError: Non-integer value in the string
        # AttributeError: Input is not a string
        return False
    return False


if __name__ == "__main__":
    # A 4 chord pattern does not mean the song only has these 4 chords. Further filter is required
    df = pd.read_csv('data/four_chord_songs.csv', sep=',')

    # Concatenate the `cp` column for combinations of `artist`, `song`, and `section`
    df_combined_per_section = (
        df.groupby(['artist', 'song', 'section'])['cp']
        .apply(lambda x: set(','.join(x)))
        .reset_index()
    )

    df_merge = pd.merge(df, df_combined_per_section, on=['artist', 'song', 'section'], how='left')
    less_than_5_chords_per_section = df_merge.cp_y.apply(lambda x: len(x) <=5)
    df = df[less_than_5_chords_per_section].reset_index(drop=True)

    if EXCLUDE_MORE_THAN_4CHORDS_PER_SONG:
        df_combined_per_song = (
            df.groupby(['artist', 'song'])['cp']
            .apply(lambda x: set(','.join(x)))
            .reset_index()
        )

        df_merge = pd.merge(df, df_combined_per_song, on=['artist', 'song'], how='left')
        less_than_5_chords_per_song = df_merge.cp_y.apply(lambda x: len(x) <=5)
        df = df[less_than_5_chords_per_song]

    # Filter NaN
    df = df[~df['key'].isna() & ~df['mode'].isna()]

    df = df[df['cp'].apply(is_valid_list)]

    df['chords'] = df.apply(
        lambda row: get_chord_progression(row['cp'], row['key'], row['mode']),
        axis=1
    )
    if EXCLUDE_ALL_MINOR:
        mask_outliers = df['chords'].apply(lambda x: sum(1 for chord in x if 'm' in chord)==4)
        df = df[~mask_outliers]

    df['chord_set'] = df['chords'].apply(lambda x: set(x))
    column_with_chords = 'chords'
    if USE_SET_NOT_SEQUENCE:
        column_with_chords = 'chord_set'

    if EXCLUDE_MORE_THAN_4CHORDS_PER_SONG:
        df = df.drop_duplicates(subset=['mode', 'key', 'song', 'artist'], keep='first')
        df['section'] = ''
        df = df.reset_index(drop=True)

    df['vectors'] = df[column_with_chords].apply(lambda x: np.array(chords_to_vector(x)).reshape(-1))

    df_vectors_reduced = pd.DataFrame(list(df['vectors'].values))
    df_vectors_reduced = df_vectors_reduced.iloc[:, 0:20]
    if USE_TEMPO:
        max_tempo = df['tempo'].max()
        df['norm_tempo'] = df['tempo']/max_tempo
        df_vectors_reduced[20]  = df['norm_tempo'].apply(lambda x: x if not pd.isna(x) else 0)

    df_vectors_reduced.to_csv('data/vectors.csv', sep='\t', header=False, index=False)
    df_metadata = df.loc[:,['chords','cp','key', 'section', 'song', 'artist', 'tempo']]
    if EXCLUDE_MORE_THAN_4CHORDS_PER_SONG:
        df_metadata = df_metadata.drop(columns='section')
    print(df_metadata)

    df_metadata.to_csv('data/metadata.csv', sep='\t', index=False)
