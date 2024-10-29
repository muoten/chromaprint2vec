"""
This code has been derived from https://github.com/acoustid/chromaprint that was published under MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math
import logging
import ctypes
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from pydub import AudioSegment
from collections import deque

NUM_BANDS = 12

SAMPLE_RATE = 11025
FRAME_SIZE = 4096
OVERLAP = FRAME_SIZE - FRAME_SIZE / 3 #// 2720;
MIN_FREQ = 28
MAX_FREQ = 3520

DFT_R2C = 0  # Real to complex
DFT_C2R = 1  # Complex to real

# Load the FFmpeg shared libraries
libavutil = ctypes.CDLL('/usr/local/lib/libavcodec.so')  # Change path if necessary
# Define constants from FFmpeg's avfft.h

# Function prototypes for initializing, calculating, and freeing RDFT
libavutil.av_rdft_init.argtypes = [ctypes.c_int, ctypes.c_int]
libavutil.av_rdft_init.restype = ctypes.c_void_p

libavutil.av_rdft_calc.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
libavutil.av_rdft_end.argtypes = [ctypes.c_void_p]

# Constants
kMinSampleRate = 1000
kMaxBufferSize = 1024 * 32

#USE_INTERNAL_AVRESAMPLE = False  # Set to True if using an internal resampling method

class AudioConsumer:
    def consume(self, data, length):
        # Placeholder for consuming audio data
        pass

class AudioProcessor:
    def __init__(self, sample_rate, consumer):
        self.m_buffer = np.zeros(kMaxBufferSize, dtype=np.int16)
        self.m_buffer_offset = 0
        self.m_resample_buffer = np.zeros(kMaxBufferSize, dtype=np.int16)
        self.m_target_sample_rate = sample_rate
        self.m_consumer = consumer
        self.m_resample_ctx = None
        self.m_num_channels = 2

    def __del__(self):
        if self.m_resample_ctx:
            self.close_resample_ctx()

    def load_mono(self, input_data, length):
        output = self.m_buffer[self.m_buffer_offset:self.m_buffer_offset + length]
        output[:] = input_data[:length]

    def load_stereo(self, input_data, length):
        input_data = np.array(input_data, dtype=np.int16)
        output = self.m_buffer[self.m_buffer_offset:self.m_buffer_offset + length]

        # Average left and right channels to produce mono output
        output[:] = (input_data[0:length * 2:2] + input_data[1:length * 2:2]) // 2

    def load_multi_channel(self, input_data, length):
        output = self.m_buffer[self.m_buffer_offset:self.m_buffer_offset + length]
        output[:] = [np.mean(input_data[i*self.m_num_channels:(i+1)*self.m_num_channels]) for i in range(length)]

    def load(self, input_data, length):
        length = min(length, kMaxBufferSize - self.m_buffer_offset)

        self.m_buffer[self.m_buffer_offset:self.m_buffer_offset+length] = input_data[self.m_buffer_offset:self.m_buffer_offset+length]
        self.m_buffer_offset += length

        return length

    def resample(self):
            self.m_consumer.consume(self.m_buffer[:self.m_buffer_offset], self.m_buffer_offset)
            self.m_buffer_offset = 0

    def reset(self, sample_rate, num_channels):
        if num_channels <= 0:
            logging.debug("AudioProcessor::reset() -- No audio channels.")
            return False
        if sample_rate <= kMinSampleRate:
            logging.debug(f"AudioProcessor::reset() -- Sample rate less than {kMinSampleRate} ({sample_rate}).")
            return False

        self.m_buffer_offset = 0

        self.m_num_channels = num_channels
        return True

    def consume(self, input_data, length):
        if length % self.m_num_channels != 0:
            logging.debug("AudioProcessor::consume() -- Length not divisible by number of channels.")
            return

        while length > 0:
            consumed = self.load(input_data, length)

            input_data = input_data[consumed:]
            length -= consumed
            if self.m_buffer_offset >= kMaxBufferSize:
                self.resample()
                if self.m_buffer_offset >= kMaxBufferSize:
                    logging.debug("AudioProcessor::consume() -- Resampling failed?")
                    return

    def flush(self):
        if self.m_buffer_offset:
            self.resample()

    def close_resample_ctx(self):
        # Close the resampling context if it exists
        self.m_resample_ctx = None


class AudioConsumer:
    def consume(self, input_data, length):
        raise NotImplementedError("This method should be implemented by subclasses.")


class FFTFrameConsumer:
    def consume_frame(self, frame):
        raise NotImplementedError("This method should be implemented by subclasses.")


class AudioSlicer:
    def __init__(self, frame_size, overlap):
        self.size = frame_size
        self.increment = frame_size - overlap

    def size(self):
        return self.size

    def increment(self):
        return self.increment


class FFT(AudioConsumer):
    def __init__(self, frame_size, overlap, consumer):
        self.m_slicer = AudioSlicer(frame_size, overlap)
        self.m_frame = np.zeros(frame_size, dtype=np.float32)
        self.m_consumer = consumer
        self.m_lib = None  # Placeholder for an FFT library if needed

    def frame_size(self):
        return self.m_slicer.size

    def increment(self):
        return self.m_slicer.increment

    def overlap(self):
        return self.m_slicer.size - self.m_slicer.increment

    def reset(self):
        # Reset any necessary internal state
        self.m_frame.fill(0)

    def consume(self, input_data, length):
        frame_size = int(self.m_slicer.size)
        increment = int(self.m_slicer.increment)
        frame_count = 0  # Track the number of frames processed

        for i in range(0, length - frame_size + 1, increment):
            self.m_frame[:] = input_data[i:i + frame_size]

            fft_result = fft_custom(self.m_frame)

            self.m_consumer.consume(fft_result)
            frame_count += 1


def fft_custom(m_input):
    m_input = compute_av_rdft(m_input)
    m_frame_size = m_input.shape[0]
    frame = np.zeros(m_frame_size // 2 + 1)  # Output array with appropriate size

    # First element (left channel of first sample)
    frame[0] = m_input[0] ** 2

    # Middle element (right channel of first sample)
    frame[m_frame_size // 2] = m_input[1] ** 2

    # Process remaining samples
    output_index = 1
    for i in range(1, m_frame_size // 2):
        frame[output_index] = m_input[2 * i] ** 2 + m_input[2 * i + 1] ** 2
        output_index += 1

    return frame


def prepare_hamming_window(input, scale=1.0):
    """
    Applies a scaled Hamming window to the input array `data`.

    Parameters:
    - data: np.array, the input array that will be modified in place.
    - scale: float, the scaling factor to apply to the Hamming window.
    """
    # scale = 3.05185e-05
    data = input.copy()
    size = len(data)
    indices = np.arange(size)

    # Apply the Hamming window with the specified scale
    data[:] = scale * (0.54 - 0.46 * np.cos(indices * 2.0 * np.pi / (size - 1)))

    return data


def compute_av_rdft(m_input):
    m_input = m_input[0:FRAME_SIZE]
    hamming_input = prepare_hamming_window(m_input)

    m_input = m_input * hamming_input
    frame_size = len(m_input)

    bits = int(np.log2(frame_size))

    # Initialize RDFT context
    m_rdft_ctx = libavutil.av_rdft_init(bits, DFT_R2C)
    if not m_rdft_ctx:
        raise RuntimeError("Failed to initialize RDFT context")

    # Prepare the input as a ctypes array
    input_array = (ctypes.c_float * frame_size)(*m_input)

    # Perform the RDFT
    libavutil.av_rdft_calc(m_rdft_ctx, input_array)

    # Convert the results back to a NumPy array
    output = np.ctypeslib.as_array(input_array, shape=(frame_size,))

    # Free the RDFT context
    libavutil.av_rdft_end(m_rdft_ctx)

    return output


def freq_to_octave(freq, base=440.0 / 16.0):
    return math.log(freq / base) / math.log(2.0)

def freq_to_index(freq, frame_size, sample_rate):
    return round((freq / sample_rate) * frame_size)

def index_to_freq(index, frame_size, sample_rate):
    return (index * sample_rate) / frame_size

class Chroma:
    def __init__(self, min_freq, max_freq, frame_size, sample_rate, consumer):
        self.interpolate = False
        self.notes = np.zeros(frame_size, dtype=int)
        self.notes_frac = np.zeros(frame_size)
        self.features = np.zeros(NUM_BANDS)
        self.consumer = consumer
        self.prepare_notes(min_freq, max_freq, frame_size, sample_rate)

    def prepare_notes(self, min_freq, max_freq, frame_size, sample_rate):
        self.min_index = max(1, freq_to_index(min_freq, frame_size, sample_rate))
        self.max_index = min(frame_size // 2, freq_to_index(max_freq, frame_size, sample_rate))
        for i in range(self.min_index, self.max_index):
            freq = index_to_freq(i, frame_size, sample_rate)
            octave = freq_to_octave(freq)
            note = NUM_BANDS * (octave - math.floor(octave))
            self.notes[i] = int(note)
            self.notes_frac[i] = note - self.notes[i]

    def reset(self):
        self.features.fill(0.0)

    def consume(self, frame):
        self.features.fill(0.0)

        for i in range(self.min_index, self.max_index):
            note = self.notes[i]
            energy = frame[i]

            if self.interpolate:
                note2 = note
                a = 1.0
                if self.notes_frac[i] < 0.5:
                    note2 = (note + NUM_BANDS - 1) % NUM_BANDS
                    a = 0.5 + self.notes_frac[i]
                elif self.notes_frac[i] > 0.5:
                    note2 = (note + 1) % NUM_BANDS
                    a = 1.5 - self.notes_frac[i]
                self.features[note] += energy * a
                self.features[note2] += energy * (1.0 - a)
            else:
                self.features[note] += energy

        self.consumer.consume(self.features)


class FeatureVectorConsumer:
    def consume(self, features):
        raise NotImplementedError("This method should be implemented by subclasses")

def euclidean_norm(features):
    return np.sqrt(np.sum(np.square(features)))

def normalize_vector(features, norm_func, epsilon=0.01):
    norm = norm_func(features)
    if norm > epsilon:
        features /= norm
    return features


class ChromaNormalizer(FeatureVectorConsumer):
    def __init__(self, consumer):
        self.consumer = consumer

    def reset(self):
        pass

    def consume(self, features):
        # Normalize the feature vector using the euclidean norm
        features =  normalize_vector(np.array(features), euclidean_norm, epsilon=0.01)
        # Pass the normalized features to the consumer
        self.consumer.consume(features)


class ChromaFilter:
    def __init__(self, coefficients, length, consumer):
        self.coefficients = coefficients
        self.length = length
        self.buffer = deque(maxlen=8)  # Circular buffer with max size 8
        self.result = np.zeros(12)     # Result vector for 12 elements
        self.buffer_offset = 0
        self.buffer_size = 1
        self.consumer = consumer

    def reset(self):
        self.buffer.clear()
        self.buffer_size = 1
        self.buffer_offset = 0

    def consume(self, features):
        if len(self.buffer) < 8:
            self.buffer.append(features)
        else:
            self.buffer[self.buffer_offset] = features

        self.buffer_offset = (self.buffer_offset + 1) % 8

        if self.buffer_size >= self.length:
            offset = (self.buffer_offset + 8 - self.length) % 8
            self.result.fill(0)  # Reset result array to zero

            for i in range(12):
                for j in range(self.length):
                    self.result[i] += self.buffer[(offset + j) % 8][i] * self.coefficients[j]

            # Pass the computed result to the consumer
            self.consumer.consume(self.result)
        else:
            self.buffer_size += 1


class Image:
    def __init__(self, num_columns):
        self.num_columns = num_columns
        self.data = []

    def num_columns(self):
        return self.num_columns

    def add_row(self, row):
        if len(row) == self.num_columns:
            self.data.append(row)
        else:
            raise ValueError("Row length does not match the number of columns")

    def num_rows(self):
        return len(self.data)

class ImageBuilder:
    def __init__(self, image):
        self.m_image = image
        self.consume_calls = 0

    def consume(self, features):
        self.consume_calls += 1
        # Ensure that features has the correct number of columns
        assert len(features) == self.m_image.num_columns, "Feature vector size does not match image columns"
        self.m_image.add_row(features)


def load_audio_raw_file(file_name):
    # Read the binary audio file
    with open(file_name, "rb") as file:
        # Read entire file content as bytes
        data = file.read()

    # Convert bytes to a numpy array of 16-bit signed integers
    # The format is little-endian, 2 bytes per sample
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Return the data as a Python list (or leave as numpy array if preferred)
    return audio_data.tolist()


def load_audio_file(file_name, convert_to_mono_11025hz=False, force_to_mono=False):
    """
    Load an audio file (WAV or MP3) and optionally convert it to mono 11025 Hz.

    Args:
        file_name (str): Path to the audio file (WAV or MP3).
        convert_to_mono_11025hz (bool): Whether to convert to mono and resample to 11025 Hz.

    Returns:
        int: Sample rate of the audio.
        list: Audio data as a list of samples.
    """
    # Load the audio file based on extension
    if file_name.endswith('.wav'):
        sample_rate, audio_data = wavfile.read(file_name)
    elif file_name.endswith('.mp3'):
        # Load MP3 file with pydub and convert to a numpy array
        try:
            audio = AudioSegment.from_file(file_name, format="mp3")
        except:
            audio = AudioSegment.from_file(file_name, format="mp4")
        sample_rate = audio.frame_rate
        # Convert pydub audio to numpy array
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.int16)

        # If stereo, reshape to 2 channels
        if audio.channels == 2:
            if not force_to_mono:
                audio_data = audio_data.reshape((-1, 2))
            else:
                audio_data = audio_data.reshape((-1,1))
    else:
        raise ValueError("Unsupported file format. Please provide a WAV or MP3 file.")

    # Convert to mono and resample to 11025 Hz if the flag is set
    if convert_to_mono_11025hz:
        # Convert to mono if stereo (average channels)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1).astype(np.int16)

        # Set new sample rate and calculate target number of samples for 11025 Hz
        new_sample_rate = SAMPLE_RATE
        num_samples = int(len(audio_data) * new_sample_rate / sample_rate)

        # Resample audio data to 11025 Hz
        audio_data = resample(audio_data, num_samples).astype(np.int16)

    # Return the modified sample rate and audio data as a list
    return audio_data.tolist()


def get_chromagram(data, with_filter=True):
    kChromaFilterCoefficients = [0.25, 0.75, 1.0, 0.75, 0.25]
    image = Image(12)
    image_builder = ImageBuilder(image)

    chroma_normalizer = ChromaNormalizer(image_builder)
    if with_filter:
        chroma_filter = ChromaFilter(kChromaFilterCoefficients, 5, chroma_normalizer)
        chroma = Chroma(MIN_FREQ, MAX_FREQ, FRAME_SIZE, SAMPLE_RATE, chroma_filter)
    else:
        chroma = Chroma(MIN_FREQ, MAX_FREQ, FRAME_SIZE, SAMPLE_RATE, chroma_normalizer)
    fft = FFT(FRAME_SIZE, OVERLAP, chroma)
    processor = AudioProcessor(SAMPLE_RATE, fft)
    processor.consume(data, len(data))
    processor.flush()
    return image


if __name__ == "__main__":

    data_from_stereo_averaged_to_mono = load_audio_file(
        "data/test_stereo_44100.mp3",
        convert_to_mono_11025hz=True
    )
    data_from_stereo_forced_to_mono = load_audio_file(
        "data/test_stereo_44100.mp3",
        convert_to_mono_11025hz=True,
        force_to_mono=True
    )

    chromagram = get_chromagram(data_from_stereo_forced_to_mono)
    for chroma_i in chromagram.data[:3]:
        print(chroma_i)

    expected_chromagram_with_filter = [
        [0.17243727, 0.31603434, 0.27871295, 0.16988209, 0.27645537, 0.31513423,
         0.16186715, 0.3202802,  0.27854012, 0.53804745, 0.15820927, 0.25926723],
        [0.14968698, 0.33201177, 0.30408855, 0.13384971, 0.32613825, 0.53775281,
        0.18715183, 0.23339042, 0.29814066, 0.27168197, 0.25030315, 0.21583479],
    [0.15063698, 0.31642244, 0.38615, 0.13157507, 0.12301962, 0.4425213, 0.19064319,
    0.23128468, 0.25860547, 0.22460978, 0.49426179, 0.21987294],
    [0.16049811, 0.34707715, 0.39079478, 0.15258541, 0.17104928, 0.37965467,
     0.14044558, 0.2270588, 0.29741916, 0.41785431, 0.39068898, 0.13191096],
    [0.14258307, 0.40953339, 0.53006773, 0.19330024, 0.44644346, 0.27537545,
    0.1399127, 0.11181377, 0.30499829, 0.26906501, 0.0932297, 0.11082735],
    [0.15809035, 0.4195309, 0.53949397, 0.20718744, 0.38233524, 0.32180061,
     0.14176973, 0.15859543, 0.30256126, 0.20412893, 0.15212064, 0.11696678],
    [0.30541002, 0.50217945, 0.51400054, 0.2700884, 0.15610186, 0.1735658,
    0.16925094, 0.24732029, 0.27099378, 0.16545649, 0.16560631, 0.21178103],
    [0.13948886, 0.3377352,  0.56132538, 0.27169872, 0.11890006, 0.1295974,
     0.21844212, 0.36937665, 0.33006235, 0.20520093, 0.21904716, 0.25189762],
    [0.18307542, 0.2670518, 0.56248821, 0.23970926, 0.26957857, 0.25501542,
     0.20850878, 0.2900477, 0.25867738, 0.21046239, 0.25188267, 0.28522075],
    [0.27970326, 0.27662484, 0.41920432, 0.17679585, 0.22226869, 0.29735982,
     0.26721408, 0.30540411, 0.27968882, 0.17349793, 0.29080941, 0.37796427],
    [0.21295005, 0.30305554,  0.32377798,  0.24305494,  0.28436297,  0.34125838,
     0.28399177,  0.31390832,  0.3759774,  0.22673138,  0.25468023,  0.2544882],
    [0.25248418, 0.29415492, 0.34328053, 0.24914779, 0.20612187, 0.42470939,
     0.23300224, 0.28603925, 0.36299176, 0.32080321, 0.2238095, 0.16145366],
    [0.24242121,  0.34001849, 0.44156608, 0.1929343, 0.17630103, 0.40064193,
     0.18583257,  0.25751297, 0.31392504, 0.3206122, 0.27149567, 0.16100136],
    [0.12294481, 0.30536938, 0.35379875, 0.13907676, 0.43709941, 0.47694233,
     0.2145147,  0.18018452, 0.31104592, 0.25800595, 0.2037315,  0.21279604],
    [0.12345586, 0.25436967, 0.37491806,  0.13900457,  0.55596139, 0.36455294,
     0.25671003, 0.18060368, 0.32387967,  0.21928005,  0.18721743, 0.1778401],
    [0.18727434, 0.27113646, 0.37640037, 0.24490499, 0.41931916, 0.15497968,
     0.17999685, 0.17374946, 0.59199845, 0.18437472, 0.16764666, 0.12149505],
    [0.04997527,  0.12204737, 0.09515486, 0.05567836, 0.15486676, 0.09418786,
     0.15019259,  0.51703827,  0.77878197,  0.19530447,  0.03681454,  0.04135736],
    [0.02652326, 0.0743342,  0.11570304, 0.01619919, 0.07121448, 0.02711689,
    0.05109763, 0.56990243, 0.79943763, 0.07058784, 0.02895727, 0.04464391],
    [0.12226437,  0.09058947,  0.71720136,  0.05280816,  0.23118741,  0.08847959,
     0.21163125,  0.2729781,  0.4264543,  0.18621638,  0.04906796,  0.2452548],
    [0.10824974, 0.06776861, 0.59016163, 0.04651423, 0.20443866, 0.06178398,
     0.1862858,  0.31709564, 0.62678466, 0.14034667, 0.04868519, 0.19350498],
    [0.06097855,  0.03074863,  0.33565482,  0.02471136,  0.14189258,  0.03421474,
     0.11138536,  0.35300558,  0.84068005,  0.08789279,  0.02119662,  0.09391562],
    [0.03301159, 0.0254605,  0.23451282, 0.01574245, 0.08297685, 0.02552662,
    0.070789,  0.35012424, 0.89386624, 0.06970749, 0.01460061, 0.06175684],
    [0.02196194,  0.02175716,  0.17013593,  0.01173161,  0.05521172,  0.01913778,
     0.05283033, 0.40370745, 0.89227976, 0.0453849, 0.01032362, 0.04945268],
    [0.03046363, 0.04036765, 0.1716292,  0.01530151, 0.10792501, 0.03591863,
     0.07890163, 0.52810209, 0.81327066, 0.06131391, 0.02735633, 0.06147785]
    ]

    len_with_filter = len(expected_chromagram_with_filter)
    assert chromagram.num_rows() == len_with_filter, "Numbers of rows doesn't match"

    # Compare each value in the chromagram with the generated image
    for y in range(len_with_filter):
        for x in range(12):
            np.testing.assert_almost_equal(chromagram.data[y][x], expected_chromagram_with_filter[y][x], decimal=2,
                                           err_msg=f"Image not equal at ({x}, {y})")

    chromagram = get_chromagram(data_from_stereo_averaged_to_mono, with_filter=False)
    for chroma_i in chromagram.data[:3]:
        print(chroma_i)

    expected_chromagram = [
        [0.155444, 0.268618, 0.474445, 0.159887, 0.1761, 0.423511, 0.178933, 0.34433, 0.360958, 0.30421, 0.200217,
         0.17072],
        [0.159809, 0.238675, 0.286526, 0.166119, 0.225144, 0.449236, 0.162444, 0.371875, 0.259626, 0.483961,
         0.24491, 0.17034],
        [0.156518, 0.271503, 0.256073, 0.152689, 0.174664, 0.52585, 0.141517, 0.253695, 0.293199, 0.332114,
         0.442906, 0.170459],
        [0.154183, 0.38592, 0.497451, 0.203884, 0.362608, 0.355691, 0.125349, 0.146766, 0.315143, 0.318133,
         0.172547, 0.112769],
        [0.201289, 0.42033, 0.509467, 0.259247, 0.322772, 0.325837, 0.140072, 0.177756, 0.320356, 0.228176,
         0.148994, 0.132588],
        [0.187921, 0.302804, 0.46976, 0.302809, 0.183035, 0.228691, 0.206216, 0.35174, 0.308208, 0.233234, 0.316017,
         0.243563],
        [0.213539, 0.240346, 0.308664, 0.250704, 0.204879, 0.365022, 0.241966, 0.312579, 0.361886, 0.277293,
         0.338944, 0.290351],
        [0.227784, 0.252841, 0.295752, 0.265796, 0.227973, 0.451155, 0.219418, 0.272508, 0.376082, 0.312717,
         0.285395, 0.165745],
        [0.168662, 0.180795, 0.264397, 0.225101, 0.562332, 0.33243, 0.236684, 0.199847, 0.409727, 0.247569, 0.21153,
         0.147286],
        [0.0491864, 0.0503369, 0.130942, 0.0505802, 0.0694409, 0.0303877, 0.0389852, 0.674067, 0.712933, 0.05762,
         0.0245158, 0.0389336],
        [0.0814379, 0.0312366, 0.240546, 0.134609, 0.063374, 0.0466124, 0.0752175, 0.657041, 0.680085, 0.0720311,
         0.0249404, 0.0673359],
        [0.139331, 0.0173442, 0.49035, 0.287237, 0.0453947, 0.0873279, 0.15423, 0.447475, 0.621502, 0.127166,
         0.0355933, 0.141163],
        [0.115417, 0.0132515, 0.356601, 0.245902, 0.0283943, 0.0588233, 0.117077, 0.499376, 0.715366, 0.100398,
         0.0281382, 0.0943482],
        [0.047297, 0.0065354, 0.181074, 0.121455, 0.0135504, 0.030693, 0.0613105, 0.631705, 0.73548, 0.0550565,
         0.0128093, 0.0460393],
    ]

    len_without_filter = len(expected_chromagram)
    # Validate the dimensions of the image
    assert chromagram.num_rows()==len_without_filter, "Numbers of rows doesn't match"

    # Compare each value in the chromagram with the generated image
    for y in range(len_without_filter):
        for x in range(12):
            np.testing.assert_almost_equal(chromagram.data[y][x], expected_chromagram[y][x],decimal=2,
                                   err_msg=f"Image not equal at ({x}, {y})")
