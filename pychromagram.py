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
OVERLAP = FRAME_SIZE - FRAME_SIZE / 3
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
    scale = 3.05185e-05
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

        self.m_buffer = [None] * 8  # Initialize buffer with 8 slots
        self.m_buffer_offset = 0
        self.m_length = length
        self.m_buffer_size = 1
        self.m_result = [0.0] * 12  # Assuming there are 12 chroma bins
        self.m_coefficients = coefficients  # Coefficients array for the weighted sum
        self.m_consumer = consumer  # Consumer to process the result

    def reset(self):
        self.buffer.clear()
        self.buffer_size = 1
        self.buffer_offset = 0

    def consume(self, features):
        new_features = features.copy()

        # Place features in the buffer at the current offset
        self.m_buffer[self.m_buffer_offset] = new_features
        self.m_buffer_offset = (self.m_buffer_offset + 1) % 8

        # If buffer is full enough, calculate the result
        if self.m_buffer_size >= self.m_length:
            offset = (self.m_buffer_offset + 8 - self.m_length) % 8
            self.m_result = [0.0] * 12  # Reset result

            # Weighted sum over the buffer for each of the 12 chroma bins
            for i in range(12):
                for j in range(self.m_length):
                    buffer_index = (offset + j) % 8
                    self.m_result[i] += self.m_buffer[buffer_index][i] * self.m_coefficients[j]

            # Pass result to consumer
            self.m_consumer.consume(self.m_result)
        else:
            # Increment buffer size if we haven't yet reached full length
            self.m_buffer_size += 1


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
    data = load_audio_raw_file('data/test_stereo_44100.raw')
    data_from_stereo_averaged_to_mono = load_audio_file(
        "data/test_stereo_44100.mp3",
        convert_to_mono_11025hz=True
    )

    chromagram = get_chromagram(data)
    for chroma_i in chromagram.data[:3]:
        print(chroma_i)

    expected_chromagram_with_filter = [
        [0.17225318376277746, 0.3179838757671984, 0.49747260316433317, 0.153243676873149, 0.24939430771386348,
         0.3444203683514613, 0.24199319139474568, 0.3120011043425964, 0.3292773509605172, 0.3029122011399653,
         0.1642425723202446, 0.1853892590462166],
        [0.16794030068642132, 0.3071718879090583, 0.429539380094578, 0.16280983962970524, 0.26782434597159155, 0.3643863994757741, 0.23566729422880286, 0.32609018423445757, 0.2793441684645535, 0.38953515226283497, 0.16781081201508583, 0.20522642514425973],
        [0.16315234316871344, 0.3143877880224486, 0.363579785959635, 0.16081796421192218, 0.2761049579358149, 0.40183723116299636, 0.21574458301962707, 0.3188128763726338, 0.266339332322072, 0.4088684875941483, 0.2091213855483044, 0.22079999075513268],
        [0.16234289410992678, 0.3269938772612027, 0.34949854894194204, 0.1505876218412775, 0.24658964788847834, 0.4378543144334966, 0.19065541635321387, 0.283732384007475, 0.28021168765731314, 0.3691797924182992, 0.3008188214277394, 0.21728607965821928],
        [0.15836760328699626, 0.37348667491901705, 0.4450665813591509, 0.16559921516258405, 0.30372650350239705, 0.38900682761520944, 0.16176219450207288, 0.20169433232377856, 0.29937235019778274, 0.32272595712359103, 0.2800969191675948, 0.16512480209744881],
        [0.15325858892912256, 0.3997876685760253, 0.5053955161854804, 0.18167379606302927, 0.37319372188362, 0.32877145200306396, 0.14620716840230974, 0.15375083573310788, 0.303753628068942, 0.29726317350849046, 0.19553050279634307, 0.13041362397419973],
        [0.1552277958061543, 0.41169658555965216, 0.5247848753045289, 0.19170245094071506, 0.3987364963642118, 0.3012407322363342, 0.14107677399736415, 0.1403190852496165, 0.3026583066559813, 0.27651329971304045, 0.1489479020506572, 0.12118271590200153],
        [0.16534518528580852, 0.42320005862027055, 0.5357354154872663, 0.20478035561562866, 0.3905865114385111, 0.2838327302355325, 0.1444505152055395, 0.1493724565057813, 0.30213814365617775, 0.25158371677094793, 0.13227433284871543, 0.12743627164188995],
        [0.18578985838143758, 0.42947474876573843, 0.5461778972178881, 0.2298912193345434, 0.3296876525694731, 0.2583036672418508, 0.16161331766550144, 0.19847237307079899, 0.3030011607989485, 0.2209673846633326, 0.15165320873550206, 0.15910196800144824],
        [0.20695421195248762, 0.39286472671317063, 0.5407543420360885, 0.256095262044106, 0.2210390731758571, 0.2180924664833372, 0.19814326213628472, 0.29071159648345, 0.29711481073136997, 0.19545663266133, 0.2112576220794602, 0.23134465549539718],
        [0.2084135675197894, 0.33030542053708456, 0.5085887346244289, 0.24556559224559252, 0.21380541000242728, 0.23124702569249866, 0.2285546252188432, 0.32503939512474944, 0.29470079713875696, 0.2009309319966573, 0.25155948928910066, 0.2844598595709213],
        [0.22182603502830706, 0.299703727640081, 0.44836014934651475, 0.23141445413686804, 0.2440359432303481, 0.291498460739123, 0.24696106534274337, 0.31898239241313975, 0.3091078707881925, 0.21446249025579722, 0.2671605538667059, 0.29361605371307115],
        [0.24084193577160148, 0.30131216337845634, 0.39112740918517386, 0.22985480246634132, 0.2444814357456425, 0.35000741683523684, 0.24976806483343808, 0.3055529039066293, 0.33466525006450504, 0.24410664973912802, 0.2610412519524396, 0.2600727069732589],
        [0.23145922503106228, 0.31225179695344396, 0.3738258347818056, 0.21971648887178494, 0.2539127508876197, 0.40129198891749557, 0.23529287375467065, 0.279098181405384, 0.3458537640256063, 0.27890173354557973, 0.24546679991377748, 0.21091212045158644],
        [0.1947794421527313, 0.31307194602838756, 0.3790213352428766, 0.18868064306241597, 0.33122888966374214, 0.43138488286232485, 0.22719384299808587, 0.23705750437978862, 0.3341140069091983, 0.28363722116986906, 0.2294028895186162, 0.1909519367132057],
        [0.16296392774736843, 0.3008085522747008, 0.37900030023302933, 0.16900218387724508, 0.42584425903205886, 0.40719663939413514, 0.2306927916734055, 0.20377297532052555, 0.34914353201464504, 0.2587253525409224, 0.21240740905039512, 0.18682093110641285],
        [0.10958588832490358, 0.22036685108211015, 0.2557761015240598, 0.12516611897015356, 0.34450589630122075, 0.2409792355895762, 0.2076303880569036, 0.3835183127038303, 0.6385507790712746, 0.23185724966507792, 0.12635098352009913, 0.11798296940033537],
        [0.028160859500435623, 0.0778894171546293, 0.11763621848313126, 0.018563287244827798, 0.07716866923051834, 0.031115203840921285, 0.05513785259363104, 0.5684728261501475, 0.7981867025338668, 0.07566334617056789, 0.030180325597265904, 0.045769277531421036],
        [0.0400504899749258, 0.07234511764748298, 0.18464414695676218, 0.0199376732801257, 0.09690539456536255, 0.032522973410020646, 0.0718747835862832, 0.5343074689880091, 0.8045012594420572, 0.07614681530222604, 0.027756928252157403, 0.061491903258535895],
        [0.052033423495519725, 0.06820707672125706, 0.2559855017654645, 0.023215931642744297, 0.1196884374181108, 0.03780657438490886, 0.0897814251640163, 0.5027269215728807, 0.7972753383766975, 0.08426005431749045, 0.02815362185536138, 0.08051516298525767],
        [0.06630215248996509, 0.06624944294370097, 0.34594090528969246, 0.028192267355457268, 0.14635182050678175, 0.046276209575802245, 0.1123014579040355, 0.46755065853406624, 0.7703933283508944, 0.09983667115964016, 0.030611758891547144, 0.10751382354817471],
        [0.08505765968790359, 0.06673907107784953, 0.46536771872832194, 0.03562490259109725, 0.17868521622175526, 0.05794504150681838, 0.14328403994419486, 0.41475839120395724, 0.710452931551007, 0.12306338921556939, 0.03559454747553217, 0.1465572671966741],
        [0.09868848186247385, 0.0644710707329798, 0.5538913941934341, 0.041606056686710385, 0.1996309105177307, 0.0646394072360731, 0.16704488409878607, 0.3519649288811134, 0.6550966133612808, 0.13993422581141673, 0.039936980870861324, 0.17729958649665972],
        [0.08597224899407757, 0.052661838093182325, 0.4876071211231275, 0.03677714602114767, 0.17598873007590385, 0.05372181538981477, 0.1487076052389396, 0.340727125787828, 0.7337746648836672, 0.1241027793562579, 0.03484910453572821, 0.15366856786614902],
        [0.0571602027110563, 0.036265939258429705, 0.33810016001844634, 0.025101941276418734, 0.1245137842646764, 0.03608834466824953, 0.10438288883689281, 0.3604263982009455, 0.8391054499865341, 0.0895680429274269, 0.023380708873482405, 0.10099302977020234],
        [0.03789559281635714, 0.0282094261988132, 0.24138615087169493, 0.017581946055292372, 0.0914446904081252, 0.027453896110534455, 0.07649324013757688, 0.38722080124827896, 0.8743669112922812, 0.06774801399482783, 0.017166557571020123, 0.06987675919366926]
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
