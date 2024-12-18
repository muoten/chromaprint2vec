"""
This code has been derived from https://github.com/acoustid/chromaprint that was published under MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import chromaprint
import math
import numpy as np
from typing import Callable, List
import matplotlib.pyplot as plt
from PIL import Image

from pychromagram import get_chromagram, load_audio_raw_file, load_audio_file

kDefaultFrameSize = 4096
kDefaultFrameOverlap = kDefaultFrameSize - kDefaultFrameSize // 3

kChromaFilterSize = 5
kChromaFilterCoefficients = [0.25, 0.75, 1.0, 0.75, 0.25]

from itertools import accumulate

class RollingIntegralImage:
    def __init__(self, max_rows):
        self.max_rows = max_rows # + 1  # Similar to C++ version
        self.num_columns = 0
        self.num_rows = 0
        self.data = []

    def reset(self):
        self.data.clear()
        self.num_rows = 0
        self.num_columns = 0

    def add_row(self, row):
        if self.num_columns == 0:
            self.num_columns = len(row)
            self.data = [0.0] * (self.max_rows * self.num_columns)

        # Compute cumulative sum for the new row
        current_row = list(accumulate(row))

        if self.num_rows > 0:
            # Add previous row values to make it an integral image
            last_row = self.get_row(self.num_rows - 1)
            current_row = [last_row[i] + current_row[i] for i in range(self.num_columns)]

        # Save the computed row in the data buffer
        self.set_row(self.num_rows, current_row)
        self.num_rows += 1

    def area(self, r1, c1, r2, c2):
        assert r1 <= self.num_rows and r2 <= self.num_rows
        assert c1 <= self.num_columns and c2 <= self.num_columns

        if r1 == r2 or c1 == c2:
            return 0.0

        assert r2 > r1 and c2 > c1

        def get_value(row, col):
            if row == 0 or col == 0:
                return 0
            return self.get_row(row - 1)[col - 1]

        return (get_value(r2, c2) - get_value(r2, c1) - get_value(r1, c2) + get_value(r1, c1))

    def get_row(self, i):
        """Retrieve row at index i (wraps around if max_rows is exceeded)"""
        index = (i % self.max_rows) * self.num_columns
        return self.data[index:index + self.num_columns]

    def set_row(self, i, row_data):
        """Set row at index i (wraps around if max_rows is exceeded)"""
        index = (i % self.max_rows) * self.num_columns
        self.data[index:index + self.num_columns] = row_data

# Utility functions
def subtract(a, b):
    return a - b

def subtract_log(a, b, epsilon=1e-10):
    ratio = (1.0 + a) / (1.0 + b + epsilon)
    if ratio <= 0:
        ratio = epsilon  # Ensure a positive value for log calculation
    r = math.log(ratio)
    assert not math.isnan(r)
    return r


# Filter functions for integral image processing
# Each filter function will accept an `IntegralImage` object with an `area` method,
# and a comparator `cmp` which is a callable (function or lambda).

def filter0(image, x, y, w, h, cmp):
    assert w >= 1
    assert h >= 1

    a = image.area(x, y, x + w, y + h)
    b = 0
    return cmp(a, b)

def filter1(image, x, y, w, h, cmp):
    assert w >= 1
    assert h >= 1

    h_2 = h // 2

    a = image.area(x, y + h_2, x + w, y + h)
    b = image.area(x, y, x + w, y + h_2)
    return cmp(a, b)

def filter2(image, x, y, w, h, cmp):
    assert w >= 1
    assert h >= 1

    w_2 = w // 2

    a = image.area(x + w_2, y, x + w, y + h)
    b = image.area(x, y, x + w_2, y + h)
    return cmp(a, b)

def filter3(image, x, y, w, h, cmp):
    assert x >= 0
    assert y >= 0
    assert w >= 1
    assert h >= 1

    w_2 = w // 2
    h_2 = h // 2

    a = image.area(x, y + h_2, x + w_2, y + h) + \
        image.area(x + w_2, y, x + w, y + h_2)
    b = image.area(x, y, x + w_2, y + h_2) + \
        image.area(x + w_2, y + h_2, x + w, y + h)
    return cmp(a, b)

def filter4(image, x, y, w, h, cmp):
    assert w >= 1
    assert h >= 1

    h_3 = h // 3

    a = image.area(x, y + h_3, x + w, y + 2 * h_3)
    b = image.area(x, y, x + w, y + h_3) + \
        image.area(x, y + 2 * h_3, x + w, y + h)
    return cmp(a, b)

def filter5(image, x, y, w, h, cmp):
    assert w >= 1
    assert h >= 1

    w_3 = w // 3

    a = image.area(x + w_3, y, x + 2 * w_3, y + h)
    b = image.area(x, y, x + w_3, y + h) + \
        image.area(x + 2 * w_3, y, x + w, y + h)
    return cmp(a, b)


# Modified Filter class to take parameters
class Filter:
    def __init__(self, filter_type=0, y=0, height=0, width=0):
        self._type = filter_type
        self._y = y
        self._height = height
        self._width = width

    def apply(self, image, x):
        if self._type == 0:
            return filter0(image, x, self._y, self._width, self._height, subtract_log)
        elif self._type == 1:
            return filter1(image, x, self._y, self._width, self._height, subtract_log)
        elif self._type == 2:
            return filter2(image, x, self._y, self._width, self._height, subtract_log)
        elif self._type == 3:
            return filter3(image, x, self._y, self._width, self._height, subtract_log)
        elif self._type == 4:
            return filter4(image, x, self._y, self._width, self._height, subtract_log)
        elif self._type == 5:
            return filter5(image, x, self._y, self._width, self._height, subtract_log)
        return 0.0


    def __str__(self):
        return f"Filter(type={self._type}, y={self._y}, height={self._height}, width={self._width})"

# Modified Quantizer class to take parameters
class Quantizer:
    def __init__(self, a=-1.0, b=0.0, c=1.0):
        self.a = a
        self.b = b
        self.c = c

    def quantize(self, value):
        # Placeholder quantization logic based on thresholds
        if value < self.a:
            return 0
        elif value < self.b:
            return 1
        elif value < self.c:
            return 2
        else:
            return 3

    def __str__(self):
        return f"Quantizer({self.a}, {self.b}, {self.c})"

# Classifier remains unchanged
class Classifier:
    def __init__(self, filter=None, quantizer=None):
        self.m_filter = filter if filter is not None else Filter()
        self.m_quantizer = quantizer if quantizer is not None else Quantizer()

    def classify(self, image, offset):
        value = self.m_filter.apply(image, offset)
        classification_result = self.m_quantizer.quantize(value)

        return classification_result

    def filter(self):
        return self.m_filter

    def quantizer(self):
        return self.m_quantizer

    def __str__(self):
        return f"Classifier({self.m_filter}, {self.m_quantizer})"


kClassifiersTest2 = [
	Classifier(Filter(0, 4, 3, 15), Quantizer(1.98215, 2.35817, 2.63523)),
	Classifier(Filter(4, 4, 6, 15), Quantizer(-1.03809, -0.651211, -0.282167)),
	Classifier(Filter(1, 0, 4, 16), Quantizer(-0.298702, 0.119262, 0.558497)),
	Classifier(Filter(3, 8, 2, 12), Quantizer(-0.105439, 0.0153946, 0.135898)),
	Classifier(Filter(3, 4, 4, 8), Quantizer(-0.142891, 0.0258736, 0.200632)),
	Classifier(Filter(4, 0, 3, 5), Quantizer(-0.826319, -0.590612, -0.368214)),
	Classifier(Filter(1, 2, 2, 9), Quantizer(-0.557409, -0.233035, 0.0534525)),
	Classifier(Filter(2, 7, 3, 4), Quantizer(-0.0646826, 0.00620476, 0.0784847)),
	Classifier(Filter(2, 6, 2, 16), Quantizer(-0.192387, -0.029699, 0.215855)),
	Classifier(Filter(2, 1, 3, 2), Quantizer(-0.0397818, -0.00568076, 0.0292026)),
	Classifier(Filter(5, 10, 1, 15), Quantizer(-0.53823, -0.369934, -0.190235)),
	Classifier(Filter(3, 6, 2, 10), Quantizer(-0.124877, 0.0296483, 0.139239)),
	Classifier(Filter(2, 1, 1, 14), Quantizer(-0.101475, 0.0225617, 0.231971)),
	Classifier(Filter(3, 5, 6, 4), Quantizer(-0.0799915, -0.00729616, 0.063262)),
	Classifier(Filter(1, 9, 2, 12), Quantizer(-0.272556, 0.019424, 0.302559)),
	Classifier(Filter(3, 4, 2, 14), Quantizer(-0.164292, -0.0321188, 0.0846339)),
]


class FingerprinterConfigurationTest2:
    def __init__(self):
        self.classifiers = []
        self.filter_coefficients = []
        self.frame_size = kDefaultFrameSize
        self.frame_overlap = kDefaultFrameOverlap
        self.interpolate = False
        self.sample_rate = 11025

        self.set_classifiers(kClassifiersTest2, 16)
        self.set_filter_coefficients(kChromaFilterCoefficients, kChromaFilterSize)
        self.set_frame_size(kDefaultFrameSize)
        self.set_frame_overlap(kDefaultFrameOverlap)
        self.set_interpolate(False)

    def set_classifiers(self, classifiers, num_classifiers):
        self.classifiers = classifiers[:num_classifiers]

    def set_filter_coefficients(self, coefficients, size):
        self.filter_coefficients = coefficients[:size]

    def set_frame_size(self, frame_size):
        self.frame_size = frame_size

    def set_frame_overlap(self, frame_overlap):
        self.frame_overlap = frame_overlap

    def set_interpolate(self, interpolate):
        self.interpolate = interpolate

    def __str__(self):
        return (f"FingerprinterConfigurationTest1("
                f"frame_size={self.frame_size}, "
                f"frame_overlap={self.frame_overlap}, "
                f"interpolate={self.interpolate}, "
                f"classifiers={self.classifiers})")




class FingerprintCalculator:
    def __init__(self, config):
        """
        Initialize FingerprintCalculator with a configuration object.

        :param config: A configuration object like FingerprinterConfigurationTest1
        """
        # Use configuration to initialize classifiers and other settings
        self.m_classifiers = config.classifiers
        self.m_num_classifiers = len(self.m_classifiers)

        self.m_frame_size = config.frame_size
        self.m_frame_overlap = config.frame_overlap
        self.m_filter_coefficients = config.filter_coefficients
        self.m_interpolate = config.interpolate
        self.m_max_filter_width = 16
        self.m_image = RollingIntegralImage(256)
        self.fingerprint = []

        # Calculate the maximum filter width from the classifiers
        for classifier in self.m_classifiers:
            self.m_max_filter_width = max(self.m_max_filter_width, classifier.filter()._width)

        assert self.m_max_filter_width > 0, "Max filter width should be greater than 0."
        assert self.m_max_filter_width < 256, "Max filter width should be less than 256."

    def gray_code(self, n):
        """
        Calculate the gray code of a number n.
        Gray code is a binary numeral system where two successive values differ in only one bit.

        :param n: Input integer
        :return: Gray code of n
        """
        return n ^ (n >> 1)

    def calculate_subfingerprint(self, offset):
        """
        Calculate the subfingerprint for the given offset using all classifiers.

        :param offset: Offset for classifying the image
        :return: 32-bit integer representing the subfingerprint
        """
        bits = 0
        for idx,classifier in enumerate(self.m_classifiers):
            classification = classifier.classify(self.m_image, offset)
            bits = (bits << 2) | self.gray_code(classification)

        return bits

    def consume(self, features):
        self.m_max_filter_width = 16

        for i,feature_row in enumerate(features):
            # Add the row of features to the integral image
            self.m_image.add_row(feature_row)

            # Check if the number of rows exceeds the filter width
            if self.m_image.num_rows >= self.m_max_filter_width:
                subfingerprint = self.calculate_subfingerprint(self.m_image.num_rows - self.m_max_filter_width)
                self.fingerprint.append(subfingerprint)

    def get_fingerprint(self):
        return self.fingerprint


    def process_image(self, image):
        """
        Process the full image and calculate subfingerprints at different offsets.

        :param image: Input image or feature matrix (2D numpy array)
        :return: List of subfingerprints
        """
        self.m_image = image

        num_offsets = image.shape[0]
        subfingerprints = []

        for offset in range(num_offsets):
            subfingerprint = self.calculate_subfingerprint(offset)
            subfingerprints.append(subfingerprint)

        return subfingerprints



def chromagram_to_image(chromagram, frame_size, frame_overlap):
    """
    Convert a chromagram (12, time) into the appropriate image format for the FingerprintCalculator.

    :param chromagram: numpy array of shape (12, time_frames)
    :param frame_size: The size of each frame in terms of time samples (e.g., 4096)
    :param frame_overlap: The overlap between frames in terms of time samples
    :return: numpy array of shape (num_windows, 12, frame_size)
    """
    num_time_frames = chromagram.shape[1]

    # If the frame_size is greater than the total number of time frames in the chromagram,
    # we treat the chromagram as a single window and pad it to fit the frame_size
    if num_time_frames < frame_size:
        print(
            f"Chromagram has fewer time frames ({num_time_frames}) than frame size ({frame_size}). Padding will be applied.")
        # Pad the chromagram so that it can be treated as a single frame of size `frame_size`
        padded_chromagram = np.pad(chromagram, ((0, 0), (0, frame_size - num_time_frames)), mode='constant')
        # Return a single window without flattening
        return np.expand_dims(padded_chromagram, axis=0)

    # Otherwise, we compute the number of overlapping windows
    num_windows = (num_time_frames - frame_size) // frame_overlap + 1

    # List to store each window
    image_windows = []

    # For each window
    for i in range(num_windows):
        start = i * frame_overlap
        end = start + frame_size

        # Extract the window from the chromagram (keeping the 12 chroma bins intact)
        window = chromagram[:, start:end]

        # If window is smaller than frame_size (e.g., at the end of the chromagram), pad it
        if window.shape[1] < frame_size:
            pad_width = frame_size - window.shape[1]
            window = np.pad(window, ((0, 0), (0, pad_width)), mode='constant')

        # Add to the list of windows without flattening
        image_windows.append(window)

    # Convert list of windows to a numpy array of shape (num_windows, 12, frame_size)
    image_for_calculator = np.array(image_windows)

    return image_for_calculator



def normalize_vector(vec: List[float], func: Callable[[List[float]], float], threshold: float = 0.01):
    norm = func(vec)
    if norm < threshold:
        # If the norm is below the threshold, set all elements to 0
        vec[:] = [0.0] * len(vec)
    else:
        # Normalize all elements by dividing by the norm
        vec[:] = [x / norm for x in vec]


# Define Euclidean (L2) norm
def euclidean_norm(arr: np.ndarray) -> float:
    # This will calculate the L2 norm (Euclidean norm)
    squares = np.sum(arr**2)  # Sum of squares
    return np.sqrt(squares) if squares > 0 else 0.0


def plot_chromagram(chromagram_in, label=""):
    chromagram = np.array(chromagram_in.data).T
    chromagram = (chromagram * 255).astype(np.uint8)
    image_original = Image.fromarray(chromagram.T)

    resize = 1

    image_original = image_original.transpose(Image.ROTATE_90)
    image_original = image_original.convert('L')

    array_resampled = np.repeat(image_original, resize, axis=1).T

    plt.imshow(array_resampled)
    plt.title("Generated Image")
    plt.colorbar(label="Intensity")
    plt.axis('off')  # Hide the axes
    plt.show()


# Create the configuration
config = FingerprinterConfigurationTest2()

# Create the FingerprintCalculator using the configuration
calculator = FingerprintCalculator(config)

#file_path = "data/test_stereo_44100.mp3"
file_path = 'data/test_stereo_44100.raw'

data_from_mp3file = load_audio_raw_file(file_path)
chromagram2 = get_chromagram(data_from_mp3file)

plot_chromagram(chromagram2)

# Process the image and calculate subfingerprints
calculator.consume(chromagram2.data)
fingerprint = calculator.get_fingerprint()

decoded_fp_actual = fingerprint
encoded_fp_actual = chromaprint.encode_fingerprint(decoded_fp_actual, algorithm=2)

print(encoded_fp_actual)

encoded_fp_expected2="AgAAC0kkZUqYREkUnFAXHk8uuMZl6EfO4zu-4ABKFGESWIIMEQE"
print(encoded_fp_expected2)
# assert encoded_fp_actual.decode('utf8') == encoded_fp_expected2
