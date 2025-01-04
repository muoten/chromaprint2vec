SEED=2
DEQUANTIZE=True
NORMALIZE=False
import numpy as np
np.random.seed(SEED)
import time
from sklearn.model_selection import train_test_split
from pychromaprint import (
    RollingIntegralImage, kClassifiersTest2, plot_chromagram
)
from pychromagram import Image
import os
import pandas as pd

# Assuming generate_synthetic_images and simulate_pipeline are already defined

def dequantize_value(quantized_value, quantizer):
    """
    Dequantize the quantized value by approximating it to the midpoint of the quantizer bins.

    Parameters:
    - quantized_value: The quantized value (integer).
    - quantizer: The Quantizer object containing bin thresholds.

    Returns:
    - Dequantized value (float).
    """
    # Map quantized values back to approximate original values using bin midpoints
    if quantized_value == 0:
        return quantizer.a - 0.001  # Assume a midpoint for values below `a`
    elif quantized_value == 1:
        return (quantizer.a + quantizer.b) / 2
    elif quantized_value == 2:
        return (quantizer.b + quantizer.c) / 2
    elif quantized_value == 3:
        return (quantizer.c + 0.05)  # Assume extrapolated range for values above `c`
    else:
        raise ValueError(f"Unexpected quantized value: {quantized_value}")

def apply_classifiers_with_filters_fixed(original_image, classifiers):
    # Use max_rows equal to the actual number of rows in the rolling window
    num_actual_rows = len(original_image)
    rolling_image = RollingIntegralImage(max_rows=256)

    # Add each row from the original image to the RollingIntegralImage
    for i, row in enumerate(original_image):
        rolling_image.add_row(row)

    # Sanity check: rolling_image.num_rows should equal the number of rows added
    if rolling_image.num_rows != num_actual_rows:
        raise ValueError(f"Unexpected num_rows: {rolling_image.num_rows} (expected {num_actual_rows})")

    multiresults = []

    for offset in range(0, num_actual_rows - 16 +1):
        # Initialize results for all classifiers
        results = []

        for classifier in classifiers:
            filter_obj = classifier.filter()
            quantizer_obj = classifier.quantizer()

            # Apply the filter to the RollingIntegralImage
            filter_value = filter_obj.apply(rolling_image, offset)

            if DEQUANTIZE:
                # Quantize the filter output
                quantized_value = quantizer_obj.quantize(filter_value)
                # and dequantize
                refilter_value = dequantize_value(quantized_value, quantizer_obj)
                requantized_value = quantizer_obj.quantize(refilter_value)
                assert quantized_value == requantized_value, \
                    f"type={classifier}, filtered={filter_value}, q={quantized_value}, refiltered={refilter_value}, requantized={requantized_value}"
                filter_value = refilter_value
            # Append results
            results.append(filter_value)
        multiresults.append(results)

    return multiresults


def generate_synthetic_images(num_images, image_size=(1,12)):
    # Define the scaling values
    scaling_matrix = np.ones((26, 12))  # Start with ones
    ndarray = np.ones((1, 26, 12))

    note_duration=8
    n_slots = int(26 / note_duration)

    n_levels = 3
    offset = 26-n_slots*note_duration
    idx_to_scale = np.random.randint(0,12,n_slots)
    idx_to_scale2 = np.random.randint(0, 12, n_slots)
    scale_factor = 4 *np.random.randint(0, n_levels, n_slots) + 1
    scale_factor2 = 4 * np.random.randint(0, n_levels, n_slots) + 1

    # Assign scaling factors to specific locations
    for i in range(n_slots):
        scaling_matrix[offset+note_duration*i:offset+note_duration*(i+1), idx_to_scale[i]] = scale_factor[i]
        scaling_matrix[offset + note_duration * i:offset + note_duration * (i + 1), idx_to_scale2[i]] = scale_factor2[i]

    # Apply scaling
    scaled_ndarray = ndarray.copy()
    scaled_ndarray[0] *= scaling_matrix

    norm_images = scaled_ndarray / np.linalg.norm(scaled_ndarray, axis=2, keepdims=True)
    return norm_images


def simulate_pipeline(image):
    """Simulate the effect of classifiers and rolling integral window."""
    output = apply_classifiers_with_filters_fixed(image.reshape(-1, 12), kClassifiersTest2)
    return output


def generate_dataset():
    # Generate data
    n_samples = N_SAMPLES
    input_data = []
    output_data = []
    start_time = time.time()
    for i in range(n_samples):
        if i % 10000 == 0:
            elapsed_time = time.time() - start_time
            print(f"{i} of {n_samples} samples generated in {elapsed_time:.0f}s...")

        image = generate_synthetic_images(N_IMAGES_PER_SAMPLE)
        while np.isnan(image).any():
            image = generate_synthetic_images(N_IMAGES_PER_SAMPLE)
        classifier_output = simulate_pipeline(image)

        input_data.append(classifier_output)
        output_data.append(image.flatten())  # Flatten image for the NN

    # Convert to arrays
    inputs = np.array(input_data).reshape(n_samples, -1)
    outputs = np.array(output_data).reshape(n_samples, -1)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Save datasets
    np.save(f"data/X_train_{DATASET_VERSION}.npy", X_train)
    np.save(f"data/X_test_{DATASET_VERSION}.npy", X_test)
    np.save(f"data/y_train_{DATASET_VERSION}.npy", y_train)
    np.save(f"data/y_test_{DATASET_VERSION}.npy", y_test)

    return X_train, X_test, y_train, y_test


def get_dataset():
    # Load or generate dataset
    if os.path.exists(f"data/X_train_{DATASET_VERSION}.npy"):
        print("Loading existing dataset...")
        X_train = np.load(f"data/X_train_{DATASET_VERSION}.npy")
        X_test = np.load(f"data/X_test_{DATASET_VERSION}.npy")
        y_train = np.load(f"data/y_train_{DATASET_VERSION}.npy")
        y_test = np.load(f"data/y_test_{DATASET_VERSION}.npy")
    else:
        print("Generating new dataset...")
        X_train, X_test, y_train, y_test = generate_dataset()
    return X_train, X_test, y_train, y_test

def predict_with_lookup_table(my_x, x_array=None, y_array=None, n_neighbor=0):
    if x_array is None:
        x_array = np.load(f'data/X_train_{DATASET_VERSION}.npy')
    if y_array is None:
        y_array = np.load(f'data/y_train_{DATASET_VERSION}.npy')
    distances_x = np.mean((x_array[0:N_SAMPLES] - my_x) ** 2, axis=1)
    neighbors = list(np.where(distances_x <= min(distances_x)))[0]
    if n_neighbor == 0:
        min_index_x = neighbors[0]
    else:
        neighbors = [i for i in np.argsort(distances_x) if i not in neighbors]
        min_index_x = neighbors[0]
    my_x = x_array[min_index_x]
    my_y = y_array[min_index_x]
    return my_y, my_x


# Main script
if __name__ == "__main__":
    DATASET_VERSION = "v0.2"
    N_IMAGES_PER_SAMPLE = 26
    N_SAMPLES = 100000

    x_array, X_test, y_array, y_test = get_dataset()
    print(x_array.shape)
    print(y_array.shape)

    actual_x_deq = [
        [1.9811500000000002, -0.232167, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.318214,
         -0.08979124999999999, -0.02923892, 0.093078, 0.01176092, -0.140235, -0.04761435, 0.12726635, 0.027982919999999998,
         0.1609915, 0.026257549999999998],
        [
            1.9811500000000002, -0.232167, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.318214, -0.08979124999999999, -0.02923892, 0.265855, 0.01176092, -0.140235, -0.04761435, 0.12726635, 0.027982919999999998, 0.1609915, 0.026257549999999998],
        [
            1.9811500000000002, -0.232167, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.318214, -0.08979124999999999, -0.02923892, 0.265855, 0.01176092, -0.140235, 0.08444365, 0.12726635, 0.027982919999999998, 0.1609915, 0.026257549999999998],
        [
            1.9811500000000002, -0.232167, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.318214, -0.395222, -0.02923892, 0.265855, 0.01176092, -0.140235, 0.08444365, 0.12726635, 0.027982919999999998, 0.1609915, 0.026257549999999998],
        [
            1.9811500000000002, -0.232167, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.318214, -0.395222, -0.02923892, 0.265855, 0.01176092, -0.140235, 0.08444365, -0.039456649999999996, 0.027982919999999998, 0.1609915, 0.026257549999999998],
        [
            1.9811500000000002, -0.232167, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.318214, -0.395222, -0.02923892, 0.265855, 0.01176092, -0.140235, 0.08444365, -0.039456649999999996, 0.027982919999999998, 0.1609915, 0.026257549999999998],
        [
            1.9811500000000002, -0.232167, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.7084655, -0.558409, -0.02923892, 0.093078, 0.01176092, -0.140235, 0.189239, -0.039456649999999996, 0.027982919999999998, 0.1609915, -0.0982054],
        [
            1.9811500000000002, -0.466689, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.827319, -0.558409, 0.042344730000000004, 0.093078, 0.01176092, -0.140235, 0.189239, -0.039456649999999996, 0.027982919999999998, 0.1609915, 0.1346339],
        [
            1.9811500000000002, -0.8446505, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.827319, -0.558409, 0.042344730000000004, 0.093078, 0.01176092, -0.140235, 0.08444365, -0.039456649999999996, 0.027982919999999998, 0.1609915, 0.1346339],
        [
            1.9811500000000002, -1.0390899999999998, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.827319, -0.558409, -0.02923892, 0.093078, 0.0792026, -0.140235, 0.08444365, -0.039456649999999996, 0.027982919999999998, 0.1609915, 0.1346339],
        [
            1.9811500000000002, -1.0390899999999998, 0.6084970000000001, -0.045022200000000005, -0.0585087, -0.827319, -0.558409, -0.02923892, 0.093078, -0.02273128, -0.140235, 0.08444365, -0.039456649999999996, 0.027982919999999998, 0.1609915, 0.1346339]

    ]

    actual_x_deq = np.array(actual_x_deq)
    transformed_array = [actual_x_deq[0], (actual_x_deq[0]+actual_x_deq[1])/2]  # Start with the first row

    for i in range(2, len(actual_x_deq)):
        averaged_row = (actual_x_deq[i] + actual_x_deq[i - 1] + actual_x_deq[i-2]) / 3
        transformed_array.append(averaged_row)

    actual_x_deq = actual_x_deq.reshape(-1,176)

    sample_input_list = [
        [2.41926389e-05, 2.49136555e-05, 2.77506293e-05, 2.80395247e-05
            , 3.18899937e-05, 3.28552537e-05, 3.54579837e-05, 3.78714863e-05
        , 4.12119108e-05, 4.44705896e-05, 4.78097346e-05, 5.09243549e-05],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0.02060239, 0.03559487, 0.85031669, 0.52145399, 0.04453175, 0.01017828
    , 0.02712364, 0.01010208, 0.00264364, 0.00853458, 0.01867243, 0.00170553],
    [1.05989182e-02, 1.31322219e-02, 9.07262360e-01, 4.18792335e-01
        , 1.55279984e-02, 3.79951908e-03, 2.67441453e-02, 4.11277034e-03
    , 1.02775940e-03, 4.22771269e-03, 1.40030050e-02, 6.81674753e-04],
    [8.90484969e-03, 8.87703937e-03, 9.12958824e-01, 4.06304492e-01
        , 1.02703141e-02, 2.78623540e-03, 3.08878230e-02, 2.88454518e-03
    , 7.13098043e-04, 3.63382089e-03, 1.32348543e-02, 5.01113486e-04],
    [8.03616167e-03, 6.41355769e-03, 9.15166001e-01, 4.01028863e-01
        , 7.35462527e-03, 2.29794228e-03, 3.59988092e-02, 2.13747677e-03
    , 5.34285976e-04, 3.49426292e-03, 1.29928802e-02, 4.04008187e-04],
    [7.28928274e-03, 4.36885188e-03, 9.16519621e-01, 3.97443436e-01
        , 5.00635590e-03, 1.93986806e-03, 4.17475100e-02, 1.50672230e-03
    , 3.94314260e-04, 3.59345651e-03, 1.30824921e-02, 3.26718211e-04],
    [6.31971517e-03, 2.30136324e-03, 9.16313232e-01, 3.97179944e-01
        , 2.69740884e-03, 1.61954585e-03, 4.85237992e-02, 8.53358329e-04
    , 2.62375392e-04, 4.12665842e-03, 1.38047514e-02, 2.49434816e-04],
    [4.89358895e-03, 6.77660583e-04, 9.12456439e-01, 4.05306050e-01
        , 8.97949457e-04, 1.36248414e-03, 5.31021638e-02, 3.31527045e-04
    , 1.75487594e-04, 5.94262350e-03, 1.63662940e-02, 1.81714707e-04],
    [3.49938953e-03, 5.12807020e-04, 9.00973806e-01, 4.30475829e-01
        , 5.86370706e-04, 1.21689076e-03, 4.81787246e-02, 2.68409573e-04
    , 1.81664281e-04, 1.03617526e-02, 2.22203597e-02, 1.51996536e-04],
    [4.09577116e-03, 4.94155261e-04, 8.81728800e-01, 4.68185883e-01
        , 3.83784355e-04, 1.05177682e-03, 4.37830425e-02, 2.46750564e-04
    , 1.92714700e-04, 1.90515957e-02, 3.25268408e-02, 1.35667889e-04],
    [1.12371575e-02, 6.41411870e-04, 8.55183837e-01, 5.11655206e-01
        , 1.06586169e-03, 1.32671484e-03, 5.45532964e-02, 4.45855178e-04
    , 4.00717688e-04, 3.47487842e-02, 5.05572117e-02, 2.06963929e-04],
    [0.01124086, 0.00725606, 0.28939326, 0.6293186, 0.71878815, 0.01189378
    , 0.0347158, 0.02861083, 0.0052843, 0.01852115, 0.02727803, 0.00969279],
    [0.00193934, 0.00551973, 0.03086197, 0.45355305, 0.8900484, 0.00595494
    , 0.00617617, 0.03077883, 0.00278113, 0.00201078, 0.00404542, 0.00822817],
    [7.25599492e-04, 4.61243764e-03, 1.29181901e-02, 3.95900355e-01
        , 9.17668424e-01, 3.56562505e-03, 2.74540349e-03, 2.96570006e-02
    , 1.70931167e-03, 6.50920941e-04, 1.85144196e-03, 7.23397157e-03],
    [3.94252254e-04, 4.08206079e-03, 8.16674575e-03, 3.51751017e-01
        , 9.35586450e-01, 2.30604371e-03, 1.70788383e-03, 2.84409938e-02
    , 1.11967795e-03, 3.51818312e-04, 1.24761251e-03, 6.75605374e-03],
    [2.45571027e-04, 3.62840032e-03, 5.11974978e-03, 3.06892303e-01
        , 9.51324865e-01, 1.39334179e-03, 1.15511220e-03, 2.66683543e-02
    , 6.76625415e-04, 2.15916930e-04, 9.17796799e-04, 6.54632196e-03],
    [1.58733333e-04, 3.14373735e-03, 2.50447258e-03, 2.52215487e-01
        , 9.67354669e-01, 6.98159932e-04, 7.94280788e-04, 2.34570701e-02
    , 3.22029319e-04, 1.30281950e-04, 6.94844430e-04, 6.64492069e-03]
    ]
    actual_y = np.array(sample_input_list).reshape(1,-1)

    labeled_y, labeled_x = predict_with_lookup_table(actual_x_deq)
    actual_x = pd.read_csv('../data/classifier_output.csv', header=None)[0]
    actual_x = np.array(actual_x)
    mse_x = np.mean((actual_x - actual_x_deq) ** 2)
    print(f"mse actual_x vs x_deq: {mse_x}")
    mse_x = np.mean((actual_x - labeled_x) ** 2)
    print(f"mse actual_x vs label_x_for_x_deq: {mse_x}")
    mse_y = np.mean((labeled_y - actual_y) ** 2)
    print(f"mse actual_y vs label_y_for_x_deq: {mse_y}")

    neighbor_y, neighbor_x = predict_with_lookup_table(actual_x_deq, n_neighbor=1)
    mse_x2 = np.mean((actual_x - neighbor_x) ** 2)
    print(f"mse actual_x vs neighbor_for_x_deq: {mse_x2}")
    mse_y2 = np.mean((actual_y - neighbor_y) ** 2)
    print(f"mse actual_y vs neighbor_y_for_x_deq: {mse_y2}")

    assert (neighbor_y != labeled_y).any()
    image = Image(12)
    image.data = actual_y.reshape(-1,12)
    plot_chromagram(image, label=f"actual_y")
    image.data = neighbor_y.reshape(-1,12)
    plot_chromagram(image, label=f"neighbor_y for dataset {DATASET_VERSION}")
    image.data = labeled_y.reshape(-1,12)
    plot_chromagram(image, label=f"label_y for dataset {DATASET_VERSION}")
    print(f"margin to get actual_y closer to label_y_for_x_deq than to neighbor_y_for_x_deq is: {mse_y2 - mse_y}")
    assert mse_y2 >= mse_y, f"neighbor_y_for_x_deq is closer to actual_y than label_y_for_x_deq in dataset {DATASET_VERSION}"


