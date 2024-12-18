import numpy as np
from pychromagram import Image
from pychromaprint import plot_chromagram, kClassifiersTest2
import chromaprint

DATASET_VERSION="v0.1"

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


def reverse_quantized_to_classified(quantized_results, classifiers, num_rows, num_columns=16):
    """
    Reconstruct the rolling window from quantized results.

    Parameters:
    - quantized_results: Quantized results from the classifiers.
    - classifiers: List of classifiers used to generate quantized results.
    - num_rows: Number of rows in the original rolling window.
    - num_columns: Number of columns in the rolling window (default is 12 for chroma features).

    Returns:
    - reconstructed_rolling_window: Reconstructed rolling window.
    """
    reconstructed_rolling_window = [[0.0] * num_columns for _ in range(num_rows)]
    #rolling_image = RollingIntegralImage(max_rows=num_rows)

    # Iterate through rows and classifiers to reconstruct each row
    for row_idx in range(num_rows):
        reconstructed_row = [0.0] * num_columns  # Initialize a row with zeros

        for classifier_idx, classifier in enumerate(classifiers):
            quantized_value = quantized_results[row_idx][classifier_idx]
            quantizer_obj = classifier.quantizer()

            # Dequantize the value
            dequantized_value = dequantize_value(quantized_value, quantizer_obj)

            # Add the dequantized value to the reconstructed row
            # Distribute it across the row, proportional to classifier influence (simplification)
            reconstructed_row[classifier_idx % num_columns] += dequantized_value

        reconstructed_rolling_window[row_idx] = reconstructed_row

    return reconstructed_rolling_window


def gray_to_binary(gray):
    """
    Convert a Gray code number to its binary equivalent.

    :param gray: Gray code (integer)
    :return: Original binary number (integer)
    """
    binary = gray
    while gray > 0:
        gray >>= 1
        binary ^= gray
    return binary


def decode_subfingerprint(bits, num_classifiers):
    """
    Decode the 32-bit subfingerprint back to the original classifications.

    :param bits: 32-bit integer representing the subfingerprint
    :param num_classifiers: Number of classifiers used (determines bit extraction)
    :return: List of original classification values
    """
    classifications = []
    for _ in range(num_classifiers):
        # Extract the last 2 bits
        gray_code = bits & 0b11
        # Decode the Gray code to the original classification
        classification = gray_to_binary(gray_code)
        classifications.insert(0, classification)  # Insert at the beginning to maintain order
        # Shift bits to process the next 2-bit chunk
        bits >>= 2
    return classifications


def predict_with_lookup_table(my_x, x_array=None, y_array=None, n_neighbor=0):
    N = 100000
    if x_array is None:
        x_array = np.load(f'data/X_train_{DATASET_VERSION}.npy')
    if y_array is None:
        y_array = np.load(f'data/y_train_{DATASET_VERSION}.npy')
    distances_x = np.mean((x_array[0:N] - my_x) ** 2, axis=1)
    neighbors = list(np.where(distances_x <= min(distances_x)))[0]
    if n_neighbor == 0:
        min_index_x = neighbors[0]
    else:
        neighbors = [i for i in np.argsort(distances_x) if i not in neighbors]
        min_index_x = neighbors[0]

    my_y = y_array[min_index_x]
    return my_y


def classified_to_chromagram(reconstructed_classified):
    sample_input_list = reconstructed_classified

    predicted_output = predict_with_lookup_table(sample_input_list)
    image = Image(12)
    image.data = np.array(predicted_output).reshape(-1,12)
    # Print predictions
    print("Predicted Output for the Sample:")
    print(np.array(predicted_output).reshape(-1,12))
    return image




def divide_list(lst, chunk_size=11):
    """
    Divide a list into sublists of a specified chunk size.

    :param lst: List to divide
    :param chunk_size: Size of each sublist (default is 26)
    :return: List of sublists
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


if __name__ == "__main__":

    fingerprint_encoded = (
        "AQAA3Uk2JYqTOPhzaJGH_rgy5EfPww9q5Dw0UczR_Mh14cGpHOEraFHz4Md1XB900WjOHOGPLzu-C3kMMc6MD_nxHX1juBt63Bm64cmOtHoE"
        "J80l6OTga9GCipuRK_iToc8wNkfMPlB6C18Hh_YQ_8GV4w_SL9BUHdWiD-eC_BNcZbhy5OPhH0-y4NGhI01yPDzxH65w_DE-aYMthN8hLc-K"
        "8DL8B1Vz9IkO__hx5IogaiNjxDuYjBl6ClNy_MON2NoGnVG8IGzRROhzFW9-_IlwfAi5LPTQR4EWRllVIS-m5giT5pBzH5mO-7iPPcfxZHpx"
        "fejboNHG4ziSigibJDvcH04eHRXu4Yd74anRI72MhLWOi_gTOPdWPFbgfDhxElvVimhP5IqhHlkUD02d4-hVNPER-TinFFr2fMgPH-3CGLVI5"
        "DuuQ_lmvIgjSj0u2EW94z4OHvnB59Cl4z0RdR7RhFtWHN6OK8bOw-_gpJJMZMpRNgr0JEQz_eiDHlLS47WRi0qK_0ejZFlwSjr8o8-PZyya5R"
        "p6Hq-OMImVQ2uWoSS2owzeDH-I_dCJRswleLrwZUS6fTjhKx_OLMVf9CFxKSFCXzm0IHqCL-VwKUnS4-ZxTPmM_IftFWAEUIgKIQBAABFjFDB"
        "MMYWMMgQAooRBRILFEAIWKMEIAkQIYQQxwBgjCECGAmMMIEggg6xggAigDDCCEEWEEVA5YBhyDgiIABBCGGBMJNohIbAiiCAqhSAGAAoSAg5Q"
        "Jx2iQBmkgACIGCYIB5AJAIxChCkkMEHAQE-EMgYwAIxCBigilLAEKQEIIEAQAQxhQghgHCWAGEIEAwQBQYwRwAAnCAHGCAMA"
    )
    fingerprint_decoded = chromaprint.decode_fingerprint(fingerprint_encoded.encode('utf8'))
    quantized_list = []
    for subfingerprint in fingerprint_decoded[0]:
        quantized_classified = decode_subfingerprint(subfingerprint, 16)

        print(quantized_classified)
        quantized_list.append(quantized_classified)
    print(fingerprint_decoded)

    sublists = divide_list(quantized_list)

    full_image = []
    count = 0
    last_images = []
    n_images_avg = 1
    if len(quantized_list) >=22:
        n_images_avg = 2

    for quantized_fixed in sublists:
        print(quantized_fixed)
        # Call the function to reverse
        num_rows = len(quantized_fixed)  # Original rolling window row count
        reconstructed_classified = reverse_quantized_to_classified(quantized_fixed, kClassifiersTest2, num_rows)

        # Output reconstructed rolling window
        print("\nReconstructed Classified:")
        for row in reconstructed_classified:
            print(row)

        num_rows = len(quantized_fixed)  # Original rolling window row count
        reconstructed_classified = reverse_quantized_to_classified(quantized_fixed, kClassifiersTest2, num_rows)
        reconstructed_classified = [item for sublist in reconstructed_classified for item in sublist]
        if len(reconstructed_classified) < 176:
            padding = 176 - len(reconstructed_classified)
            reconstructed_classified.extend(padding * [0])
        try:

            reconstructed_classified = np.array(reconstructed_classified).reshape(-1,176)
            image = classified_to_chromagram(reconstructed_classified)
            last_images.append(image.data)
            if count % n_images_avg == n_images_avg - 1:
                my_array = np.array(last_images)
                avg_image = np.mean(abs(my_array), axis=0)
                full_image.extend(avg_image)
                last_images = []

                image = Image(12)
                image.data = full_image
            count = count + 1
        except ValueError as e:
            print(e)


    print(np.array(full_image).shape)
    image = Image(12)
    image.data = full_image
    plot_chromagram(image, improve=True, debug=False)





