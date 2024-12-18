import unittest
from pychromaprint import (
    FingerprinterConfigurationTest2,
    FingerprintCalculator,
    load_audio_file,
    get_chromagram,
    chromaprint
)

class TestFingerprintCalculator(unittest.TestCase):

    def test_fingerprint_generation(self):
        # Create the configuration
        config = FingerprinterConfigurationTest2()

        # Create the FingerprintCalculator using the configuration
        calculator = FingerprintCalculator(config)

        # File path to test audio
        file_path = ('data/videoplayback_10s.wav')

        # Load audio data and process it
        data_from_mp3file = load_audio_file(file_path, convert_to_mono_11025hz=True)

        # Generate the chromagram
        chromagram2 = get_chromagram(data_from_mp3file)
        self.assertIsNotNone(chromagram2.data, "Chromagram data should not be None")

        # Process the image and calculate subfingerprints
        calculator.consume(chromagram2.data)
        fingerprint = calculator.get_fingerprint()

        # Encode the fingerprint using the expected parameters
        decoded_fp_actual = fingerprint
        encoded_fp_actual = chromaprint.encode_fingerprint(decoded_fp_actual, algorithm=2, base64=True)

        # Define the expected fingerprint output
        encoded_fp_expected = (
            "AgAAO0nYhJIkAfuJZ6lCNOuHv0bZ40F_HM15_DhxQaNlBWYmPAcGH7pjHO7BH8d6fDiO3niEH3eO5jzx7Tg-wDqqL8C"
            "HPQlRg8cPHIeP7_gl-MGFAz_8Dv8Lo5gxgoBmlADCOoqgIkoyYglACRgmgQRCKCUYYsIYIS5kxiigMRMAAAE"
        )

        # Validate the fingerprint
        self.assertEqual(
            encoded_fp_actual.decode('utf8'),
            encoded_fp_expected,
            "Encoded fingerprint does not match the expected value"
        )

if __name__ == '__main__':
    unittest.main()
