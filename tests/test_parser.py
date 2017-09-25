"""test_parser.py

Test the dicom_contour_parser.dicom_contour_parser module.
"""


import unittest
import tempfile
import numpy as np
from dicom_contour_parser import DicomContourParser, InvalidDataFolder


class test_parser(unittest.TestCase):
    """Test the correctness of the DicomContourParser class
    """

    def setUp(self):
        import os.path as opath
        # this folder was extracted with only the first patient's data
        self.TEST_FOLDER = opath.join(opath.dirname(opath.realpath(__file__)),
                                      '../test_data/')

    def test_emtpy_folder(self):
        """Initializing with an empty folder should fail with an exception
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(InvalidDataFolder):
                DicomContourParser(tmpdirname)

    def test_one_epoch(self):
        """Only images should be returned and the count should be correct
        """
        parser = DicomContourParser(self.TEST_FOLDER)
        data = []
        for chunk in parser.random_shuffled_iterator(100):
            data.extend(chunk)
        self.assertEqual(len(data), 240)  # pre-counted manually
        for item in data:
            self.assertEqual(len(item), 2)
            self.assertEqual(type(item[0]), np.ndarray)
            self.assertEqual(type(item[1]), np.ndarray)
