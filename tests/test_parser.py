"""test_parser.py

Test the dicom_contour_parser.dicom_contour_parser module.
"""


import unittest
import tempfile
from dicom_contour_parser import DicomContourParser, InvalidDataFolder


class test_parser(unittest.TestCase):
    """Test the correctness of the DicomContourParser class
    """

    def test_emtpy_folder(self):
        """Initializing with an empty folder should fail with an exception
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(InvalidDataFolder):
                DicomContourParser(tmpdirname)
