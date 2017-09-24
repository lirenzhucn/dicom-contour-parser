"""test_parsing.py

Test the dicom_contour_parser.parsing module.

"""


import unittest
import tempfile
import os
import numpy as np
import matplotlib.path as mpl_path
from dicom_contour_parser import parsing


CONTOUR_STR = """
20.00 20.00
30.00 20.00
40.00 30.00
40.00 50.00
20.00 50.00
10.00 30.00
"""


CONTOUR_DATA = [
    (20.0, 20.0),
    (30.0, 20.0),
    (40.0, 30.0),
    (40.0, 50.0),
    (20.0, 50.0),
    (10.0, 30.0),
]


class test_parsing(unittest.TestCase):
    """Test the correctness of functions in the parsing module.
    """

    def test_parse_contour_file(self):
        """Test parse_contour_file.

        Just check one known case.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, 'test_contour.txt')
            with open(filename, 'w') as fp:
                fp.write(CONTOUR_STR)
            coords_lst = parsing.parse_contour_file(filename)
            self.assertEqual(coords_lst, CONTOUR_DATA)

    def test_parse_dicom_file(self):
        """Test parse_dicom_file.

        Always pass for now. Ideally we should have a few known cases.
        """

    def test_poly_to_mask(self):
        """Test poly_to_mask.

        Perform the inside check for all True pixels
        """
        max_x, max_y = 0, 0
        for x, y in CONTOUR_DATA:
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        mask = parsing.poly_to_mask(CONTOUR_DATA, round(max_x + 1),
                                    round(max_y + 1))
        polygon = mpl_path.Path(CONTOUR_DATA)
        for y, x in np.transpose(np.nonzero(mask)):
            self.assertTrue(polygon.contains_point((x, y)),
                            msg='({:.2f}, {:.2f}) is not inside the polygon'
                            .format(x, y))
