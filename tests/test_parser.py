"""test_parser.py

Test the dicom_contour_parser.dicom_contour_parser module.
"""


import unittest
import tempfile
import numpy as np
from dicom_contour_parser import DicomContourParser, InvalidDataFolder, Record


class test_record(unittest.TestCase):
    """Test the correctness of the Record class
    """

    def test_dicom_only(self):
        """Check a DICOM only entry
        """
        r = Record('SCD0000101', 'SC-HF-I-1', 1,
                   ('final_data/dicoms/SCD0000101/1.dcm', '', ''))
        self.assertTrue(r.has_dicom())
        self.assertFalse(np.all(r.data[0] == np.zeros_like(r.data[0])))
        self.assertFalse(r.has_icontour())
        self.assertTrue(np.all(np.logical_not(r.data[1])))
        self.assertFalse(r.has_ocontour())
        self.assertTrue(np.all(np.logical_not(r.data[2])))

    def test_dicom_icontour_only(self):
        """Check a DICOM, i-contour only entry
        """
        r = Record('SCD0000101', 'SC-HF-I-1', 128,
                   ('final_data/dicoms/SCD0000101/128.dcm',
                    'final_data/contourfiles/SC-HF-I-1/i-contours/' +
                    'IM-0001-0128-icontour-manual.txt', ''))
        self.assertTrue(r.has_dicom())
        self.assertFalse(np.all(r.data[0] == np.zeros_like(r.data[0])))
        self.assertTrue(r.has_icontour())
        self.assertFalse(np.all(np.logical_not(r.data[1])))
        self.assertFalse(r.has_ocontour())
        self.assertTrue(np.all(np.logical_not(r.data[2])))

    def test_full_entry(self):
        """Check a DICOM, i-contour, o-contour, full entry
        """
        r = Record('SCD0000101', 'SC-HF-I-1', 59,
                   ('final_data/dicoms/SCD0000101/59.dcm',
                    'final_data/contourfiles/SC-HF-I-1/i-contours/' +
                    'IM-0001-0059-icontour-manual.txt',
                    'final_data/contourfiles/SC-HF-I-1/o-contours/' +
                    'IM-0001-0059-ocontour-manual.txt'))
        self.assertTrue(r.has_dicom())
        self.assertFalse(np.all(r.data[0] == np.zeros_like(r.data[0])))
        self.assertTrue(r.has_icontour())
        self.assertFalse(np.all(np.logical_not(r.data[1])))
        self.assertTrue(r.has_ocontour())
        self.assertFalse(np.all(np.logical_not(r.data[2])))


class test_parser(unittest.TestCase):
    """Test the correctness of the DicomContourParser class
    """

    def setUp(self):
        import os.path as opath
        self.TEST_FOLDER = opath.join(opath.dirname(opath.realpath(__file__)),
                                      '../final_data/')

    def test_emtpy_folder(self):
        """Initializing with an empty folder should fail with an exception
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(InvalidDataFolder):
                DicomContourParser(tmpdirname)

    def test_known_records(self):
        """Known records should be of certain properties
        """
        parser = DicomContourParser(self.TEST_FOLDER)
        records = parser.record_list
        for r in records:
            if r.patient_id == 'SCD0000101':
                if r.serial_id == 1:
                    # DICOM only record
                    self.assertTrue(r.has_dicom())
                    self.assertFalse(np.all(r.data[0] ==
                                            np.zeros_like(r.data[0])))
                    self.assertFalse(r.has_icontour())
                    self.assertTrue(np.all(np.logical_not(r.data[1])))
                    self.assertFalse(r.has_ocontour())
                    self.assertTrue(np.all(np.logical_not(r.data[2])))
                elif r.serial_id == 128:
                    self.assertTrue(r.has_dicom())
                    self.assertFalse(np.all(r.data[0] ==
                                            np.zeros_like(r.data[0])))
                    self.assertTrue(r.has_icontour())
                    self.assertFalse(np.all(np.logical_not(r.data[1])))
                    self.assertFalse(r.has_ocontour())
                    self.assertTrue(np.all(np.logical_not(r.data[2])))
                elif r.serial_id == 59:
                    self.assertTrue(r.has_dicom())
                    self.assertFalse(np.all(r.data[0] ==
                                            np.zeros_like(r.data[0])))
                    self.assertTrue(r.has_icontour())
                    self.assertFalse(np.all(np.logical_not(r.data[1])))
                    self.assertTrue(r.has_ocontour())
                    self.assertFalse(np.all(np.logical_not(r.data[2])))

    def test_one_epoch(self):
        """Only images should be returned and the count should be correct
        """
        parser = DicomContourParser(self.TEST_FOLDER)
        data = []
        for chunk in parser.random_shuffled_iterator(100):
            data.extend(chunk)
        self.assertEqual(len(data), 1140)  # pre-counted manually
        for item in data:
            self.assertEqual(len(item), 3)
            self.assertEqual(type(item[0]), np.ndarray)
            self.assertEqual(type(item[1]), np.ndarray)
            self.assertEqual(type(item[2]), np.ndarray)

    def test_async_loader_speed(self):
        """Async loader should out-perform sync loader with comp tasks.
        """
        SLEEP_TIME = 1
        from time import time, sleep
        st1 = time()
        p_sync = DicomContourParser(self.TEST_FOLDER, async_load=False)
        for chunk in p_sync.random_shuffled_iterator(300, tag_records=True):
            sleep(SLEEP_TIME)
        lapse1 = time() - st1
        st2 = time()
        p_sync = DicomContourParser(self.TEST_FOLDER, async_load=True)
        for chunk in p_sync.random_shuffled_iterator(300, tag_records=True):
            sleep(SLEEP_TIME)
        lapse2 = time() - st2
        print('async {:.6f} s vs. sync {:.6f} s'.format(lapse2, lapse1))
        self.assertLess(lapse2, lapse1)

    def test_async_loader_correctness(self):
        """Async loader should produce the same results.
        """
        p_sync = DicomContourParser(self.TEST_FOLDER, async_load=False)
        d_sync = []
        for chunk in p_sync.random_shuffled_iterator(100, tag_records=True):
            d_sync.extend(chunk)
        p_async = DicomContourParser(self.TEST_FOLDER, async_load=True)
        d_async = []
        for chunk in p_async.random_shuffled_iterator(100, tag_records=True):
            d_async.extend(chunk)
        self.assertEqual(len(d_sync), len(d_async))
        d_sync.sort(key=lambda e: e[0])
        d_async.sort(key=lambda e: e[0])
        for (_, e_sync), (_, e_async) in zip(d_sync, d_async):
            np.testing.assert_allclose(e_sync[0], e_async[0])
            np.testing.assert_allclose(e_sync[1], e_async[1])

    def test_random_shuffle(self):
        """Each epoch should produce the same set of records at a random order
        """
        parser = DicomContourParser(self.TEST_FOLDER)
        pass1 = []
        for chunk in parser.random_shuffled_iterator(100, tag_records=True):
            pass1.extend((item[0] for item in chunk))
        pass2 = []
        for chunk in parser.random_shuffled_iterator(100, tag_records=True):
            pass2.extend((item[0] for item in chunk))
        self.assertNotEqual(pass1, pass2)
        self.assertEqual(set(pass1), set(pass2))
