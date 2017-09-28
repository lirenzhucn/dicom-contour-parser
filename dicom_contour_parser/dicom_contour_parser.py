"""dicom_contour_parser.py

This module provides a class that can parse a data folder and produce the
entire data set as a list of records organized by patients and then slices.

author: Liren Zhu (liren.zhu.cn@gmail.com)
date: 2017-09-23

"""

import csv
import os
import re
import random
import numpy as np
import os.path as opath
from threading import Thread

from . import parsing


class InvalidDataFolder(Exception):
    pass


def _parse_dicom_and_contour_files(filenames):
    """Convert two filenames to valid image data

    :param filenames: a 3-tuple record of dicom_filename, icontour_filename,
           and ocontour_filename:
           dicom_filename is the path string to the DICOM file
           icontour_filename is the path string to the i-contour file
           ocontour_filename is the path string to the o-contour file
    :return: 3-tuple containing the DICOM image data and contour mask data
    """
    dicom_filename, icontour_filename, ocontour_filename = filenames
    dicom_data, icontour_data, ocontour_data = None, None, None
    if dicom_filename:
        dicom_data = parsing.parse_dicom_file(dicom_filename)
        if dicom_data is not None:
            dicom_data = dicom_data['pixel_data']
    if icontour_filename:
        icontour_path = parsing.parse_contour_file(icontour_filename)
        if dicom_data is not None:
            height, width = dicom_data.shape
        else:
            max_x, max_y = 0, 0
            for x, y in icontour_path:
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            height = round(max_x + 1)
            width = round(max_y + 1)
        icontour_data = parsing.poly_to_mask(icontour_path, width, height)
    if ocontour_filename:
        ocontour_path = parsing.parse_contour_file(ocontour_filename)
        if dicom_data is not None:
            height, width = dicom_data.shape
        else:
            max_x, max_y = 0, 0
            for x, y in ocontour_path:
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            height = round(max_x + 1)
            width = round(max_y + 1)
        ocontour_data = parsing.poly_to_mask(ocontour_path, width, height)
    # TODO: fix the case in which all of them are None
    if dicom_data is not None:
        if icontour_data is None:
            icontour_data = np.zeros(dicom_data.shape, dtype=np.bool_)
        if ocontour_data is None:
            ocontour_data = np.zeros(dicom_data.shape, dtype=np.bool_)
    else:
        if icontour_data is not None:
            dicom_data = np.zeors(icontour_data.shape, dtype=np.int16)
        elif ocontour_data is not None:
            dicom_data = np.zeors(ocontour_data.shape, dtype=np.int16)
    return (dicom_data, icontour_data, ocontour_data)


def _list_valid_files(directory):
    """List valid files in the given directory

    :param directory: path string to the directory under question
    :return: iterator of filenames
    """
    return (f for f in os.listdir(directory)
            if opath.exists(opath.join(directory, f)) and
            opath.isfile(opath.join(directory, f)))


class Record:
    """A class that holds the record of one patient.
    """

    def __init__(self, patient_id, original_id, serial_id, filenames):
        """Initialize with patient ID, original ID, and image-label data.

        :param patient_id: string with the patient's id
        :param original_id: string with the original id
        :param serial_id: integer index within one patient's record
        :param filenames: a 2-tuple that contains dicom filename and contour
                          filename
        """
        self.patient_id = patient_id
        self.original_id = original_id
        self.serial_id = serial_id
        self.filenames = filenames
        self._data = None

    def clear_data(self):
        """Free up space occupied by the data field.
        """
        self._data = None

    def load_data(self):
        """Load and parse data from disk.
        """
        self._data = _parse_dicom_and_contour_files(self.filenames)

    def has_dicom(self):
        """Check if there is DICOM image
        """
        return self.filenames[0] != ''

    def has_icontour(self):
        """Check if there is i-contour image
        """
        return self.filenames[1] != ''

    def has_ocontour(self):
        """Check if there is o-contour image
        """
        return self.filenames[2] != ''

    @property
    def data(self):
        """A property that when called will load and parse DICOM image file and
        contour file lazily, unless they are already loaded.
        """
        if self._data is None:
            self.load_data()
        return self._data


class DicomContourParser:
    """A class that parses a data folder containing DICOM and contour files.

    usage:
    batch_size = 100
    parser = DicomContourParser('/path/to/data/folder')
    for batch in parser.random_shuffled_iterator(batch_size):
        # do something with batch
        # batch is a list of (image, label) 2-tuples
    """

    ICONTOUR_PATTERN = re.compile(r'IM-\d{4}-(\d{4})-icontour.*.txt')
    OCONTOUR_PATTERN = re.compile(r'IM-\d{4}-(\d{4})-ocontour.*.txt')
    DICOM_PATTERN = re.compile(r'(\d+).dcm')

    def __init__(self, path_to_data, async_load=False):
        """Initialize using the path to the data folder.

        :param path_to_data: a string containing the path to the data folder
        :return: a DicomContourParser object

        """
        self.async_load = async_load
        self.loader_thread = None
        self.link_file = opath.join(path_to_data, 'link.csv')
        self.dicom_dir = opath.join(path_to_data, 'dicoms')
        self.contour_dir = opath.join(path_to_data, 'contourfiles')
        self.id_list = []
        self.record_list = []
        if not opath.exists(path_to_data) or\
           not opath.isdir(path_to_data) or\
           not opath.exists(self.link_file) or\
           not opath.isfile(self.link_file) or\
           not opath.exists(self.dicom_dir) or\
           not opath.isdir(self.dicom_dir) or\
           not opath.exists(self.contour_dir) or\
           not opath.isdir(self.contour_dir):
            raise InvalidDataFolder('Path {} is not a valid data folder'
                                    .format(path_to_data))
        with open(self.link_file, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            # skip the header
            next(reader)
            for pid, oid in reader:
                self.id_list.append((pid, oid))
        self._parse()

    def _get_valid_sids(self, dicom_dir, icontour_dir, ocontour_dir):
        """Scan the DICOM and contour directories to find the union of serial
        IDs.

        :param dicom_dir: path string of the DICOM data folder
        :param icontour_dir: path string of the i-contour data folder
        :param ocontour_dir: path string of the o-contour data folder
        :return: ascending ordered list of 4-tuples, elements of which contains
                 serial ID, DICOM filename, i-contour filename, and o-contour
                 filename
        """
        # list valid files in the directories, match to the corresponding regex
        # and extract the serial IDs with the corresponding filename
        dicom_ids = dict((int(match.group(1)), match.group(0)) for match in
                         map(lambda f: re.match(self.DICOM_PATTERN, f),
                             _list_valid_files(dicom_dir))
                         if match is not None)
        icontour_ids = dict((int(match.group(1)), match.group(0)) for match in
                            map(lambda f: re.match(self.ICONTOUR_PATTERN, f),
                                _list_valid_files(icontour_dir))
                            if match is not None)
        ocontour_ids = dict((int(match.group(1)), match.group(0)) for match in
                            map(lambda f: re.match(self.OCONTOUR_PATTERN, f),
                                _list_valid_files(ocontour_dir))
                            if match is not None)
        # take the union of the IDs, convert to list, and sort
        sids = sorted(list(set().union((key for key in dicom_ids),
                                       (key for key in icontour_ids),
                                       (key for key in ocontour_ids))))
        res = []
        for sid in sids:
            if sid in icontour_ids:
                icontour_filename = opath.join(icontour_dir, icontour_ids[sid])
            else:
                icontour_filename = ''
            if sid in ocontour_ids:
                ocontour_filename = opath.join(ocontour_dir, ocontour_ids[sid])
            else:
                ocontour_filename = ''
            if sid in dicom_ids:
                dicom_filename = opath.join(dicom_dir, dicom_ids[sid])
            else:
                dicom_filename = ''
            res.append((sid, dicom_filename, icontour_filename,
                        ocontour_filename))
        return res

    def _parse(self):
        """Parse the data folder to produce data records.

        :return: a list of data records (typed Record) organized by images
        """
        self.record_list = []
        for pid, oid in self.id_list:
            dicom_dir = opath.join(self.dicom_dir, pid)
            icontour_dir = opath.join(opath.join(self.contour_dir, oid),
                                      'i-contours')
            ocontour_dir = opath.join(opath.join(self.contour_dir, oid),
                                      'o-contours')
            # skip any id item whose dicom folder or contour folder is missing
            if not opath.exists(dicom_dir) or not opath.exists(icontour_dir)\
               or not opath.exists(ocontour_dir):
                continue
            sids = self._get_valid_sids(dicom_dir, icontour_dir, ocontour_dir)
            self.record_list.extend((Record(pid, oid, item[0], item[1:])
                                     for item in sids))

    def _prepare_batch_data(self, low, high):
        """Issue data loading on the records in [low, high)

        :param low: inclusive lower bound of indices
        :param high: exclusive higher bound of indices
        """
        low = max(low, 0)
        high = min(high, len(self.record_list))
        for i in range(low, high):
            self.record_list[i].load_data()

    def _async_prepare_batch_data(self, low, high):
        """Issue data loading on the records in [low, high) in a separate
        thread.

        :param low: inclusive lower bound of indices
        :param high: exclusive higher bound of indices
        """
        if self.loader_thread is not None:
            self.loader_thread.join()
        self.loader_thread = Thread(target=self._prepare_batch_data,
                                    args=(low, high))
        self.loader_thread.start()

    def _invalidate_batch_data(self, low, high):
        """Issue data discarding on the records in [low, high)

        :param low: inclusive lower bound of indices
        :param high: exclusive higher bound of indices
        """
        low = max(low, 0)
        high = min(high, len(self.record_list))
        for i in range(low, high):
            self.record_list[i].clear_data()

    def random_shuffled_iterator(self, batch_size=1, tag_records=False):
        """Get an iterator that randomly iterate through the dataset
        """
        if tag_records:
            map_func = lambda r: ('{}:{}:{:06d}'.format(
                r.patient_id, r.original_id, r.serial_id), r._data)
        else:
            map_func = lambda r: r._data
        random.shuffle(self.record_list)
        num_records = len(self.record_list)
        prepared_batch = None
        for low in range(0, len(self.record_list), batch_size):
            high = min(low + batch_size, num_records)
            if self.async_load:
                self._async_prepare_batch_data(low, high)
            else:
                self._prepare_batch_data(low, high)
            if prepared_batch is not None:
                l, h = prepared_batch
                yield list(map(map_func, self.record_list[l:h]))
                self._invalidate_batch_data(l, h)
            prepared_batch = (low, high)
        if self.async_load and self.loader_thread is not None:
            self.loader_thread.join()
        if prepared_batch is not None:
            l, h = prepared_batch
            yield list(map(map_func, self.record_list[l:h]))
            self._invalidate_batch_data(l, h)
