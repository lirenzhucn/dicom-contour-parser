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

from . import parsing


class InvalidDataFolder(Exception):
    pass


def _parse_dicom_and_contour_files(filenames):
    """Convert two filenames to valid image data

    :param filenames: a 2-tuple record of dicom_filename and contour_filename:
           dicom_filename is the path string to the DICOM file
           contour_filename is the path string to the contour file
    :return: 2-tuple containing the DICOM image data and contour mask data
    """
    dicom_filename, contour_filename = filenames
    dicom_data, contour_data = None, None
    if dicom_filename:
        dicom_data = parsing.parse_dicom_file(dicom_filename)
        if dicom_data is not None:
            dicom_data = dicom_data['pixel_data']
    if contour_filename:
        contour_path = parsing.parse_contour_file(contour_filename)
        if dicom_data is not None:
            height, width = dicom_data.shape
        else:
            max_x, max_y = 0, 0
            for x, y in contour_path:
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            height = round(max_x + 1)
            width = round(max_y + 1)
        contour_data = parsing.poly_to_mask(contour_path, width, height)
    # TODO: fix the case in which both of them are None
    if contour_data is None and dicom_data is not None:
        contour_data = np.zeros(dicom_data.shape, dtype=np.bool_)
    elif contour_data is not None and dicom_data is None:
        dicom_data = np.zeors(contour_data.shape, dtype=np.int16)
    return (dicom_data, contour_data)


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
    parser = DicomContourParser('/path/to/data/folder')
    records = parser.parse()
    records[0].patient_id
    records[0].original_id
    for image, label in records[0].data:
        # do something with image and label

    """

    CONTOUR_PATTERN = re.compile(r'IM-\d{4}-(\d{4})-icontour.*.txt')
    DICOM_PATTERN = re.compile(r'(\d+).dcm')

    def __init__(self, path_to_data):
        """Initialize using the path to the data folder.

        :param path_to_data: a string containing the path to the data folder
        :return: a DicomContourParser object

        """
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
        self.parse()

    def _get_valid_sids(self, dicom_dir, contour_dir):
        """Scan the DICOM and contour directories to find the union of serial
        IDs.

        :param dicom_dir: path string of the DICOM data folder
        :param contour_dir: path string of the contour data folder
        :return: ascending ordered list of 3-tuples, elements of which contains
                 serial ID, DICOM filename, and contour filename
        """
        # list valid files in the directories, match to the corresponding regex
        # and extract the serial IDs with the corresponding filename
        dicom_ids = dict((int(match.group(1)), match.group(0)) for match in
                         map(lambda f: re.match(self.DICOM_PATTERN, f),
                             _list_valid_files(dicom_dir))
                         if match is not None)
        contour_ids = dict((int(match.group(1)), match.group(0)) for match in
                           map(lambda f: re.match(self.CONTOUR_PATTERN, f),
                               _list_valid_files(contour_dir))
                           if match is not None)
        # take the union of the IDs, convert to list, and sort
        sids = sorted(list(set().union((key for key in dicom_ids),
                                       (key for key in contour_ids))))
        res = []
        for sid in sids:
            if sid in contour_ids:
                contour_filename = opath.join(contour_dir, contour_ids[sid])
            else:
                contour_filename = ''
            if sid in dicom_ids:
                dicom_filename = opath.join(dicom_dir, dicom_ids[sid])
            else:
                dicom_filename = ''
            res.append((sid, dicom_filename, contour_filename))
        return res

    def parse(self):
        """Parse the data folder to produce data records.

        :return: a list of data records (typed Record) organized by images
        """
        self.record_list = []
        for pid, oid in self.id_list:
            dicom_dir = opath.join(self.dicom_dir, pid)
            contour_dir = opath.join(opath.join(self.contour_dir, oid),
                                     'i-contours')
            # skip any id item whose dicom folder or contour folder is missing
            if not opath.exists(dicom_dir) or not opath.exists(contour_dir):
                continue
            sids = self._get_valid_sids(dicom_dir, contour_dir)
            self.record_list.extend((Record(pid, oid, item[0], item[1:])
                                     for item in sids))
        return self.record_list

    def _prepare_batch_data(self, low, high):
        """Issue data loading on the records in [low, high)

        :param low: inclusive lower bound of indices
        :param high: exclusive higher bound of indices
        """
        low = max(low, 0)
        high = min(high, len(self.record_list))
        for i in range(low, high):
            self.record_list[i].load_data()

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
            self._prepare_batch_data(low, high)
            if prepared_batch is not None:
                l, h = prepared_batch
                yield list(map(map_func, self.record_list[l:h]))
                self._invalidate_batch_data(l, h)
            prepared_batch = (low, high)
        if prepared_batch is not None:
            l, h = prepared_batch
            yield list(map(map_func, self.record_list[l:h]))
            self._invalidate_batch_data(l, h)
