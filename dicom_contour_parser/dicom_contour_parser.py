"""dicom_contour_parser.py

This module provides a class that can parse a data folder and produce the
entire data set as a list of records organized by patients and then slices.

author: Liren Zhu (liren.zhu.cn@gmail.com)
date: 2017-09-23

"""

import csv
import os
import re
import os.path as opath

from . import parsing


class InvalidDataFolder(Exception):
    pass


class Record:
    """A class that holds the record of one patient.
    """

    def __init__(self, patient_id, original_id, data):
        """ Initialize with patient ID, original ID, and image-label data.

        :param patient_id: string with the patient's id
        :param original_id: string with the original id
        :param data: a list of dicom image-contour pairs
                     [(i1, c1), (i2, c2), ...]
        """
        self.patient_id = patient_id
        self.original_id = original_id
        self.data = data


def _list_valid_files(directory):
    """List valid files in the given directory

    :param directory: path string to the directory under question
    :return: iterator of filenames
    """
    return (f for f in os.listdir(directory)
            if opath.exists(opath.join(directory, f)) and
            opath.isfile(opath.join(directory, f)))


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
           not opath.isfile(self.link_file):
            raise InvalidDataFolder('Path {} is not a valid data folder'
                                    .format(path_to_data))
        with open(self.link_file, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            # skip the header
            next(reader)
            for pid, oid in reader:
                self.id_list.append((pid, oid))

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

    @staticmethod
    def _parse_dicom_and_contour_files(sid_record):
        """Convert two filenames to valid image data

        :param sid_record: a 3-tuple record of sid, dicom_filename,
               and contour_filename, where:
               sid is the serial ID of the record
               dicom_filename is the path string to the DICOM file
               contour_filename is the path string to the contour file
        :return: 2-tuple containing the DICOM image data and contour mask data
        """
        _, dicom_filename, contour_filename = sid_record
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
        return (dicom_data, contour_data)

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
            self.record_list.extend((
                Record(pid, oid, self._parse_dicom_and_contour_files(item))
                for item in sids))
        return self.record_list
