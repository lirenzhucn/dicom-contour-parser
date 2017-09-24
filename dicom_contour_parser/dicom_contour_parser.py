"""dicom_contour_parser.py

This module provides a class that can parse a data folder and produce the
entire data set as a list of records organized by patients and then slices.

author: Liren Zhu (liren.zhu.cn@gmail.com)
date: 2017-09-23

"""

import csv
import os
import os.path as opath
from collections import namedtuple


class InvalidDataFolder(Exception):
    pass


# :param patient_id: string with the patient's id
# :param original_id: string with the original id
# :param data: a list of dicom image-contour pairs [(i1, c1), (i2, c2), ...]
Record = namedtuple('Record', ['patient_id', 'original_id', 'data'])


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

    def parse(self):
        """Parse the data folder to produce data records.

        :return: a list of data records (typed Record) organized by patients

        """
        self.record_list = []
        for pid, oid in self.id_list:
            dicom_dir = opath.join(self.dicom_dir, pid)
            contour_dir = opath.join(opath.join(self.dicom_dir, oid),
                                     'i-contours')
            # skip any id item whose dicom folder or contour folder is missing
            if not opath.exists(dicom_dir) or not opath.exists(contour_dir):
                continue
            dicom_files = os.listdir(dicom_dir)
            contour_files = os.listdir(contour_dir)
