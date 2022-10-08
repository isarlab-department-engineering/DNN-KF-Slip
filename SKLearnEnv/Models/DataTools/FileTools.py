import os
import csv
import numpy as np


class CsvFile:

    def __init__(self, path: str,  filename: str):

        if not os.path.isdir(path):
            print('ERROR: folder %s not found!' % path)
            raise OSError
        self.path = path
        self.filename = filename
        self.full_path = os.path.join(self.path, ''.join([self.filename, '.csv']))
        self._create_csv()

    def get_path(self) -> str:
        return self.path

    def get_filename(self) -> str:
        return self.filename

    def get_full_path(self) -> str:
        return self.full_path

    def get_number_of_rows(self) -> int:
        try:
            with open(self.full_path, newline='') as csvFile:
                reader = csv.reader(csvFile)
                row_count = sum(1 for row in reader)
        except csv.Error:
            print("ERROR: unable to read the rows of the file %s" % self.path)
            exit(0)
        return row_count

    def get_number_of_columns(self) -> int:
        try:
            with open(self.full_path, newline='') as csvFile:
                reader = csv.reader(csvFile)
                col_count = len(next(reader))
        except csv.Error:
            print("ERROR: unable to read the columns of the file %s" % self.path)
            exit(0)
        return col_count

    def append_row_to_csv(self, new_row: np.array):
        try:
            with open(self.full_path, 'a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(new_row)
                csvFile.close()
        except csv.Error:
            print("ERROR: unable to append row to the file %s" % self.path)
            exit(0)

    def _create_csv(self):
        full_path = os.path.join(self.path, ''.join([self.filename, '.csv']))

        if os.path.isdir(full_path):
            print('WARNING: FILE EXISTS! Change the filename and retry -> Stop')
            exit(0)
        try:
            with open(full_path, 'w', newline='') as writeFile:
                csv.writer(writeFile, delimiter=',',
                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writeFile.close()
        except csv.Error:
            print("ERROR:Creation of the file.csv %s failed" % full_path)
            exit(0)
        else:
            print("Successfully created the file %s " % full_path)


class FileTxt:

    def __init__(self, path: str, filename: str):
        if not os.path.isdir(path):
            print('ERROR: folder %s not found!' % path)
            raise OSError
        self.path = path
        self.filename = filename
        self.full_path = os.path.join(self.path, ''.join([self.filename, '.txt']))
        file = open(self.full_path, "w+")
        file.close()

    def append_row_to_txt(self, string: str):
        with open(self.full_path, 'a') as file:
            file.write('%s' % string)
        file.close()


class Folder:

    def __init__(self, path: str, foldername: str):
        if not os.path.isdir(path):
            print('ERROR: folder %s not found!' % path)
            raise OSError
        self.directory = os.path.join(path, foldername)

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
            print("Directory ",  self.directory, " Created ")
        else:
            print("Directory ",  self.directory, " already exists")
            raise OSError

    def get_path(self):
        return self.directory
