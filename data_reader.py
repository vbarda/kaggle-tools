import logging
import platform
import subprocess

from xlrd import XLRDError
from zipfile import ZipFile


logger = logging.getLogger(__name__)


def _get_zip_file_sizes(zip_file):
    '''list all the files in the zip archive and their sizes'''
    with ZipFile(zip_file) as f:
        file_names = f.namelist()
        sizes = {}
        for file_name in file_names:
            file_info = f.getinfo(file_name)
            sizes[file_name] = file_info.file_size
    return sizes


def _read_zip(zip_file, reader):
    '''Read zip file into pandas (since sometimes pd.read_csv
    can't handle multiple files in the archive)

    Args:
        zip_file: path to zip file
        reader: pandas reader function
            pd.read_xlsx | pd.read_csv
    '''
    sizes = _get_zip_file_sizes(zip_file)
    # assumption is that the dataset is the biggest file
    # gets the filename of the biggest file (key)
    biggest_file_name = max(sizes)
    with ZipFile(zip_file) as f:
        _f = f.open(biggest_file_name)
        df = reader(_f)
    return df


def _infer_reader(file_name):
    '''infer pandas reader from the file name'''
    # TODO: make it more robust
    logger.info('Inferring the reader from the file name')
    if 'xlsx' in file_name:
        reader = pd.read_excel
    elif 'csv' in file_name:
        reader = pd.read_csv
    else:
        raise AssertionError('Cannot infer the the reader from file name')
    return reader


# PUBLIC

def read_zip(zip_file):
    '''Try reading using pandas, and fall back on a custom reader if it fails'''
    reader = _infer_reader(zip_file)
    try:
        logger.info('Trying to read using pandas')
        df = reader(zip_file)
    except (ValueError, XLRDError):
        logger.info('Pandas reader failed, falling back on the custom zip reader')
        df = _read_zip(zip_file, reader)
    return df


def get_n_rows(file_name):
    '''Get row count'''
    _system = platform.system()

    # use filename as default string separator
    if _system == 'Darwin':
        sep = file_name
        base_command = 'wc -l {}'

    elif _system == 'Windows':
        sep = file_name.upper()
        base_command = 'find /c /v "" {}'

    else:
        raise ValueError('System "{}" is not supported'.format(_system))

    full_command = base_command.format(file_name)
    command_output = subprocess.Popen(full_command,
                                      shell=True,
                                      stdout=subprocess.PIPE).stdout.read()
    clean_output = ''.join(command_output.split(sep))
    digit_output = filter(lambda x: x.isdigit(), clean_output)
    return int(digit_output)
