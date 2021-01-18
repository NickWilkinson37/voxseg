# Various utility functions for sub-tasks frequently used by the voxseg module
# Author: Nick Wilkinson 2021
import pandas as pd
import os
import sys
from typing import Iterable, TextIO, Tuple

def read_data_file(path: str) -> pd.DataFrame:
    '''Function for reading standard Kaldi-style text data files (eg. wav.scp, utt2spk etc.)

    Args:
        path: The path to the data file.

    Returns:
        A pd.DataFrame containing the enteries in the data file.
    
    Example:
        Given a file 'data/utt2spk' with the following contents:
        utt0    spk0
        utt1    spk1
        utt1    spk2

        Running the function yeilds:
        >>> print(read_data_file('data/utt2spk'))
                0       1
        0    utt0    spk0
        1    utt1    spk1
        2    utt2    spk2
    
    '''

    with open(path, 'r') as f:
        return pd.DataFrame([i.split() for i in f.readlines()])


def process_data_dir(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''Function for processing Kaldi-style data directory containing wav.scp,
    segments (optional), and utt2spk (optional).

    Args:
        path: The path to the data directory.

    Returns:
        A tuple of pd.DataFrame in the format (wav_scp, segments, utt2spk), where
        pd.DataFrame contain data from the original files -- see docs for read_data_file().
        If a file is missing a null value is returned eg. a directory without utt2spk would
        return:
        (wav_scp, segments, None)

    Raises:
        FileNotFoundError: If wav.scp is not found.
    '''

    files = [f for f in os.listdir(path) if os.path.isfile(f'{path}/{f}')]
    try:
        wav_scp = read_data_file(f'{path}/wav.scp')
        wav_scp.columns = ['recording-id', 'extended filename']
    except FileNotFoundError:
        print('ERROR: Data directory needs to contain wav.scp file to be processed.')
        raise
    if 'segments' not in files:
        segments = None
    else:
        segments = read_data_file(f'{path}/segments')
        segments.columns = ['utterance-id', 'recording-id', 'start', 'end']
    if 'utt2spk' not in files:
        utt2spk = None
    else:
        utt2spk = read_data_file(f'{path}/utt2spk')
        utt2spk.columns = ['utterance-id', 'speaker-id']
    return wav_scp, segments, utt2spk


def progressbar(it: Iterable, prefix: str = "", size: int = 45, file: TextIO = sys.stdout) -> None:
    '''Provides a progress bar for an iterated process.

    Args:
        it: An Iterable type for which to provide a progess bar.
        prefix (optional): A string to print before the progress bar. Defaults to empty string.
        size (optional): The number of '#' characters to makeup the progressbar. Defaults to 45.
        file (optional): A text file type for output. Defaults to stdout.

    Code written by eusoubrasileiro, found at:
    https://stackoverflow.com/questions/3160699/python-progress-bar/34482761#34482761
    '''

    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()