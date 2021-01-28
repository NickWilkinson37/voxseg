# Various utility functions for sub-tasks frequently used by the voxseg module
# Author: Nick Wilkinson 2021
import pickle
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd
import numpy as np
import os
import sys
from scipy.io import wavfile
from typing import Iterable, TextIO, Tuple
import warnings


def load(path: str) -> pd.DataFrame:
    '''Reads a pd.DataFrame from a .h5 file.

    Args:
        path: The filepath of the .h5 file to be read.

    Returns:
        A pd.DataFrame of the data loaded from the .h5 file
    '''

    return pd.read_hdf(path)


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
        segments[['start', 'end']] = segments[['start', 'end']].astype(float)
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
        return pd.DataFrame([i.split() for i in f.readlines()], dtype=str)


def read_sigs(data: pd.DataFrame) -> pd.DataFrame:
    '''Reads audio signals from a pd.DataFrame containing the directories of
    .wav files, and optionally start and end points within the .wav files. 

    Args:
        data: A pd.DataFrame created by prep_data().

    Returns:
        A pd.DataFrame with columns 'recording-id' and 'signal', or if segments were provided
        'utterance-id' and 'signal'. The 'signal' column contains audio as np.ndarrays.

    Raises:
        AssertionError: If a wav file is not 16k mono.
    '''

    wavs = {}
    ret = []
    for i, j in zip(data['recording-id'].unique(), data['extended filename'].unique()):
        rate, wavs[i] = wavfile.read(j)
        assert rate == 16000 and wavs[i].ndim == 1, f'{j} is not formatted in 16k mono.'
    if 'utterance-id' in data:
        for _, row in data.iterrows():
            ret.append([row['utterance-id'], wavs[row['recording-id']][int(float(row['start']) * rate): int(float(row['end']) * rate)]])
    else:
        for _, row in data.iterrows():
            ret.append([row['recording-id'], wavs[row['recording-id']]])
    return pd.DataFrame(ret, columns=['utterance-id', 'signal'])


def save(data: pd.DataFrame, path: str) -> None:
    '''Saves a pd.DataFrame to a .h5 file.

    Args:
        data: A pd.DataFrame for saving.
        path: The filepath where the pd.DataFrame should be saved.
    '''

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    data.to_hdf(path, key='data', mode='w')


def time_distribute(data: np.ndarray, sequence_length: int, stride: int = None, z_pad: bool = True) -> np.ndarray:
    '''Takes a sequence of features or labels and creates an np.ndarray of time
    distributed sequences for input to a Keras TimeDistributed() layer.

    Args:
        data: The array to be time distributed.
        sequence_length: The length of the output sequences in samples.
        stride (optional): The number of samples between sequences. Defaults to sequence_length.
        z_pad (optional): Zero padding to ensure all sequences to have the same dimensions.
        Defaults to True.

    Returns:
        The time ditributed data sequences.

    Example:
        Given an np.ndarray of data:
        >>> data.shape
        (10000, 32, 32, 1)
        >>> time_distribute(data, 10).shape
        (1000, 10, 32, 32, 1)
        The function yeilds 1000 training sequences, each 10 samples long.
    '''

    if stride is None:
        stride = sequence_length
    if stride > sequence_length:
        print('WARNING: Stride longer than sequence length, causing missed samples. This is not recommended.')
    td_data = []
    for n in range(0, len(data)-sequence_length+1, stride):
        td_data.append(data[n:n+sequence_length])
    if z_pad:
        if len(td_data)*stride+sequence_length != len(data)+stride:
            z_needed = len(td_data)*stride+sequence_length - len(data)
            z_padded = np.zeros(td_data[0].shape)
            for i in range(sequence_length-z_needed):
                z_padded[i] = data[-(sequence_length-z_needed)+i]
            td_data.append(z_padded)
    return np.array(td_data)