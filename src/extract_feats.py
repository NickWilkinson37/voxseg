# Script for extracting log-mel spectrogram features
# Author: Nick Wilkinson 2021
import argparse
import numpy as np
import pandas as pd
import utils
from typing import Dict
from scipy.io import wavfile
from python_speech_features import logfbank


def calulate_feats(row: pd.DataFrame, frame_length: float, nfilt: int, rate: int) -> np.ndarray:
    '''Auxiliary function used by extract(). Extracts log-mel spectrograms from a row of a pd.DataFrame
    containing dataset information created by prep_data().

    Args:
        row: A row of a pd.DataFrame created by prep_data().
        frame_length: Length of a spectrogram feature in seconds.
        nfilt: Number of filterbanks to use.
        rate: Sample rate.

    Returns:
        An np.ndarray of features.
    '''

    sig = row['signal']
    if 'utterance-id' in row:
        id = row['utterance-id']
    else:
        id = row['recording-id']
    try:
        assert len(range(0, int(len(sig)-1 - (frame_length+0.01) * rate), int(frame_length * rate))) > 0
        feats = []
        for j in utils.progressbar(range(0, int(len(sig)-1 - (frame_length+0.01) * rate), int(frame_length * rate)), id):
            feats.append(np.flipud(logfbank(sig[j:int(j + (frame_length+0.01) * rate)], rate, nfilt=nfilt).T))
        return np.array(feats)
    except AssertionError:
        print(f'WARNING: {id} is too short to extract features, will be ignored.')


def extract(data: pd.DataFrame, frame_length: float = 0.32, nfilt: int = 32, rate: int = 16000) -> pd.DataFrame:
    '''Function for extracting log-mel filterbank spectrogram features.

    Args:
        data: A pd.DataFrame containing datatset information and signals -- see docs for prep_data().
        frame_length (optional): Length of a spectrogram feature in seconds. Default is 0.32.
        nfilt (optional): Number of filterbanks to use. Default is 32.
        rate (optional): Sample rate. Default is 16k.

    Returns:
        A pd.DataFrame containing features and metadata.
    '''
    
    data['features'] = data.apply(lambda x: calulate_feats(x, frame_length, nfilt, rate), axis=1)
    return data


def normalize(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    pass


def prep_data(data_dir: str) -> pd.DataFrame:
    '''Function for creating pd.DataFrame containing dataset specified by Kaldi-style data directory.

    Args:
        data_dir: The path to the data directory.

    Returns:
        A pd.DataFrame of dataset information. For example:

            recording-id  extended filename        utterance-id  start  end  signal
        0   rec_00        ~/Documents/test_00.wav  utt_00        10     20   [-49, -43, -35...
        1   rec_00        ~/Documents/test_00.wav  utt_01        50     60   [-35, -23, -12...
        2   rec_01        ~/Documents/test_01.wav  utt_02        135    163  [25, 32, 54...

        Note that 'utterance-id', 'start' and 'end' are optional, will only appear if data directory
        contains 'segments' file.

    Raises:
        AssertionError: If a wav file is not 16k mono.
    '''

    wav_scp, segments, _  = utils.process_data_dir(data_dir)

    # check for segments file and process if found
    if segments is None:
        print('WARNING: Segments file not found, entire audio files will be processed.')
        wav_scp['signal'] = wav_scp.apply(lambda x: read_sig(x), axis=1)
        return wav_scp
    else:
        data = wav_scp.merge(segments)
        data['signal'] = data.apply(lambda x: read_sig(x), axis=1)
        return data


def read_sig(row: pd.DataFrame) -> np.ndarray:
    '''Auxiliary function used by prep_data(). Reads an audio signal from a row of a pd.DataFrame
    containing the directory of a .wav file, and optionally start and end points within the .wav. 

    Args:
        row: A row of a pd.DataFrame created by prep_data().

    Returns:
        An np.ndarray of the audio signal.

    Raises:
        AssertionError: If a wav file is not 16k mono.
    '''
    
    filename = row['extended filename']
    rate, sig = wavfile.read(filename)
    assert rate == 16000 and sig.ndim == 1, f'{filename} is not formatted in 16k mono.'
    if 'utterance-id' in row:
        return sig[int(float(row['start']) * rate): int(float(row['end']) * rate)]
    else:
        return sig

# Handle args when run directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='extract_feats',
                                     description='Extract log-mel spectrogram features.')

    parser.add_argument('data_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', and optionally \'segments\'')

    args = parser.parse_args()
    data = prep_data(args.data_dir)
    features = extract(data)
    print(features)