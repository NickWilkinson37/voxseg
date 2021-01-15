# Script for extracting log-mel spectrogram features
# Author: Nick Wilkinson 2021
import argparse
import numpy as np
import utils
from typing import Dict
from scipy.io import wavfile
from python_speech_features import logfbank


def extract(signals: Dict[str, np.ndarray], frame_length: float = 0.32, nfilt: int = 32, rate: int = 16000) -> Dict[str, np.ndarray]:
    '''Function for extracting log-mel filterbank spectrogram features.

    Args:
        signals: A dict of np.ndarrays constaining audio signals, with <utterance-id> as key.
            See docs for prep_data().
        frame_length (optional): Length of a spectrogram feature in seconds. Default is 0.32.
        nfilt (optional): Number of filterbanks to use. Default is 32.
        rate (optional): Sample rate. Default is 16k.

    Returns:
        A dict of np.ndarrays constaining spectrogram features, with <utterance-id> as key.
    '''

    features = {}
    for i in signals:
        feats = []
        try:
            assert len(range(0, int(len(signals[i])-1 - (frame_length+0.01) * rate), int(frame_length * rate))) > 0
            for j in utils.progressbar(range(0, int(len(signals[i])-1 - (frame_length+0.01) * rate), int(frame_length * rate)), i):
                feats.append(np.flipud(logfbank(signals[i][j:int(j + (frame_length+0.01) * rate)], rate, nfilt=nfilt).T))
            feats = np.array(feats)
            features[i] = feats
        except AssertionError:
            print(f'WARNING: {i} is too short to extract features, will be ignored.')
    return features


def normalize(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    pass


def prep_data(data_dir: str) -> Dict[str, np.ndarray]:
    '''Function for extracting signal arrays given a Kaldi-style data directory.

    Args:
        data_dir: The path to the data directory.

    Returns:
        A dict of np.ndarrays constaining audio signals, with <utterance-id> as key. For example:
        {utterance_000 : [53, 21, -7, ... , -32, -17, 5],
        ...
        utterance_099 : [42, 32, 56, ... , 13, -2, -10]}

    Raises:
        AssertionError: If a wav file is not 16k mono.
    '''

    wav_scp, segments, _  = utils.process_data_dir(data_dir)

    # read wavs, raise AssertionError if not 16k mono
    wavs = {}
    for _, i in wav_scp.iterrows():
        rate, wavs[i[0]] = wavfile.read(i[1])
        assert rate == 16000 and wavs[i[0]].ndim == 1, f'{i[1]} is not formatted in 16k mono.'

    # check for segments file and process if found
    if segments is None:
        print('WARNING: Segments file not found, features will be extracted across entire audio files.')
        return wavs
    else:
        segs = {}
        for _, i in segments.iterrows():
            start = int(float(i[2]) * rate)
            end = int(float(i[3]) * rate)
            segs[i[0]] = wavs[i[1]][start:end]
        return segs


def read_sigs(row: pd.DataFrame) -> np.ndarray:
    '''Function for reading an audio signal from a row of a pd.DataFrame containing dataset information.

    Args:
        row: A row of a pd.DataFrame created by prep_data().

    Returns:
        A np.ndarray of the audio signal.
    '''
    
    if 'utterance-id' in row:
        rate, sig = wavfile.read(row['extended filename'])
        return sig[int(float(row['start']) * rate): int(float(row['end']) * rate)]
    else:
        rate, sig = wavfile.read(row['extended filename'])
        return sig

# Handle args when run directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='extract_feats',
                                     description='Extract log-mel spectrogram features.')

    parser.add_argument('data_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', and optionally \'segments\'')

    args = parser.parse_args()
    signals = prep_data(args.data_dir)
    features = extract(signals)
    print(features)