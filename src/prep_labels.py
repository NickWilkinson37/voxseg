# Script for preparing training labels
# Author: Nick Wilkinson 2021
import argparse
import numpy as np
import pandas as pd
import os
import utils


def get_labels(data: pd.DataFrame) -> pd.DataFrame:
    data['labels'] = data.apply(lambda x: _generate_label_sequence(x), axis=1)
    data = data.drop(['signal', 'label'], axis=1)
    return data


def prep_data(path: str) -> pd.DataFrame:
    wav_scp, segments, utt2spk = utils.process_data_dir(path)
    assert utt2spk is not None and segments is not None, \
        'ERROR: Data directory needs to contain \'segments\' and \'utt2spk\'\
            containing label information.'
    data = wav_scp.merge(segments).merge(utt2spk)
    data = data.rename(columns={"speaker-id": "label"})
    data['signal'] = data.apply(lambda x: utils.read_sig(x), axis=1)
    return data


def _generate_label_sequence(row: pd.DataFrame, frame_length: float = 0.32, nfilt: int = 32, rate: int = 16000) -> pd.DataFrame:
    sig = row['signal']
    if 'utterance-id' in row:
        id = row['utterance-id']
    else:
        id = row['recording-id']
    try:
        assert len(range(0, int(len(sig)-1 - (frame_length+0.01) * rate), int(frame_length * rate))) > 0
        labels = []
        for _ in utils.progressbar(range(0, int(len(sig)-1 - (frame_length+0.01) * rate), int(frame_length * rate)), id):
            labels.append(row['label'])
        return np.array(labels)
    except AssertionError:
        pass

# Handle args when run directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='prep_labels',
                                     description='Prepare labels for model training.')

    parser.add_argument('data_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', \'segments\', and \'utt2spk\'')
    
    parser.add_argument('out_dir', type=str,
                        help='a path to an output directory where labels and metadata will be saved as labels.h5')

    args = parser.parse_args()
    data = prep_data(args.data_dir)
    labels = get_labels(data)
    if not os.path.exists(args.out_dir):
        print(f'Directory {args.out_dir} does not exist, creating it.')
        os.mkdir(args.out_dir)
    utils.save(labels, f'{args.out_dir}/labels.h5')