# Script for running CNN-BiLSTM vad model
# Author: Nick Wilkinson 2021
import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple
from tensorflow.keras import models
from voxseg import utils

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU, quick enough for decoding
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10,inter_op_parallelism_threads=10)
sess = tf.compat.v1.Session(config=session_conf)


def decode(targets: pd.DataFrame) -> pd.DataFrame:
    '''Function for converting target sequences within a pd.DataFrame to endpoints.

    Args:
        targets: A pd.DataFrame containing predicted targets (in array form) and metadata.

    Returns:
        A pd.DataFrame containing speech segment endpoints and metadata.
    '''

    temp = np.array([_targets_to_endpoints(i[:,1] < 0.5, 0.32) for i in targets['predicted-targets']], dtype=object)
    if 'start' in targets.columns:
        targets['end'] = targets['start'] + temp[:,1]
        targets['start'] = targets['start'] + temp[:,0]
    else:
        targets['start'] = temp[:,0]
        targets['end'] = temp[:,1]
    targets = targets.drop(['predicted-targets'], axis=1)
    targets = targets.apply(pd.Series.explode).reset_index(drop=True)
    targets['utterance-id'] = targets['recording-id'].astype(str) + '_' + \
                        ((targets['start'] * 100).astype(int)).astype(str).str.zfill(7) + '_' + \
                        ((targets['end'] * 100).astype(int)).astype(str).str.zfill(7)
    return targets


def predict_targets(model: tf.keras.Model, features: pd.DataFrame) -> pd.DataFrame:
    '''Function for applying a pretrained model to predict targets from features.

    Args:
        model: A pretrained tf.keras model.
        features: A pd.DataFrame containing features and metadata.

    Returns:
        A pd.DataFrame containing predicted targets and metadata. 
    '''

    targets = features.drop(['normalized-features'], axis=1)
    print('------------------- Running VAD -------------------')
    targets['predicted-targets'] = _predict(model, features['normalized-features'])
    return targets
    

def to_data_dir(endpoints: pd.DataFrame, out_dir: str) -> None:
    '''A function for generating a Kaldi-style data directory output of the dicovered speech segments.
    
    Args:
        endpoints: A pd.DataFrame containing speech segment endpoints and metadata.
        out_dir: A path to an output directory where data files will be placed.
    '''

    if not os.path.exists(out_dir):
        print(f'Directory {out_dir} does not exist, creating it.')
        os.mkdir(out_dir)
    endpoints[['recording-id', 'extended filename']].drop_duplicates().to_csv(
                    f'{out_dir}/wav.scp',sep=' ', index=False, header=False)
    pd.concat([endpoints[['utterance-id', 'recording-id']], endpoints[['start', 'end']].astype(float).round(3)],
                    axis=1).to_csv(f'{out_dir}/segments', sep=' ', index=False, header=False)


def _predict(model: tf.keras.Model, col: pd.Series) -> pd.Series:
    '''Auxiliary function used by predict_targets(). Applies a pretrained model to 
    each feature set in the 'normalized-features' or 'features' column of a pd.DataFrame
    containing features and metadata.

    Args:
        model: A pretrained tf.keras model.
        col: A column of a pd.DataFrame containing features.

    Returns:
        A pd.Series containing the predicted target sequences. 
    '''

    targets = []
    for features in col:
        temp = model.predict(utils.time_distribute(features, 15)[:,:,:,:,np.newaxis])
        targets.append(temp.reshape(-1, temp.shape[-1]))
    return pd.Series(targets)


def _targets_to_endpoints(targets: np.ndarray, frame_length: float) -> Tuple[np.ndarray, np.ndarray]:
    '''Auxilory function used by decode() for converting a target sequence to endpoints.

    Args:
        targets: A binary np.ndarray of speech/nonspeech targets where 1 indicates the presence of speech.
        frame_length: The length of each target in seconds.

    Returns:
        Two np.ndarrays, the first containing speech segment start boundaries, the second containing
        speech segment end boundaries.
    '''
    
    starts = []
    ends = []
    state = 0
    for n, i in enumerate(targets):
        state, emmision = _update_fst(state, i)
        if emmision == 'start':
            starts.append(n)
        elif emmision == 'end':
            ends.append(n)
    state, emmision = _update_fst(state, None)
    if emmision == 'start':
        starts.append(n)
    elif emmision == 'end':
        ends.append(n + 1)
    starts = np.array([i * frame_length for i in starts])
    ends = np.array([i * frame_length for i in ends])
    return starts, ends


def _update_fst(state: int, transition: int) -> Tuple[int, str]:
    '''Auxiliary function used by _targets_to_endpoints() for updating finite state
    transducer.

    Args:
        state: The current state.
        transition: The input (the next binary target).

    Returns:
        A tuple consisting of the new state and the output ('start', 'end' or None,
        representing a start, end or no endpoint detections respectively).
    '''

    if state == 0:
        if transition == 0:
            state = 1
            return state, None
        elif transition == 1:
            state = 2
            return state, 'start'
    elif state == 1:
        if transition == 0:
            return state, None
        elif transition == 1:
            state = 2
            return state, 'start'
        elif transition is None:
            state = 3
            return state, None
    elif state == 2:
        if transition == 0:
            state = 1
            return state, 'end'
        elif transition == 1:
            return state, None
        elif transition is None:
            state = 3
            return state, 'end'


# Handle args when run directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='run_vad',
                                     description='Run a trained voice activity detector on extracted feature set.')

    parser.add_argument('model_path', type=str,
                        help='a path to a trained vad model saved as in .h5 format')

    parser.add_argument('feat_dir', type=str,
                        help='a path to a directory containing a feats.h5 file with extracted features')
    
    parser.add_argument('out_dir', type=str,
                        help='a path to an output directory where the output segments will be saved')

    args = parser.parse_args()
    feats = pd.read_hdf(args.feat_dir)
    model = models.load_model(args.model_path)
    targets = predict_targets(model, feats)
    endpoints = decode(targets)
    to_data_dir(endpoints, args.out_dir)
