# Module for running CNN-BiLSTM vad model,
# may also be run directly as a script
# Author: Nick Wilkinson 2021
import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple
from tensorflow.keras import models
from voxseg import utils
from scipy.signal import medfilt

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


def decode(targets: pd.DataFrame, speech_thresh: float = 0.5, speech_w_music_thresh: float = 0.5, filt: int = 1) -> pd.DataFrame:
    '''Function for converting target sequences within a pd.DataFrame to endpoints.

    Args:
        targets: A pd.DataFrame containing predicted targets (in array form) and metadata.
        speech_thresh (optional): A decision threshold between 0 and 1 for the speech class, lower values
        result in more frames being classified as speech. (Default: 0.5)
        speech_w_music_thresh (optional):  A decision threshold between 0 and 1 for the speech_with_music class.
        Setting this threshold higher will filter out more music which may be desirable for ASR. (Default: 0.5)
        filt (optional): a kernel size for the median filter to apply to the output labels for smoothing. (Default: 1)

    Returns:
        A pd.DataFrame containing speech segment endpoints and metadata.
    '''

    targets = targets.copy()
    prior = np.array([(1-speech_thresh) * speech_w_music_thresh,
                    speech_thresh * speech_w_music_thresh,
                    (1-speech_thresh) * (1-speech_w_music_thresh),
                    (1-speech_thresh) * speech_w_music_thresh])
    temp = pd.concat([_targets_to_endpoints(medfilt([0 if (j*prior).argmax() == 1 else 1 for j in i], filt), 0.32) \
                     for i in targets['predicted-targets']], ignore_index=True)
    if 'start' in targets.columns:
        targets['end'] = targets['start'] + temp['end']
        targets['start'] = targets['start'] + temp['start']
    else:
        targets['start'] = temp['start']
        targets['end'] = temp['end']
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
        #temp = model.predict(utils.time_distribute(features, 15)[:,:,:,:,np.newaxis])
        temp = model.predict(features[np.newaxis,:,:,:,np.newaxis])
        targets.append(temp.reshape(-1, temp.shape[-1]))
    return pd.Series(targets)


def _targets_to_endpoints(targets: np.ndarray, frame_length: float) -> pd.DataFrame:
    '''Auxilory function used by decode() for converting a target sequence to endpoints.

    Args:
        targets: A binary np.ndarray of speech/nonspeech targets where 1 indicates the presence of speech.
        frame_length: The length of each target in seconds.

    Returns:
        A pd.DataFrame, containing the speech segment start and end boundaries in arrays.
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
    starts = np.around(np.array([i * frame_length for i in starts]), 3)
    ends = np.around(np.array([i * frame_length for i in ends]), 3)
    return pd.DataFrame({'start': [starts],'end': [ends]})


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
    parser = argparse.ArgumentParser(prog='run_cnnlstm.py',
                                     description='Run a trained voice activity detector on extracted feature set.')
    
    parser.add_argument('-s', '--speech_thresh', type=float,
                       help='a decision threshold value between (0,1) for speech vs non-speech, defaults to 0.5')

    parser.add_argument('-m', '--speech_w_music_thresh', type=float,
                       help='a decision threshold value between (0,1) for speech_with_music vs non-speech, defaults to 0.5, \
                       increasing will remove more speech_with_music, useful for downsteam ASR')
    
    parser.add_argument('-f', '--median_filter_kernel', type=int,
                       help='a kernel size for a median filter to smooth the output labels, defaults to 1 (no smoothing)')

    parser.add_argument('-M', '--model_path', type=str,
                        help='a path to a trained vad model saved as in .h5 format, overrides default pretrained model')

    parser.add_argument('feat_dir', type=str,
                        help='a path to a directory containing a feats.h5 file with extracted features')
    
    parser.add_argument('out_dir', type=str,
                        help='a path to an output directory where the output segments will be saved')

    args = parser.parse_args()
    if args.speech_thresh is not None:
        speech_thresh = args.speech_thresh
    else:
        speech_thresh = 0.5
    if args.speech_w_music_thresh is not None:
        speech_w_music_thresh = args.speech_w_music_thresh 
    else:
        speech_w_music_thresh = 0.5
    if args.median_filter_kernel is not None:
        filt = args.median_filter_kernel 
    else:
        filt = 1
    feats = pd.read_hdf(f'{args.feat_dir}/feats.h5')
    if args.model_path is not None:
        model = models.load_model(args.model_path)
    else:
        model = models.load_model(f'{os.path.dirname(os.path.realpath(__file__))}/models/cnn_bilstm.h5')
    targets = predict_targets(model, feats)
    endpoints = decode(targets, speech_thresh, speech_w_music_thresh, filt)
    to_data_dir(endpoints, args.out_dir)
