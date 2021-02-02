# Module for evaluating performance of vad models,
# may also be run directly as a script
# Author: Nick Wilkinson 2021
import argparse
import numpy as np
import pandas as pd
from tinytag import TinyTag
from typing import Dict
from voxseg import utils


def print_confusion_matrix(scores: Dict[str,Dict[str,int]]) -> None:
    '''Prints the normalized confusion matrix.

    Args:
        scores: A dictionary of dictionaries containing TP, FP, FN and TN counts generated by score().
    '''

    tp, fp, fn, tn = 0, 0, 0, 0
    for i in scores:
        tp += scores[i]['TP']
        fp += scores[i]['FP']
        fn += scores[i]['FN']
        tn += scores[i]['TN']
    print('\t\t\t\tTrue\n' +
          '\t\t\t\tSpeech\tNon-speech\n' +
          'Predicted\tSpeech\t\t' + str(round(tp / (tp + fn), 3)) + '\t' + str(round(fp / (tn + fp), 3)) + '\n' +
          '\t\tNon-speech\t'+ str(round(fn / (tp + fn), 3)) + '\t' + str(round(tn / (tn + fp), 3)))


def score(wav_scp: pd.DataFrame, sys_segs: pd.DataFrame, ref_segs: pd.DataFrame, wav_segs: pd.DataFrame = None) -> Dict[str,Dict[str,int]]:
    '''Function for calculating the TP, FP, FN and TN counts from VAD segments and ground truth reference segments.

    Args:
        wav_scp: A pd.DataFrame containing information about the wavefiles that have been segmented.
        sys_segs: A pd.DataFrame containing the endpoints produced by a VAD system.
        ref_segs: A pd.DataFrame containing the ground truth reference endpoints.
        wav_segs (optional): A pd.DataFrame containing endpoints used prior to VAD segmentation. Only
        required if VAD was applied to subsets of wavefiles rather than the full files. (Default: None)

    Return:
        A dictionary of dictionaries containing TP, FP, FN and TN counts 
    '''

    ref_segs_masks = _segments_to_mask(wav_scp, ref_segs)
    sys_segs_masks = _segments_to_mask(wav_scp, sys_segs) 
    if wav_segs is not None:
        wav_segs_masks = _segments_to_mask(wav_scp, wav_segs)
    scores = {}
    for i in ref_segs_masks:
        if wav_segs is not None:
            score_array = wav_segs_masks[i] * (ref_segs_masks[i] - sys_segs_masks[i])
            num_ground_truth_p = int(np.sum(wav_segs_masks[i] * ref_segs_masks[i]))
            num_frames = int(np.sum(wav_segs_masks[i]))
        else:
            score_array = ref_segs_masks[i] - sys_segs_masks[i]
            num_ground_truth_p = int(np.sum(ref_segs_masks[i]))
            num_frames = len(ref_segs_masks[i])
        num_ground_truth_n = num_frames - num_ground_truth_p
        num_fn = (score_array == 1.0).sum()
        num_fp = (score_array == -1.0).sum()
        num_tp = num_ground_truth_p - num_fn
        num_tn = num_ground_truth_n - num_fp
        scores[i] = {'TP': num_tp, 'FP': num_fp, 'FN': num_fn, 'TN': num_tn}
    return scores


def _segments_to_mask(wav_scp: pd.DataFrame, segments: pd.DataFrame, frame_length: float = 0.01) -> Dict[str,np.ndarray]:
    '''Auxillary function used by score(). Creates a dictionary of recording-ids to np.ndarrays,
    which are boolean masks indicating the presence of segments within a recording.

    Args:
        wav_scp: A pd.DataFrame containing wav file data in the following columns:
            [recording-id, extended filename]
        segments: A pd.DataFrame containing segments file data in the following columns:
            [utterance-id, recording-id, start, end]
        frame_length (optional): The length of the frames used for scoring in seconds. (Default: 0.01)

    Returns:
        A dictionary mapping recording-ids to np.ndarrays, which are boolean masks of the frames
        which makeup segments within a recording.

    Example:
        A 0.1 second clip with a segment starting at 0.03 and ending 0.07 would yeild a mask:
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    '''

    wav_scp['duration'] = wav_scp['extended filename'].apply(lambda x: TinyTag.get(x).duration).astype(int)
    wav_scp['mask'] = round(wav_scp['duration'] / frame_length).astype(int).apply(np.zeros)
    segments['frames'] = (round(segments['end'] / frame_length).astype(int) - \
                          round(segments['start'] / frame_length).astype(int)).apply(np.ones)
    temp = wav_scp.merge(segments, on='recording-id')
    for n,_ in enumerate(temp['mask']):
        temp['mask'][n][round(temp['start'][n] / frame_length):round(temp['end'][n] / frame_length)] = temp['frames'][n]
    wav_scp['mask'] = temp['mask'].drop_duplicates().reset_index(drop=True)
    return wav_scp[['recording-id', 'mask']].set_index('recording-id')['mask'].to_dict()

# Handle args when run directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='evaluate',
                                     description='Evaluate the performance of VAD model output.')

    parser.add_argument('vad_input_dir', type=str,
                        help='a path to a Kaldi-style data directory that was used as input for the VAD experiment')
    
    parser.add_argument('vad_out_dir', type=str,
                        help='a path to a Kaldi-style data directory that was the output of the VAD experiment')

    parser.add_argument('ground_truth_dir', type=str,
                        help='a path to a Kaldi-style data directory containing the ground truth VAD segments')

    args = parser.parse_args()
    wav_scp, wav_segs, _ = utils.process_data_dir(args.vad_input_dir)
    _, sys_segs, _ = utils.process_data_dir(args.vad_out_dir)
    _, ref_segs, _ = utils.process_data_dir(args.ground_truth_dir)
    scores = score(wav_scp, sys_segs, ref_segs, wav_segs)
    print_confusion_matrix(scores)