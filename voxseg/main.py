import argparse
import os
import tensorflow as tf
from voxseg import extract_feats, run_cnnlstm, utils, evaluate
from tensorflow.keras import models

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py',
                                     description='Extracts features and run VAD to generate endpoints.')
    parser.add_argument('-M', '--model_path', type=str,
                        help='a path to a trained vad model saved as in .h5 format, overrides default pretrained model')
    parser.add_argument('-s', '--speech_thresh', type=float,
                       help='a decision threshold value between (0,1) for speech vs non-speech, defaults to 0.5')
    parser.add_argument('-m', '--speech_w_music_thresh', type=float,
                       help='a decision threshold value between (0,1) for speech_with_music vs non-speech, defaults to 0.5, \
                       increasing will remove more speech_with_music, useful for downsteam ASR')
    parser.add_argument('-f', '--median_filter_kernel', type=int,
                       help='a kernel size for a median filter to smooth the output labels, defaults to 1 (no smoothing)')
    parser.add_argument('-e', '--eval_dir', type=str,
                       help='a path to a Kaldi-style data directory containing the ground truth VAD segments for evaluation')
    parser.add_argument('data_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', and optionally \'segments\'')
    parser.add_argument('out_dir', type=str,
                        help='a path to an output directory where the output segments will be saved')
    args = parser.parse_args()

    data = extract_feats.prep_data(args.data_dir)
    feats = extract_feats.extract(data)
    feats = extract_feats.normalize(feats)
    if args.model_path is not None:
        model = models.load_model(args.model_path)
    else:
        model = models.load_model(f'{os.path.dirname(os.path.realpath(__file__))}/models/cnn_bilstm.h5')
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
    targets = run_cnnlstm.predict_targets(model, feats)
    endpoints = run_cnnlstm.decode(targets, speech_thresh, speech_w_music_thresh, filt)
    run_cnnlstm.to_data_dir(endpoints, args.out_dir)
    if args.eval_dir is not None:
        wav_scp, wav_segs, _ = utils.process_data_dir(args.data_dir)
        _, sys_segs, _ = utils.process_data_dir(args.out_dir)
        _, ref_segs, _ = utils.process_data_dir(args.eval_dir)
        scores = evaluate.score(wav_scp, sys_segs, ref_segs, wav_segs)
        print(scores)
        evaluate.print_confusion_matrix(scores)
