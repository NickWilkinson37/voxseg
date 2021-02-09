import argparse
import tensorflow as tf
from voxseg import extract_feats, run_cnnlstm
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
    parser = argparse.ArgumentParser(prog='VAD',
                                     description='Extracts features and run VAD to generate endpoints.')
    parser.add_argument('-m', '--model_path', type=str,
                        help='a path to a trained vad model saved as in .h5 format, overriding the default model')
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
        model = models.load_model('voxseg/models/cnn_bilstm.h5')
    targets = run_cnnlstm.predict_targets(model, feats)
    endpoints = run_cnnlstm.decode(targets)
    run_cnnlstm.to_data_dir(endpoints, args.out_dir)