# Script for running CNN-BiLSTM vad model
# Author: Nick Wilkinson 2021
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
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


def predict_labels(model_path, feat_dir):
    feats = pd.read_hdf(feat_dir)
    model = models.load_model(model_path)
    labels = feats.drop(['normalized-features'], axis=1)
    labels['predicted-labels'] = _predict(model, feats['normalized-features'])
    return labels
    

def _predict(model, col):
    labels = []
    for feats in col:
        temp = model.predict(utils.time_distribute(feats, 15)[:,:,:,:,np.newaxis])
        labels.append(temp.reshape(-1, temp.shape[-1]))
    return pd.Series(labels)


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
    print(predict_labels(args.model_path, args.feat_dir))
