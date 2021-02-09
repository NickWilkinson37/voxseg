# Tests for the voxseg package
# Author: Nick Wilkinson 2021
import os
import unittest
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
from tensorflow.keras import models
from voxseg import evaluate, extract_feats, run_cnnlstm, utils


class TestExample(unittest.TestCase):
    """Basic test cases."""

    def test_feature_extraction(self):
        data = extract_feats.prep_data('tests/data')
        feats = extract_feats.extract(data)
        feats = extract_feats.normalize(feats)
        utils.save(feats, 'tests/features/feats.h5')
        print(feats)

    def test_model(self):
        feats = pd.read_hdf('tests/features/feats.h5')
        model = models.load_model('voxseg/models/cnn_bilstm.h5')
        targets = run_cnnlstm.predict_targets(model, feats)
        endpoints = run_cnnlstm.decode(targets)
        run_cnnlstm.to_data_dir(endpoints, 'tests/output')
        print(endpoints)

    def test_evaluate(self):
        wav_scp, wav_segs, _ = utils.process_data_dir('tests/data')
        _, sys_segs, _ = utils.process_data_dir('tests/output')
        _, ref_segs, _ = utils.process_data_dir('tests/ground_truth')
        scores = evaluate.score(wav_scp, sys_segs, ref_segs, wav_segs)
        evaluate.print_confusion_matrix(scores)


if __name__ == '__main__':
    unittest.main()