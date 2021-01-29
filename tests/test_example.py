import os
import unittest
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
from tensorflow.keras import models
from voxseg import extract_feats, run_cnnlstm, utils


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
        model = models.load_model('tests/model/model.h5')
        labels = run_cnnlstm.predict_labels(model, feats)
        print(labels)


if __name__ == '__main__':
    unittest.main()