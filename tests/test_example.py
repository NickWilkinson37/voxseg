import os
import unittest
from voxseg import extract_feats, run_cnnlstm
import pandas as pd
from tensorflow.keras import models


class TestExample(unittest.TestCase):
    """Basic test cases."""

    def test_feature_extraction(self):
        print(os.path.dirname(__file__))
        data = extract_feats.prep_data('tests/data')
        feats = extract_feats.extract(data)
        feats = extract_feats.normalize(feats)
        print(feats)

    def test_model(self):
        print(os.path.dirname(__file__))
        feats = pd.read_hdf('tests/features/feats.h5')
        model = models.load_model('tests/model/model.h5')
        labels = run_cnnlstm.predict_labels(model, feats)
        print(labels)


if __name__ == '__main__':
    unittest.main()