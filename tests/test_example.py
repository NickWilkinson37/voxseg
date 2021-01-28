import os
import unittest
from voxseg import extract_feats


class TestExample(unittest.TestCase):
    """Basic test cases."""

    def test_feature_extraction(self):
        print(os.path.dirname(__file__))
        data = extract_feats.prep_data('tests/data')
        feats = extract_feats.extract(data)
        feats = extract_feats.normalize(feats)
        print(feats)


if __name__ == '__main__':
    unittest.main()