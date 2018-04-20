"""Tests for extractors module."""

import unittest

import numpy as np
from sklearn.pipeline import Pipeline

from toolkit.pipelines import get_preprocessing_pipeline
from toolkit.transformers import FeatureExtractor, Hook
# noinspection PyProtectedMember
from toolkit.transformers.extractors import _FeatureExtractor  # pylint: disable=protected-access
from toolkit.utils import clear


class TestFeatureExtractor(unittest.TestCase):
    """Tests for FeatureExtractor class."""

    @clear
    def test_init(self):
        """Test FeatureExtractor initialization."""
        # default parameters
        prep = FeatureExtractor()

        self.assertIsInstance(prep, FeatureExtractor)
        self.assertIsInstance(prep, FeatureExtractor)

        # custom feature_keys
        feature = 'useless-feature'
        # delete to get rid of old keys
        del prep

        prep = FeatureExtractor(
            feature_hooks={
                feature: lambda w, t: True,
            }
        )

        self.assertIsInstance(prep, FeatureExtractor)
        self.assertIsInstance(prep, FeatureExtractor)
        # check that the custom feature_keys has been added
        self.assertTrue(feature in prep.feature_keys)

    @clear
    def test_extract_features(self):
        """Test FeatureExtractor `_extract_features` method."""
        # preprocess the sentences
        test_data = _get_preprocessed_test_data()
        # get tokenized sentence
        sent = test_data[0].values

        # apply default extractors transformation
        prep = FeatureExtractor()
        result = prep._extract_features(sent, word_pos=0)

        self.assertIsInstance(result, dict)
        # check few expected results
        self.assertEqual(result['prev-word'], '<start>')
        self.assertEqual(result['prev-tag'], '<start>')

    @clear
    def test_fit_transform(self):
        """Test FeatureExtractor `fit_transform` method."""
        # preprocess the sentences
        test_data = _get_preprocessed_test_data()
        test_data = np.array(test_data)

        test_data, test_labels = test_data[:, 0], test_data[:, 1]

        # apply default extractors transformation
        prep = FeatureExtractor()
        result = prep.fit_transform(X=test_data)

        self.assertEqual(len(result), len(test_data))

        # delete to get rid of old keys
        del prep

        # apply transformation with custom feature_keys
        prep = FeatureExtractor(
            feature_hooks={
                'useless-feature': lambda s, w, t: True,
            }
        )

        with self.assertRaises(TypeError):
            # raises if skip=False (default), since arguments `s`, `w`, `t`
            # were not fed
            _ = prep.fit_transform(X=test_data)

        # skip=True
        result = prep.fit_transform(X=test_data, skip_unfed_hooks=True)

        self.assertEqual(len(result), len(test_data))

    @clear
    def test_pipeline(self):
        """Test FeatureExtractor as a single pipeline unit."""
        # should not raise, since NLTKPreprocessor does implement `fit`
        # and `transform` methods
        _ = Pipeline([
            ('preprocessor', FeatureExtractor)
        ])


# noinspection PyPep8Naming
class Test_FeatureExtractor(unittest.TestCase):
    """Tests for _FeatureExtractor class."""

    @clear
    def test_init(self):
        """Test _FeatureExtractor initialization."""
        _prep = _FeatureExtractor()

        self.assertTrue(any(_prep._hooks))  # pylint: disable=protected-access

    @clear
    def test_update(self):
        """Test _FeatureExtractor update method."""
        hook = Hook(key='key', func=lambda: None)
        _prep = _FeatureExtractor().update(hook)

        self.assertTrue('key' in _prep.keys)

    @clear
    def test_feed(self):
        """Test _FeatureExtractor feed method."""
        hook = Hook(key='key', func=lambda x: x)
        _prep = _FeatureExtractor().update(hook)

        # feed the extractor with skip=True
        result = _prep.feed({'x': 'test'}, skip_unfed_hooks=True)
        self.assertIsInstance(result, dict)

        key, value = list(*result.items())

        self.assertEqual(key, 'key')
        self.assertEqual(value, 'test')

        # feed and disable skip
        with self.assertRaises(TypeError):
            result = _prep.feed({'x': 'test'}, skip_unfed_hooks=False)
            key, value = list(*result.items())

            self.assertEqual(key, 'key')
            self.assertEqual(value, 'test')


def _get_preprocessed_test_data():
    """Return preprocessed data.

    Note: used for tests only.
    """
    from nvdlib.nvd import NVD

    feed = NVD.from_feeds(feed_names=['recent'])
    # download and update
    feed.update()

    # get the sample cves
    __cve_iter = feed.cves()
    __records = 500

    data = [next(__cve_iter) for _ in range(__records)]  # only first n to speed up tests
    pipeline = get_preprocessing_pipeline()
    steps, preps = list(zip(*pipeline.steps))

    # set up fit parameters (see sklearn fit_params notation)
    fit_params = {
        "%s__feed_attributes" % steps[2]: ['description'],
        "%s__output_attributes" % steps[2]: ['label']
    }

    prep_data = pipeline.fit_transform(
        X=data,
        **fit_params
    )

    return prep_data
