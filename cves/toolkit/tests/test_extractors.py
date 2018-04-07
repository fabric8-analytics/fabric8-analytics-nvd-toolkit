"""Tests for extractors module."""

import unittest

from sklearn.pipeline import Pipeline

from toolkit.preprocessing import NLTKPreprocessor
from toolkit.transformers import FeatureExtractor, Hook
# noinspection PyProtectedMember
from toolkit.transformers.extractors import _FeatureExtractor  # pylint: disable=protected-access
from toolkit.utils import clear

TEST_DATA = [
    "Test sentence, better not to worry too much.",
    "Test sentence, better not to worry too much.",
]


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
            features={
                feature: lambda w, t: True,
            }
        )

        self.assertIsInstance(prep, FeatureExtractor)
        self.assertIsInstance(prep, FeatureExtractor)
        # check that the custom feature_keys has been added
        self.assertTrue(feature in prep.feature_keys)

    @clear
    def test_extract_features(self):
        """Test FeatureExtractor `_extract_features` method"""
        # preprocess the sentences
        tokenizer = NLTKPreprocessor()
        tokenized = tokenizer.fit_transform(TEST_DATA)
        sent = tokenized.values[0][0]

        # apply default extractors transformation
        prep = FeatureExtractor()
        result = prep._extract_features(sent, word_pos=0)
        print(result)

        self.assertIsInstance(result, dict)
        # check few expected results
        self.assertEqual(result['prev-word'], '<start>')
        self.assertEqual(result['prev-tag'], '<start>')

    @clear
    def test_fit_transform(self):
        """Test FeatureExtractor `fit_transform` method."""
        # preprocess the sentences
        tokenizer = NLTKPreprocessor()
        tokenized = tokenizer.fit_transform(TEST_DATA)

        data = tokenized.values

        # apply default extractors transformation
        prep = FeatureExtractor()
        result = prep.fit_transform(X=data)

        self.assertEqual(len(result), len(data))
        # check that all elements ale dicts
        self.assertTrue(all([isinstance(obj, dict) for obj in result[0, :1]]))

        # delete to get rid of old keys
        del prep

        # apply transformation with custom feature_keys
        prep = FeatureExtractor(
            features={
                'useless-feature': lambda s, w, t: True,
            }
        )

        with self.assertRaises(TypeError):
            # raises if skip=False (default), since arguments `s`, `w`, `t`
            # were not fed
            _ = prep.fit_transform(X=data)

        # skip=True
        result = prep.fit_transform(X=data, skip_unfed_hooks=True)

        self.assertEqual(len(result), len(TEST_DATA))
        # check that all elements ale lists
        self.assertTrue(all(isinstance(obj, dict) for obj in result[:, :1]))

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

        self.assertFalse(not _prep._hooks)  # pylint: disable=protected-access

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
