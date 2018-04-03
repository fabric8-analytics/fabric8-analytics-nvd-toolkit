"""Tests for classifier module."""

import re
import typing
import pytest
import unittest

from sklearn.pipeline import Pipeline

from toolkit.preprocessing import \
    GitHubHandler, \
    NVDFeedPreprocessor, \
    NLTKPreprocessor,\
    FeatureExtractor,\
    Hook

# noinspection PyProtectedMember
from toolkit.preprocessing.preprocessors import _FeatureExtractor
from toolkit.config import GITHUB_BASE_URL

TEST_SENT = "Test sentence, better not to worry too much."
TEST_DATA = [
    "Test sentence, better not to worry too much.",
    "Test sentence, better not to worry too much.",
]

TEST_REPOSITORY = GITHUB_BASE_URL + 'python/cpython'

# create TestCVE type
TestCVE = type('TestCVE', (), {})

# set up cve attributes and their values
TEST_CVE_ATTR = ('cve_id', 'references', 'description')
TEST_CVE_ATTR_VALS = ('cve_id', [TEST_REPOSITORY], 'description')

# assign attributes
_ = [
    setattr(TestCVE, attr, val)
    for attr, val in zip(TEST_CVE_ATTR, TEST_CVE_ATTR_VALS)
]
TEST_CVE = TestCVE()


def clear(func):
    """Decorator which performs cleanup before function call."""

    # noinspection PyUnusedLocal,PyUnusedLocal
    def wrapper(*args, **kwargs):  # pylint: disable=unused-argument
        # perform cleanup
        Hook.clear_current_instances()
        exc = None
        ret_values = None

        # run the function
        try:
            ret_values = func(*args, **kwargs)
        except BaseException as e:
            # caught any exceptions which will be reraised
            exc = e
        finally:
            # cleanup again
            Hook.clear_current_instances()

            if exc is not None:
                raise exc
            return ret_values

    return wrapper


class TestNLTKPreprocessor(unittest.TestCase):
    """Tests for NLTKPreprocessor class."""

    def test_init(self):
        """Test NLTKPreprocessor initialization."""
        # default parameters
        prep = NLTKPreprocessor()

        self.assertIsInstance(prep, NLTKPreprocessor)

        # custom parameters
        prep = NLTKPreprocessor(
            stopwords=True,
            lower=True
        )

        self.assertIsNotNone(prep._stopwords)  # pylint: disable=protected-access
        self.assertIsInstance(prep, NLTKPreprocessor)

    def test_tokenize(self):
        """Test NLTKPreprocessor `tokenize` method."""
        prep = NLTKPreprocessor(
            stopwords=True,
            lower=True
        )
        tokenized = prep.tokenize(TEST_SENT)
        self.assertIsInstance(tokenized, typing.Generator)

        result = list(tokenized)

        # check that the list is not empty
        self.assertIsInstance(result, list)
        # check that punctuation has been got rid of
        self.assertFalse(any(re.match(u"[,.]", t[0]) for t in result))
        # check that the resulting list contains tuples
        self.assertTrue(all(isinstance(t, tuple) for t in result))
        # check that the list contains tuples of same type
        self.assertTrue(all(isinstance(t[0], type(t[1])) for t in result))

    def test_transform(self):
        """Test NLTKPreprocessor `transform` method."""
        # custom parameters
        prep = NLTKPreprocessor(
            stopwords=True,
            lower=True
        )
        transform = prep.transform(X=TEST_DATA)

        self.assertTrue(len(transform), len(TEST_DATA))
        # check that neither list is empty
        self.assertFalse(any(not l for l in transform))

        # rest of the tests should be covered by `test_tokenize`

    def test_pipeline(self):
        """Test NLTKPreprocessor as a single pipeline unit."""
        # should raise, since NLTKPreprocessor does not implement `fit` method
        with pytest.raises(TypeError):
            _ = Pipeline([
                ('preprocessor', NLTKPreprocessor)
            ])


class TestNVDFeedPreprocessor(unittest.TestCase):
    """Tests for NVDFeedPreprocessor class."""

    def test_init(self):
        """Test NVDFeedPreprocessor `__init__` method."""
        # default parameters
        prep = NVDFeedPreprocessor()

        self.assertIsInstance(prep, NVDFeedPreprocessor)

        # custom parameters
        prep = NVDFeedPreprocessor(
            attributes=['cve_id']
        )
        self.assertIsInstance(prep, NVDFeedPreprocessor)

    def test_transform(self):
        """Test NVDFeedPreprocessor `transform` method."""
        # custom parameters
        prep = NVDFeedPreprocessor(
            attributes=TEST_CVE_ATTR  # only extract cve_id
        )
        result, = prep.transform(cves=[TestCVE])

        # result should be a tuple of default handlers and cve attributes
        # check only the cve attributes here to avoid calling handler
        # separately
        self.assertEqual(result[-len(TEST_CVE_ATTR):], TEST_CVE_ATTR_VALS)

    def test_filter_by_handler(self):
        """Test NVDFeedPreprocessor `_filter_by_handler` method."""
        prep = NVDFeedPreprocessor()

        # make another cve
        cve_without_ref = TestCVE()
        # add reasonable reference
        cve_without_ref.references = 'non-matching-reference'
        cves = [cve_without_ref]
        cves = prep._filter_by_handler(cves)  # pylint: disable=protected-access

        # check that cves list is empty
        self.assertTrue(not cves)

        # noinspection PyTypeChecker
        cves = [TEST_CVE]
        cves = prep._filter_by_handler(cves)  # pylint: disable=protected-access

        self.assertFalse(not cves)
        self.assertIsInstance(cves[0], tuple)

        # resulting tuple
        cve, ref = cves[0]

        # check that cves list is not empty
        self.assertIsInstance(cve, TestCVE)
        self.assertIsInstance(ref, str)

    # noinspection PyUnresolvedReferences
    def test_get_cve_attributes(self):
        """Test NVDFeedPreprocessor `_get_cve_attributes` method."""
        prep = NVDFeedPreprocessor()
        cve_tuple, = prep._filter_by_handler(cves=[TestCVE()])
        result = prep._get_cve_attributes(cve_tuple)  # pylint: disable=protected-access

        print(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result.repository, TEST_REPOSITORY)
        self.assertEqual(result.languages, None)
        self.assertIsInstance(result.references, list)

    def test_pipeline(self):
        """Test NVDFeedPreprocessor as a single pipeline unit."""
        # should raise, since NLTKPreprocessor does not implement `fit` method
        with pytest.raises(TypeError):
            _ = Pipeline([
                ('preprocessor', NVDFeedPreprocessor)
            ])


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
        tokenized = tokenizer.transform(TEST_DATA)

        # apply default extractors transformation
        prep = FeatureExtractor()

        sent = tokenized[0]
        result = prep._extract_features(sent, word_pos=0)

        self.assertIsInstance(result, dict)
        # check few expected results
        self.assertEqual(result['prev-word'], '<start>')
        self.assertEqual(result['prev-tag'], '<start>')

    @clear
    def test_transform(self):
        """Test FeatureExtractor `transform` method."""
        # preprocess the sentences
        tokenizer = NLTKPreprocessor()
        tokenized = tokenizer.transform(TEST_DATA)

        # apply default extractors transformation
        prep = FeatureExtractor()
        transform = prep.transform(X=tokenized)

        self.assertTrue(len(transform), len(TEST_DATA))
        # check that all elements ale dicts
        self.assertFalse(all(isinstance(obj, dict) for obj in transform))

        # delete to get rid of old keys
        del prep

        # apply transformation with custom feature_keys
        prep = FeatureExtractor(
            features={
                'useless-feature': lambda s, w, t: True,
            }
        )
        with pytest.raises(TypeError):
            # raises if skip=False (default), since arguments `s`, `w`, `t`
            # were not fed
            _ = prep.transform(X=tokenized)

        # skip=True
        transform = prep.transform(X=tokenized, skip=True)

        self.assertTrue(len(transform), len(TEST_DATA))
        # check that all elements ale lists
        self.assertTrue(all(isinstance(obj, list) for obj in transform))

    @clear
    def test_pipeline(self):
        """Test FeatureExtractor as a single pipeline unit."""
        # should raise, since NLTKPreprocessor does not implement `fit` method
        with pytest.raises(TypeError):
            _ = Pipeline([
                ('preprocessor', FeatureExtractor)
            ])


class TestHook(unittest.TestCase):
    """Tests for Hook class."""

    def test_hook(self):
        """Test Hook initialization and key handling."""
        hook = Hook(key='key', func=lambda: 'test')

        self.assertEqual(hook.key, 'key')
        # w/o args
        self.assertEqual(hook(), 'test')
        # check invalid key
        with pytest.raises(ValueError):
            _ = Hook(key='key', func=lambda: None)

        # hook with args
        hook_args = Hook(key='key_', func=lambda x: x)

        self.assertEqual(Hook.get_current_keys(), {'key', 'key_'})
        self.assertEqual(hook_args.key, 'key_')


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
        result = _prep.feed({'x': 'test'}, skip=True)
        self.assertIsInstance(result, dict)

        key, value = list(*result.items())

        self.assertEqual(key, 'key')
        self.assertEqual(value, 'test')

        # feed and disable skip
        with pytest.raises(TypeError):
            result = _prep.feed({'x': 'test'}, skip=False)
            key, value = list(*result.items())

            self.assertEqual(key, 'key')
            self.assertEqual(value, 'test')
