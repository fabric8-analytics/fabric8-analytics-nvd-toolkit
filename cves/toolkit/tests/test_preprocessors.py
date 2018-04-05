"""Tests for classifier module."""

import re
import typing
import unittest

from sklearn.pipeline import Pipeline

from toolkit.preprocessing import \
    NVDFeedPreprocessor, \
    LabelPreprocessor,\
    NLTKPreprocessor

from toolkit.pipeline import Hook
from toolkit.config import GITHUB_BASE_URL
from toolkit.utils import clear

TEST_SENT = "Test sentence, better not to worry too much."
TEST_DATA = [
    "Test sentence, better not to worry too much.",
    "Test sentence, better not to worry too much.",
]

TEST_REPOSITORY = GITHUB_BASE_URL + 'python/cpython'

# create TestCVE type
TestCVE = type('TestCVE', (), {})

# set up cve attributes and their values
TEST_CVE_ATTR = ['cve_id', 'references', 'description']
TEST_CVE_ATTR_VALS = ('cve_id', [TEST_REPOSITORY], 'description')

# assign attributes
_ = [
    setattr(TestCVE, attr, val)
    for attr, val in zip(TEST_CVE_ATTR, TEST_CVE_ATTR_VALS)
]
TEST_CVE = TestCVE()


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
        result = prep.tokenize(TEST_SENT)
        self.assertIsInstance(result, typing.Iterable)

        print(result)
        # check that punctuation has been got rid of
        self.assertFalse(any(re.match(u"[,.]", t[0]) for t in result))
        # check that the list contains elements of same type
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
        # check that the array is not empty
        self.assertTrue(transform.size > 0)

        # rest of the tests should be covered by `test_tokenize`

    def test_pipeline(self):
        """Test NLTKPreprocessor as a single pipeline unit."""
        # should not raise, since NLTKPreprocessor does implement `fit `
        # and `transform` methods
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
        result, = prep.transform(X=[TEST_CVE])

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

        self.assertIsInstance(result, tuple)
        self.assertEqual(result.repository, TEST_REPOSITORY)
        self.assertIsInstance(result.references, list)

    def test_pipeline(self):
        """Test NVDFeedPreprocessor as a single pipeline unit."""
        # should not raise, since NLTKPreprocessor does implement `fit`
        # and `transform` methods
        _ = Pipeline([
            ('preprocessor', NVDFeedPreprocessor)
        ])


class TestLabelPreprocessor(unittest.TestCase):
    """Tests for LabelPreprocessor class."""

    def test_init(self):
        hook = Hook(key='label', func=lambda x: x)
        attributes = ['project']

        label_prep = LabelPreprocessor(feed_attributes=attributes,
                                       hook=hook)

        self.assertIsInstance(label_prep, LabelPreprocessor)
        self.assertIsInstance(label_prep._hook, Hook)  # pylint: disable=protected-access

    @clear
    def test_transform(self):
        """Test LabelPreprocessor `transform` method"""
        hook = Hook(key='label', func=lambda x: x)
        attributes = ['repository']

        test_data = [TEST_CVE]

        # prepare the data for label preprocessor
        feed_prep = NVDFeedPreprocessor(attributes, skip_duplicity=True)
        test_data = feed_prep.transform(X=test_data)

        label_prep = LabelPreprocessor(feed_attributes=attributes,
                                       hook=hook)
        test_data = label_prep.fit(test_data)
        self.assertFalse(not test_data)

        res = test_data,
        self.assertTrue(res, TEST_REPOSITORY)

    @clear
    def test_fit_transform(self):
        """Test LabelPreprocessor `fit_transform` method"""
        hook = Hook(key='label', func=lambda p, d: 'label')
        attributes = ['project', 'description']

        test_data = [TEST_CVE]

        # prepare the data for label preprocessor
        feed_prep = NVDFeedPreprocessor(attributes, skip_duplicity=True)
        test_data = feed_prep.transform(X=test_data)

        label_prep = LabelPreprocessor(feed_attributes=attributes,
                                       hook=hook)

        test_data = label_prep.fit_transform(test_data)
        # check that correct label is returned by the Hook
        label = test_data[0, 1]

        self.assertTrue(test_data.size > 0)
        self.assertEqual(label, 'label')

