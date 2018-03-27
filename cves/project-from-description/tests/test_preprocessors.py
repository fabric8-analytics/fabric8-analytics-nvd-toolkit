"""Tests for classifier module."""

import re
import typing
import unittest

from toolkit.preprocessing import NLTKPreprocessor


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
        test_sent = "Test sentence, better not to worry too much."
        tokenized = prep.tokenize(test_sent)
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
        test_data = [
            "Test sentence, better not to worry too much.",
            "Test sentence, better not to worry too much.",
        ]
        transform = prep.transform(X=test_data)

        self.assertTrue(len(transform), len(test_data))
        # check that neither list is empty
        self.assertFalse(any(not l for l in transform))

        # rest of the tests should be covered by `test_tokenize`
