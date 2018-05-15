"""Tests for utils module."""

import pytest
import unittest

from nvdlib.nvd import NVD

from toolkit import utils
from toolkit.preprocessing.handlers import GitHubHandler, StatusError

TEST_REFERENCE_HTTP = 'http://github.com/user/project/blob/master'
TEST_REFERENCE_HTTPS = 'https://github.com/user/project/blob/master'
TEST_REFERENCE_WRONG = 'http://gitlab.com/user/project/blob/master'

TEST_REFERENCE_PATTERNS = {
    TEST_REFERENCE_HTTP: True,
    TEST_REFERENCE_HTTPS: True,
    TEST_REFERENCE_WRONG: False,
}


class TestUtils(unittest.TestCase):
    """Tests for utils module."""

    def test_classproperty(self):
        """Test classproperty decorator."""
        class Sample:
            _secret = 'secret'

            # noinspection PyMethodParameters
            @utils.classproperty
            def secret(cls):
                return cls._secret

        # check readability
        self.assertEqual(Sample.secret, 'secret')

        # check overwrite protection and delete protections
        # # TODO: solve these -- should raise
        # with pytest.raises(AttributeError):
        #     # setter
        #     Sample.secret = 'not_so_secret'
        #     # delete
        #     del Sample.secret

    def test_has_reference(self):
        """Test utils.has_reference() function."""
        # Create sample extensible cve object for testing
        cve = type('', (), {})
        cve.references = TEST_REFERENCE_PATTERNS.keys()
        # test urls
        ret = utils.has_reference(cve, url=TEST_REFERENCE_HTTP)
        self.assertTrue(ret)

        for k, v in TEST_REFERENCE_PATTERNS.items():  # pylint: disable=invalid-name
            # test  patterns
            cve.references = [k]
            ret = utils.has_reference(cve, pattern='github')
            self.assertEqual(ret, v)

    def test_get_reference(self):
        """Test utils.get_reference() function."""
        # Create sample extensible cve object for testing
        cve = type('', (), {})
        cve.references = TEST_REFERENCE_PATTERNS.keys()
        # test urls
        ret = utils.get_reference(cve, url=TEST_REFERENCE_HTTP)
        self.assertEqual(ret, TEST_REFERENCE_HTTP)

        for k, v in TEST_REFERENCE_PATTERNS.items():  # pylint: disable=invalid-name
            # test  patterns
            cve.references = [k]
            ret = utils.get_reference(cve, pattern='github')
            self.assertEqual(ret, [None, k][v])

    def test_find_(self):
        """Test utils.find_ function."""
        word = 'project'
        # test case insensitive (default)
        sample = 'This document belongs to the Project.'
        found = utils.find_(word, sample)

        self.assertIsNotNone(found)
        self.assertEqual(found.lower(), word.lower())

        # test case sensitive
        sample = 'This document belongs to the Project.'
        found = utils.find_(word, sample, ignore_case=False)

        self.assertIsNone(found)

    def test_nvd_to_dataframe(self):
        """Test NVD feed transformation to pandas.DataFrame object."""
        from pandas import DataFrame

        # test without handler
        cves = list(NVD.from_feeds(['recent']).cves())
        df = utils.nvd_to_dataframe(cves)

        self.assertIsNotNone(df)
        self.assertIsInstance(df, DataFrame)

        # test with handler - should raise cause of missing gh token
        with self.assertRaises(StatusError):
            _ = utils.nvd_to_dataframe(cves, handler=GitHubHandler)
