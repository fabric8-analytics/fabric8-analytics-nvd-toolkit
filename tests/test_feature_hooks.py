"""Tests for feature_hooks module."""

import operator
import unittest

from toolkit.transformers import feature_hooks


class TestFeatureHooks(unittest.TestCase):
    """Module level unit tests for feature_hooks module."""

    def test_has_uppercase_hook(self):
        """Test has_uppercase_hook."""
        hook = feature_hooks.has_uppercase_hook

        features = [('test', '<tag>')]
        result = hook.__call__(features=features, pos=0)

        self.assertEqual(result, False)

        features = [('Test', '<tag>')]
        result = hook.__call__(features=features, pos=0)

        self.assertEqual(result, True)

        features = [('dayD', '<tag>')]
        result = hook.__call__(features=features, pos=0)

        self.assertEqual(result, True)

    def test_is_alnum_hook(self):
        """Test is_alnum_hook."""
        hook = feature_hooks.is_alnum_hook

        features = [('test', '<tag>')]
        result = hook.__call__(features=features, pos=0)

        self.assertEqual(result, True)

        features = [('test123', '<tag>')]
        result = hook.__call__(features=features, pos=0)

        self.assertEqual(result, True)

        features = [('test.alnum', '<tag>')]
        result = hook.__call__(features=features, pos=0)

        self.assertEqual(result, False)

        features = [('test&alnum', '<tag>')]
        result = hook.__call__(features=features, pos=0)

        self.assertEqual(result, False)

    def test_vendor_product_match_hook(self):
        """Test vendor_product_hook."""
        hook = feature_hooks.vendor_product_match_hook

        from nvdlib.nvd import NVD
        feed = NVD.from_feeds(['recent'])
        feed.update()

        cve_list = list(feed.cves())

        # find application CPE
        cpe = cve = None
        for cve in cve_list:
            try:
                cpe = cve.configurations[0].cpe[0]
            except IndexError:
                continue

            if cpe.is_application():
                break

        assert all([cve, cpe]), "Failed to gather test data."

        vendor, product = cpe.vendor[0], cpe.product[0]

        # mock CVE with empty configurations instead of searching it
        empty_cve = type('emptyCVE', (), {})
        empty_cve.configurations = []

        cve_dict = {
            cve.cve_id: cve,
            'empty': empty_cve
        }

        # empty configurations
        features = [(product, 'NUM')]
        result = hook.__call__(features, 0, cve_dict, 'empty')

        self.assertFalse(result)

        # non existing ID
        result = hook.__call__(features, 0, cve_dict, 'non-existing-id')

        self.assertFalse(result)

        # matching product
        result = hook.__call__(features, 0, cve_dict, cve.cve_id)

        self.assertTrue(result)

        # matching vendor
        features = [(vendor, 'NUM')]
        result = hook.__call__(features, 0, cve_dict, cve.cve_id)

        self.assertTrue(result)

        # neither of vendor and product match
        features = [('mismatch', 'NUM')]
        result = hook.__call__(features, 0, cve_dict, cve.cve_id)

        self.assertFalse(result)

    def test_ver_follows_hook(self):
        """Test ver_follows_hook."""
        hook = feature_hooks.ver_follows_hook

        features = [('test', '<tag>'), ('proj', 'NUM'), ('1.0.0', '<VERSION>')]
        result = hook.__call__(features, pos=1)

        self.assertTrue(result)

        features = [('1.0.0', '<VERSION>'), ('proj', 'NUM'), ('test', '<tag>')]
        result = hook.__call__(features, pos=1)

        self.assertFalse(result)

        # incorrect version string
        features = [('1.0.0-wrong', '<tag>'), ('proj', 'NUM'), ('test', '<tag>')]
        result = hook.__call__(features, pos=1)

        self.assertFalse(result)

    def test_ver_precedes_hook(self):
        """Test ver_precedes_hook."""
        hook = feature_hooks.ver_precedes_hook

        features = [('test', '<tag>'), ('proj', 'NUM'), ('1.0.0', '<VERSION>')]
        result = hook.__call__(features, pos=1)

        self.assertFalse(result)

        features = [('1.0.0', '<VERSION>'), ('proj', 'NUM'), ('test', '<tag>')]
        result = hook.__call__(features, pos=1)

        self.assertTrue(result)

        # incorrect version string
        features = [('1.0.0-wrong', '<tag>'), ('proj', 'NUM'), ('test', '<tag>')]
        result = hook.__call__(features, pos=1)

        self.assertFalse(result)

    def test_ver_pos_hook(self):
        """Test ver_pos_hook."""
        hook = feature_hooks.ver_pos_hook

        features = [('test', '<tag>'), ('proj', 'NUM'), ('1.0.0', '<VERSION>')]
        result = hook.__call__(features, pos=1)

        self.assertEqual(result, 1)

        features = [('1.0.0', '<VERSION>'), ('proj', 'NUM'), ('test', '<tag>')]
        result = hook.__call__(features, pos=1)

        self.assertEqual(result, -1)

        features = [('1.0.0', '<VERSION>'), ('test', '<tag>'), ('proj', 'NUM'),
                    ('1.0.0', '<VERSION>')]

        result = hook.__call__(features, pos=2)

        self.assertEqual(result, 1)

    def test_word_len_hook(self):
        """Test word_len_hook."""
        hook = feature_hooks.word_len_hook
        features = [('test', '<tag>')]

        # default limit argument
        result = hook.__call__(features=features, pos=0)

        self.assertEqual(result, True)

        # raise limit
        result = hook.__call__(features=features, pos=0, limit=5)

        self.assertEqual(result, False)

        # custom comparator
        result = hook.__call__(features=features, pos=0, cmp=operator.__lt__,
                               limit=5)

        self.assertEqual(result, True)

    def test_word_in_dict_hook(self):
        """Test word_in_dict_hook."""
        hook = feature_hooks.word_in_dict_hook
        features = [('test', '<tag>')]

        result = hook.__call__(features=features, pos=0)

        self.assertEqual(result, True)
