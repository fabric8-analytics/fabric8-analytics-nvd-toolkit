"""Tests for feature_hooks module."""

import operator
import unittest

from toolkit.transformers import feature_hooks


class TestFeatureHooks(unittest.TestCase):
    """Module level unit tests for feature_hooks module."""

    def test_has_uppercase_hook(self):
        """Test has_uppercase_hook."""
        hook = feature_hooks.has_uppercase_hook

        tagged = [('test', '<tag>')]
        result = hook.__call__(tagged=tagged, pos=0)

        self.assertEqual(result, False)

        tagged = [('Test', '<tag>')]
        result = hook.__call__(tagged=tagged, pos=0)

        self.assertEqual(result, True)

        tagged = [('dayD', '<tag>')]
        result = hook.__call__(tagged=tagged, pos=0)

        self.assertEqual(result, True)

    def test_is_alnum_hook(self):
        """Test is_alnum_hook."""
        hook = feature_hooks.is_alnum_hook

        tagged = [('test', '<tag>')]
        result = hook.__call__(tagged=tagged, pos=0)

        self.assertEqual(result, True)

        tagged = [('test123', '<tag>')]
        result = hook.__call__(tagged=tagged, pos=0)

        self.assertEqual(result, True)

        tagged = [('test.alnum', '<tag>')]
        result = hook.__call__(tagged=tagged, pos=0)

        self.assertEqual(result, False)

        tagged = [('test&alnum', '<tag>')]
        result = hook.__call__(tagged=tagged, pos=0)

        self.assertEqual(result, False)

    def test_vendor_product_match_hook(self):
        """Test vendor_product_hook."""
        hook = feature_hooks.vendor_product_match_hook
        tagged = [('test', '<tag>'), ('proj', 'NUM'), ('1.0.0', '<VERSION>')]

        vendor, product = 'proj.io', 'codehouse'
        result = hook.__call__(tagged, vendor, product)

        self.assertTrue(result)

        vendor, product = 'codehouse', 'proj.io'
        result = hook.__call__(tagged, vendor, product)

        self.assertTrue(result)

        vendor, product = 'codehouse', 'test.io'
        result = hook.__call__(tagged, vendor, product)

        self.assertFalse(result)

    def test_ver_follows_hook(self):
        """Test ver_follows_hook."""
        hook = feature_hooks.ver_follows_hook

        tagged = [('test', '<tag>'), ('proj', 'NUM'), ('1.0.0', '<VERSION>')]
        result = hook.__call__(tagged, pos=1)

        self.assertTrue(result)

        tagged = [('1.0.0', '<VERSION>'), ('proj', 'NUM'), ('test', '<tag>')]
        result = hook.__call__(tagged, pos=1)

        self.assertFalse(result)

        tagged = [('1.0.0-wrong', '<tag>'), ('proj', 'NUM'), ('test', '<tag>')]
        result = hook.__call__(tagged, pos=1)

        self.assertFalse(result)

    def test_ver_pos_hook(self):
        """Test ver_pos_hook."""
        hook = feature_hooks.ver_pos_hook

        tagged = [('test', '<tag>'), ('proj', 'NUM'), ('1.0.0', '<VERSION>')]
        result = hook.__call__(tagged, pos=1)

        self.assertEqual(result, 1)

        tagged = [('1.0.0', '<VERSION>'), ('proj', 'NUM'), ('test', '<tag>')]
        result = hook.__call__(tagged, pos=1)

        self.assertEqual(result, -1)

        tagged = [('1.0.0', '<VERSION>'), ('test', '<tag>'), ('proj', 'NUM'),
                  ('1.0.0', '<VERSION>')]

        result = hook.__call__(tagged, pos=2)

        self.assertEqual(result, 1)

    def test_word_len_hook(self):
        """Test word_len_hook."""
        hook = feature_hooks.word_len_hook
        tagged = [('test', '<tag>')]

        # default limit argument
        result = hook.__call__(tagged=tagged, pos=0)

        self.assertEqual(result, True)

        # raise limit
        result = hook.__call__(tagged=tagged, pos=0, limit=5)

        self.assertEqual(result, False)

        # custom comparator
        result = hook.__call__(tagged=tagged, pos=0, cmp=operator.__lt__,
                               limit=5)

        self.assertEqual(result, True)
