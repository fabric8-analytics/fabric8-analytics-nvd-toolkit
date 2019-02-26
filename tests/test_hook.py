"""Tests for Hook module."""

import pytest
import unittest

from toolkit.transformers import Hook


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

        self.assertIsNotNone(hook_args)
        self.assertEqual(Hook.get_current_keys(), {'key', 'key_'})
        self.assertEqual(hook_args.key, 'key_')
