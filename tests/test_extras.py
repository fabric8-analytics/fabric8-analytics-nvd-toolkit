"""Tests for extra modules."""

import unittest

from toolkit import __about__


class TestExtras(unittest.TestCase):
    """Unit tests for extra functionality."""

    def test_about(self):
        """Test __about__ module."""
        about = dict()

        with open(__about__.__file__) as f:
            exec(f.read(), about)

        self.assertFalse(not about)
