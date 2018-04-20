"""Tests for handlers module."""

import unittest

from toolkit.preprocessing.handlers import GitHubHandler, StatusError


TEST_USER_PROJ = ('fabric8-analytics', 'fabric8-analytics-common')
TEST_REPO_SRC_URL = u'https://github.com/fabric8-analytics/fabric8-analytics-common'
TEST_REPO_BLOB_URL = TEST_REPO_SRC_URL + u'/blob/master'


class TestGitHubHandler(unittest.TestCase):
    """Tests for handlers module."""

    def test_init(self):
        """Test repo handler initialization and attributes."""
        repo = GitHubHandler(url=TEST_REPO_BLOB_URL)

        self.assertIsInstance(repo, GitHubHandler)

        # check for attribute initialization
        self.assertTrue(hasattr(repo, 'repository'))
        self.assertTrue(hasattr(repo, 'user'))
        self.assertTrue(hasattr(repo, 'project'))
        try:
            self.assertTrue(hasattr(repo, 'languages'))
        except StatusError:
            # except error (probably caused by OAUTH_TOKEN missing, not
            # purpose of this test
            pass

    def test_strip_src_url(self):
        """Test GitHubHandler `strip_src_url` method."""
        repo = GitHubHandler(url=TEST_REPO_BLOB_URL)

        self.assertEqual(repo.repository, TEST_REPO_SRC_URL)

    def test_get_user_project(self):
        """Test GitHubHandler `get_user_project` method."""
        repo = GitHubHandler(url=TEST_REPO_BLOB_URL)
        user, project = repo.user, repo.project

        self.assertEqual((user, project), TEST_USER_PROJ)

    def test_languages(self):
        """Test GitHubHandler `languages` property."""
        # TODO: think of how to provide public OAUTH_TOKEN for testing
        pass
