"""Tests for handlers module."""

import os
import tempfile
import unittest

from toolkit.preprocessing.handlers import StatusError
from toolkit.preprocessing.handlers import GitHubHandler, GitHandler


TEST_USER_PROJ = ('fabric8-analytics', 'fabric8-analytics-common')
TEST_REPO_SRC_URL = u'https://github.com/fabric8-analytics/fabric8-analytics-common'
TEST_REPO_BLOB_URL = TEST_REPO_SRC_URL + u'/blob/master'

TEST_GIT_REPO_URL = u'https://github.com/fabric8-analytics/fabric8-analytics-nvd-toolkit'
TEST_GIT_REPO = u'fabric8-analytics-nvd-toolkit'

TEST_COMMITS = ['6cd15101c976c5de3b9fdc834d0b4d3f']


class TestGitHubHandler(unittest.TestCase):
    """Tests for GitHubHandler class."""

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


class TestGitHandler(unittest.TestCase):
    """Tests for GitHandler class."""

    def test_init(self):
        """Test GitHandler initialization."""
        tmp_dir = tempfile.mkdtemp(prefix='test_', suffix='_init')
        with self.assertRaises(StatusError):
            _ = GitHandler(tmp_dir)

        handler = GitHandler.clone(url=TEST_GIT_REPO_URL)

        self.assertIsInstance(handler, GitHandler)

    def test_clone(self):
        """Test GitHandler's `clone` method."""
        with self.assertRaises(StatusError):
            _ = GitHandler.clone(url='https://nonexisting/repo')

        handler = GitHandler.clone(url=TEST_GIT_REPO_URL)

        self.assertIsInstance(handler, GitHandler)

    def test_exec_cmd(self):
        """Test GitHandler's `exec_cmd` method."""
        with self.assertRaises(ValueError):
            # should raise, command does not start with git
            GitHandler.exec_cmd('echo "Hello Git!"')

        tmp_dir = tempfile.mkdtemp(prefix='test_', suffix='_exec_cmd')

        _, __ = GitHandler.exec_cmd(
            'git clone {repo} {dest}'.format(
                repo=TEST_GIT_REPO_URL, dest=tmp_dir
            ))

        self.assertTrue(any(os.listdir(tmp_dir)))

    def test_get_modified_files(self):
        """Test GitHandler's `get_modified_files` method."""

        handler = GitHandler.clone(url=TEST_GIT_REPO_URL)

        mod_file, = handler.get_modified_files(commits=TEST_COMMITS)

        self.assertEqual(os.path.basename(mod_file), 'README.md')

        # test multiple commits
        mod_files = handler.get_modified_files(commits=TEST_COMMITS * 2)

        self.assertEqual(len(mod_files), 2)
