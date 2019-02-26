"""Tests for scripts module."""

import shlex
import subprocess
import tempfile
import unittest

from toolkit.scripts.get_packages_by_commit import \
    main,\
    get_packages_by_commits

TEST_REPO = "https://github.com/inversoft/prime-jwt/"
TEST_COMMITS = ["d31f908b46fe9d5088d4a605484afb844206d75"]


class TestScripts(unittest.TestCase):
    """Test various scripts in scripts module."""

    def test_main_help(self):
        """Test argument parser's help."""
        argv = ['--help']
        with self.assertRaises(SystemExit):
            main(argv)

    def test_main_no_args(self):
        """Test main function with no arguments."""
        argv = []
        with self.assertRaises(SystemExit) as exc:
            main(argv)

        self.assertNotEqual(exc.exception.code, 0)

    def test_main_default(self):
        """Test main function with default arguments."""
        argv = ['-repo', TEST_REPO,
                '--commits', *TEST_COMMITS]
        ret_val = main(argv)

        self.assertIsNone(ret_val)

    def test_get_packages_by_commits(self):
        """Test package retrieval by commit messages."""
        tmp_dir = tempfile.mkdtemp(prefix="unittest_",
                                   suffix="_get_package")

        self.assertIsNotNone(tmp_dir)

        git_clone_cmd = "git clone {repo} {dest}".format(
            repo=TEST_REPO,
            dest=tmp_dir
        )
        cmd = [
            # ensure every argument is shell-quoted to prevent
            # accidental shell injection
            shlex.quote(arg) for arg in shlex.split(git_clone_cmd)
        ]

        pcs = subprocess.Popen(cmd)
        pcs.communicate()

        self.assertEqual(pcs.returncode, 0)

        maven_packages = get_packages_by_commits(
            repository=tmp_dir,
            commits=TEST_COMMITS,
            ecosystem='maven',
        )

        # make sure at least one Maven package is found
        assert maven_packages is not None
        assert len(maven_packages) >= 1

        maven_package = maven_packages[0]
        gid, aid = maven_package.gid, maven_package.aid

        # check the attributes
        self.assertEqual(gid, 'io.fusionauth')
        self.assertEqual(aid, 'fusionauth-jwt')
