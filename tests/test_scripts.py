"""Tests for scripts module."""

import shlex
import subprocess
import tempfile
import unittest

from toolkit.scripts.get_packages_by_commit import get_packages_by_commit

TEST_REPO = "https://github.com/inversoft/prime-jwt/"
TEST_COMMIT = "0d94dcef0133d699f21d217e922564adbb83a227"


class TestScripts(unittest.TestCase):
    """Test various scripts in scripts module."""

    def test_get_packages_by_commit(self):
        tmp_dir = tempfile.mkdtemp(prefix="unittest_",
                                   suffix="_get_package")

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

        maven_package, = get_packages_by_commit(
            repository=tmp_dir,
            commit=TEST_COMMIT,
            ecosystem='maven',
        )

        gid, aid = maven_package.gid, maven_package.aid

        self.assertEqual(gid, 'com.inversoft')
        self.assertEqual(aid, 'prime-jwt')
