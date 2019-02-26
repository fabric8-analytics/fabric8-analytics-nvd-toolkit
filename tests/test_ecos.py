"""Tests for ecos module."""

import os
import unittest

from unittest.mock import patch

from toolkit.preprocessing.handlers import GitHandler
from toolkit.preprocessing.ecos import Maven, Package

TEST_MAVEN_REPO_URL = "https://github.com/inversoft/prime-jwt/"
TEST_TRAVERSAL_PATH = "src/main/java/io/fusionauth/jwt/"


class TestMaven(unittest.TestCase):
    """Tests for maven ecosystem."""

    def test_find_pom_files(self):
        """Test MavenRepository `find_pom_files` method."""
        with GitHandler.clone(TEST_MAVEN_REPO_URL) as git:
            repo_dir = git.repository

            pom_files = Maven.find_pom_files(repo_dir)

            self.assertIsNotNone(pom_files)
            self.assertTrue(any(pom_files))
            self.assertTrue(all([f.endswith('.xml') for f in pom_files]))

            # try it with reverse search
            child_path = os.path.join(repo_dir, TEST_TRAVERSAL_PATH)
            pom_files = Maven.find_pom_files(child_path, topdown=False)

            self.assertIsNotNone(pom_files)
            self.assertTrue(any(pom_files))
            self.assertTrue(all([f.endswith('.xml') for f in pom_files]))

    def test_get_package_from_spec(self):
        """Test MavenRepository `get_package_from_spec` method."""
        with GitHandler.clone(TEST_MAVEN_REPO_URL) as git:
            repo_dir = git.repository

            pom_files = Maven.find_pom_files(repo_dir)
            assert any(pom_files)

            with open(pom_files[0], 'r') as pom_spec:
                package = Maven.get_package_from_spec(pom_spec)

                self.assertIsNotNone(package)
                self.assertTrue(isinstance(package, Package))
                self.assertTrue(
                    all(
                        [getattr(package, attr, None)]
                        for attr in ['name', 'owner', 'version', 'description']
                    )
                )

    @patch(target='toolkit.preprocessing.ecos.ElementTree.find')
    def test_get_package_from_spec_exception(self,
                                             mock_find):
        """Test `get_package_from_spec` with missing parent namespace."""
        with GitHandler.clone(TEST_MAVEN_REPO_URL) as git:
            repo_dir = git.repository

            pom_files = Maven.find_pom_files(repo_dir)
            assert any(pom_files)

            with open(pom_files[0], 'r') as pom_spec:
                # set up mock
                mock_find.return_value = None

                package = Maven.get_package_from_spec(pom_spec)

                self.assertIsNotNone(package)
                self.assertTrue(isinstance(package, Package))

                self.assertTrue(
                    all([
                        getattr(package, attr, None) is None
                        for attr in ['aid', 'gid', 'name',
                                     'owner', 'version', 'description']
                    ])
                )

    def test_find_packages(self):
        """Test MavenRepository `find_packages` method."""
        with GitHandler.clone(TEST_MAVEN_REPO_URL) as git:
            repo_dir = git.repository
            self.assertIsNotNone(repo_dir)

            packages = Maven.find_packages(repo_dir)

            self.assertIsNotNone(packages)
            self.assertTrue(any(packages))
            self.assertTrue(all([isinstance(f, Package) for f in packages]))
