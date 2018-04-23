"""Module containing ecosystem specific tooling."""

import os
import re


class Package(object):
    """Ecosystem invariant package base class."""

    def __init__(self,
                 ecosystem,
                 name=None,
                 owner=None,
                 version=None,
                 description=None,
                 licenses=None,
                 url=None):
        """Initialize package class with project details."""
        self._ecosystem = ecosystem
        self._name = name
        self._owner = owner
        self._version = version
        self._description = description
        self._licenses = licenses
        self._url = url

    @property
    def ecosystem(self):
        """Package ecosystem."""
        return self._ecosystem

    @property
    def name(self):
        """Package name."""
        return self._name

    @property
    def owner(self):
        """Package owner."""
        return self._owner

    @property
    def version(self):
        """Package version."""
        return self._version

    @property
    def description(self):
        """Package description."""
        return self._description

    @property
    def licenses(self):
        """Licenses used in the repository."""
        return self._licenses

    @property
    def url(self):
        """Url of the project."""
        return self._url

    def __str__(self):
        """Return string representation of the current instance."""
        return "[{eco}]: {proj}, version {ver}".format(
            eco=self.ecosystem,
            proj=self._name,
            ver=self._version
        )

    def __hash__(self):
        """Return hash of the current instance."""
        return hash(str(self))

    def __eq__(self, other):
        """Compare two package objects for equality."""
        return hash(other) == hash(self)

    def get_attributes(self, skip_none=False):
        """Get packages attribute dict."""
        return {
            k.strip('_'): v for k, v in self.__dict__.items()
            if (skip_none and v) or (not skip_none)
        }


class MavenPackage(Package):
    """Maven package class."""

    def __init__(self,
                 groupId: str,  # pylint: disable=invalid-name
                 artifactId: str,  # pylint: disable=invalid-name
                 *args, **kwargs):
        """Initialize maven package."""
        self._gid = groupId
        self._aid = artifactId

        super(MavenPackage, self).__init__(
            ecosystem='maven', *args, **kwargs
        )

    def __str__(self):
        """Return string represenation."""
        super_rep = super().__str__()
        rep = super_rep + ", gid {gid}, aid {aid}".format(
            gid=self.gid,
            aid=self.aid
        )

        return rep

    @property
    def gid(self):
        """Package group id."""
        return self._gid

    @property
    def aid(self):
        """Package artifact id."""
        return self._aid


class Maven(object):
    """Maven ecosystem class.

    The class acts as a namespace for maven-specific operations.
    """

    @staticmethod
    def find_packages(path=None, recurse=True, topdown=True):
        """Find project packages belonging to the specific ecosystem.

        :param path: str, parent directory
        :param recurse: whether to recurse child directories
        :param topdown: proceed traversal from child to parent directory
        """
        # find pom.xml files
        pom_files = Maven.find_pom_files(
            path,
            recurse=recurse,
            topdown=topdown
        )

        packages = list()
        for pom_file in pom_files:
            with open(pom_file, 'r') as pom_spec:
                packages.append(Maven.get_package_from_spec(pom_spec))

        return packages

    @staticmethod
    def find_pom_files(path: str, recurse=True, topdown=True):
        """Find pom.xml files in the given path."""
        # validate the path
        pom_files = list()
        if topdown:
            for root, walkdir, walkfiles in os.walk(path):
                pom_files.extend([
                    os.path.join(root, f)
                    for f in walkfiles if re.match(r"[pP]om.xml", f)
                ])
                if not recurse and len(walkdir) > 1:
                    del walkdir[:]
        else:
            # traverse to parent directories - do not recurse back down
            root = path
            while os.path.exists(root):
                pom_files.extend([
                    os.path.join(root, f) for f in os.listdir(root)
                    if re.match(r"[pP]om.xml", f)
                ])
                root = root[:root.rfind(os.sep)]

        return pom_files

    @staticmethod
    def get_package_from_spec(pom_file) -> Package:
        """Create MavenPackage object from pom.xml file."""
        from xml.etree import ElementTree
        tree = ElementTree.parse(pom_file)
        root = tree.getroot()

        # get xml namespace
        namespace = re.search(r"({.*})(.*)", root.tag)
        if namespace is None:
            raise ValueError("namespace was not found in the xml file")
        namespace = namespace.group(1)

        attributes = [
            'groupId', 'artifactId', 'name',
            'version', 'description', 'url'
        ]

        package_spec = dict()
        for attr in attributes:
            try:
                val = tree.find(namespace + attr).text
            except AttributeError:
                val = None
            package_spec[attr] = val

        package = MavenPackage(
            **package_spec
        )

        return package
