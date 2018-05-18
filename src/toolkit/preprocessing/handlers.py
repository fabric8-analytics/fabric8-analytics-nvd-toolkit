"""Module containing handlers for source control management systems."""

import json
import os
import sys
import re
import shlex
import subprocess
import tempfile
import urllib3

from toolkit import config
from toolkit.utils import classproperty

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class StatusError(Exception):
    """Custom exception returned by Git Hub API."""

    def __init__(self, status: int, *args):
        """Initialize custom exception."""
        msg = "Response status returned: `%d`" % int(status)

        super(StatusError, self).__init__(msg, *args)


class GitHubHandler(object):
    """
    The handler manages Git Hub repository.

    Strips its source directory and handles access to Git Hub API
    making it easy to interact with the repository.

    :param url: str, url of any Git Hub repository or blob
    """

    __URL_BASE_PATTERN = r"http[s]://github.com/([\w-]+)/([\w-]+[.]*[\w-]*)"
    __API_URL = r"https://api.github.com/repos/{user}/{project}/languages"
    __DEFAULT_PROPERTIES = ('user', 'project', 'repository')

    def __init__(self, url: str = None):
        """Initialize GitHubHandler."""
        self._src_url = self.strip_src_url(url or "")
        self._user, self._project = self.get_user_project(self._src_url)
        self._http = urllib3.PoolManager()

        # prototyped variables
        self._languages = None

    # noinspection PyMethodParameters
    @classproperty
    def pattern(cls):  # pylint: disable=no-self-argument
        """Get reference pattern handled by the handler."""
        return cls.__URL_BASE_PATTERN

    # noinspection PyMethodParameters
    @classproperty
    def default_properties(cls):  # pylint: disable=no-self-argument
        """Get default handler's properties."""
        return cls.__DEFAULT_PROPERTIES

    @property
    def repository(self):
        """Git Hub repository source url."""
        return self._src_url

    @property
    def user(self):
        """Git Hub repository owner."""
        return self._user

    @property
    def project(self):
        """Git Hub project name."""
        return self._project

    @property
    def languages(self):
        """Languages used by the project."""
        if not self._languages:
            # query API only the first time
            self._languages = self.get_languages()
        return self._languages

    def strip_src_url(self, url: str) -> str:
        """Strip the source url from a given url.

        :param url: str, url to be stripped

        :returns: str, source url
        :raises: ValueError if url does not match handler's pattern
        """
        strip_url = re.search(self.__URL_BASE_PATTERN, url)

        if not strip_url:
            raise ValueError("url `%s` does not match handler's base pattern" % url)

        # noinspection PyUnresolvedReferences
        return strip_url[0]

    @staticmethod
    def get_user_project(src_url: str) -> tuple:
        """Split the source url and extracts username and project name.

        :param src_url: url to the source repository of a project

        :returns: tuple (username, project)
        """
        # TODO: improve the splitting with regex .. this way it misses some projects
        gh_user, gh_project = src_url.rsplit(r'/', 2)[-2:]

        return gh_user, gh_project

    def get_languages(self) -> dict:
        """
        Query Git Hub API languages used for the given user/project.

        Note: Git Hub will most likely require OAUTH_TOKEN specified,
        provide your token via environment variable OAUTH_TOKEN.

        :returns: dict, {"str"language: "int"bytes_of_code} or None
        :raises: HTPPError on wrong response status
        """
        request_url = self.__API_URL.format(user=self._user,
                                            project=self._project)
        response = self._http.request('GET', request_url,
                                      headers=config.HEADERS)

        # Handle limits and wrong responses
        if response.status > 205:
            raise StatusError(status=response.status)

        return json.loads(response.data)


class GitHandler(object):
    """The handler manages local git repository."""

    def __init__(self, path: str):
        """Initialize GitHandler."""
        if not os.path.isdir(path):
            raise FileNotFoundError("path `%s` is not a directory." % path)

        self._cwd = os.getcwd()
        self._chdir = os.path.abspath(path)

        # check directory correctness only (status raises error if incorrect)
        _, __ = self.status

    def __enter__(self):
        """Enter the git context manager."""
        os.chdir(self._chdir)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the git context manager."""
        os.chdir(self._cwd)

    @property
    def repository(self):
        """Return change root directory, ie. main git repository."""
        return self._chdir

    @property
    def status(self):
        """Return git status of the current repository."""
        return self.exec_cmd("git status", chdir=self._chdir)

    @classmethod
    def clone(cls, url: str):
        """Initialize handler from a repository url."""
        tmp_dir = tempfile.mkdtemp(prefix='nvd-toolkit_', suffix='_clone')

        clone_cmd = "git clone {url} {dest}".format(
            url=url,
            dest=tmp_dir
        )

        _, __ = cls.exec_cmd(clone_cmd)

        return cls(path=tmp_dir)

    def get_modified_files(self, commits: list) -> list:
        """Get modified files by a commit hash."""
        if not all([isinstance(c, str) for c in commits]):
            raise TypeError("Each commit in  `commits` expected"
                            "to be of type str, got `{}`"
                            .format(type(commits)))

        mod_files = list()
        for commit in commits:
            stdout, _ = self.exec_cmd(
                cmd='git diff-tree --no-commit-id --name-only -r %s' % commit,
                chdir=self._chdir
            )

            mod_files.extend([
                os.path.join(self._chdir, f) for f in stdout.split()
            ])

        return mod_files

    @staticmethod
    def exec_cmd(cmd, chdir=None):
        """Execute git command.

        :param cmd: command to execute
        :param chdir: change directory to use as current working dir

        :returns: tuple (stdout, stderr), output of the command
        """
        cwd = os.getcwd()
        if chdir is not None:
            os.chdir(chdir)

        # ensure every argument is shell-quoted
        # to prevent accidental shell injection
        cmd = [
            shlex.quote(arg) for arg in shlex.split(cmd)
        ]

        if cmd[0].lower() != 'git':
            raise ValueError("Invalid command `{}`, expected `git`"
                             .format(cmd[0]))

        pcs = subprocess.Popen(
            cmd,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = pcs.communicate()
        ec = pcs.wait()

        if ec != 0:
            print(stderr, file=sys.stderr)
            raise StatusError(ec, stderr)

        # change the cwd back to the original one
        os.chdir(cwd)

        return stdout, stderr
