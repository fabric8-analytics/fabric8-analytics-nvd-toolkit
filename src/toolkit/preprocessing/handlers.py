"""Module containing handlers for source control management systems."""

import json
import re
import urllib3

from toolkit import config
from toolkit.utils import classproperty

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class StatusError(Exception):
    """Custom exception returned by Git Hub API."""

    def __init__(self, status: int):
        """Initialize custom exception."""
        msg = "Response status returned: `%d`" % status

        super(StatusError, self).__init__(msg)


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
        self._src_url = self.strip_src_url(url)
        self._user, self._project = self.get_user_project(self._src_url)
        self._http = urllib3.PoolManager()

        # prototyped variables
        self._languages = None

    # noinspection PyMethodParameters
    @classproperty
    def pattern(cls):
        """Reference pattern handled by the handler."""
        return cls.__URL_BASE_PATTERN

    # noinspection PyMethodParameters
    @classproperty
    def default_properties(cls):
        """Default handler's properties."""
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
