"""Module containing utilities for processing and extraction
of data from NVD feed."""

import re
import typing


# noinspection PyPep8Naming
class classproperty(object):  # pylint: disable=invalid-name
    """Class property implementation.

    Usage: Same as a @property decorator for instance properties,
    @classproperty can be used for class members.

    TODO: solve the setter and deleter by adding a metaclass
    """

    def __init__(self, fget=None):  # only getter needed atm
        if fget is not None and hasattr(fget, '__doc__'):
            doc = fget.__doc__
        else:
            doc = None

        self.__get = fget
        self.__doc = doc

    def __get__(self, instance, cls=None):
        if cls is None:
            raise AttributeError("instance type not filled")
        if self.__get is None:
            raise AttributeError("unreadable attribute")
        return self.__get(cls)

    def __set__(self, inst, value):
        raise AttributeError("can't set attribute")

    def __delete__(self, inst):
        raise AttributeError("can't delete attribute")


def has_reference(cve, url=None, pattern=None) -> bool:
    """Iterate over references of the given CVE and reports existence
    based on {url, pattern} given.

    :param cve: CVE object whose `references` attribute is of type Iterable[str]
    :param url: str, if specified, requires exact match of cve reference
    with the given `url` argument and takes precedence over `pattern` argument
    :param pattern: str, regex expression used for comparison,
    `re.search()` method is used

    :returns: bool, whether cve references found in one of {url, pattern}
    """
    assert any([url, pattern]), "either `url` or `pattern` must be provided"
    assert hasattr(cve, 'references'), "cve object `%s` has no attribute `references`" % cve
    assert isinstance(cve.references, typing.Iterable), "`cve.references` is not `Iterable`, " \
                                                        "got: %s" % type(cve.references)
    if url:
        for ref in cve.references:
            if url == ref:
                return True
    else:
        for ref in cve.references:
            if re.search(pattern, ref):
                return True

    return False


def get_reference(cve, url=None, pattern=None) -> typing.Union[str, None]:
    """Iterate over references of the given CVE and returns reference (if exists)
    based on {url, pattern} given. If multiple references are present, only the first
    encountered is returned.

    :param cve: CVE object whose `references` attribute is an iterable of str references
    :param url: str, if specified, requires exact match of cve reference with the given `url`
    argument and takes precedence over `pattern` argument
    :param pattern: str, regex expression used for comparison, `re.search()` method is used

    :returns: str, cve reference matching one of {url, pattern}
    """
    assert any([url, pattern]), "either `url` or `pattern` must be provided"
    assert hasattr(cve, 'references'), "cve object `%s` has no attribute `references`" % cve
    assert isinstance(cve.references, typing.Iterable), "`cve.references` is not `Iterable`, " \
                                                        "got: %s" % type(cve.references)
    if url:
        for ref in cve.references:
            if url == ref:
                return ref
    else:
        for ref in cve.references:
            if re.search(pattern, ref):
                return ref

    return None


# this function is meant to be used as a hook for LabelPreprocessor
def find_(word, stream, ignore_case=True):
    """Trys to find the `word` in the `stream`.

    :param word: str, word or pattern to be searched in stream
    :param stream: str, stream to be searched in
    :param ignore_case: whether to ignore the case
    :returns: the corresponding `word` (could be of different case
    if `ignore_case` is specified) from the stream
    """
    match = re.search(word,
                      stream,
                      flags=re.IGNORECASE if ignore_case else 0)

    if match is not None:
        match = match.group(0)

    return match


def nvd_to_dataframe(cve_list: list,
                     handler):
    """Creates a pandas DataFrame from nvdlib.NVD object.

    NOTE: This function requires `pandas` package to be installed
    (which is not a requirement for the standard installation or
    for production usage)

    :param cve_list: list of nvdlib.model.CVE objects
    :param handler: handler for cves, right now only GitHubHandler is supported

    :return: pandas.DataFrame
    """
    from pandas import DataFrame, Series

    projects = set()
    "set of tuples ('str'username, 'str'project)"
    languages = set()
    "set of languages present in all projects"
    project_langs = dict()
    "dict of ('tuple'(username, project), 'dict'(lang, byte))"

    data = list()
    for cve in cve_list:
        # Get reference supported by the handler to gather information
        # about the CVE
        ref = get_reference(cve, pattern=handler.pattern)
        if ref is None:
            continue

        # initialize handler
        handle = handler(url=ref)

        username, project = handle.user, handle.project
        if (username, project) in projects:
            # Multiple CVEs are possible for a project, but not important to cover for package prediction
            continue
        projects.add((username, project))

        # query GitHub API for project languages
        cve_languages = handle.languages()
        if not cve_languages:
            cve_languages = dict()

        languages |= set(cve_languages)
        project_langs[(username, project)] = cve_languages

        data.append([username, project, handle.repository])

    columns = ['username', 'project', 'url']

    # Create list of series representing each language and their data
    lang_data = list()
    for lang in languages:
        # create series of languages and their byte counts per project
        series_data = [project_langs[(u, p)].get(lang, 0) for u, p, _, __ in data]
        lang_series = Series(data=series_data, name=lang)

        lang_data.append(lang_series)

    # finalize and return the dataframe
    return DataFrame(data=data, columns=columns).assign(**{s.name: s.values for s in lang_data})
