"""Module containing utilities for processing data from NVD feed."""

import argparse
import re
import typing

from toolkit.transformers.hooks import Hook


class BooleanAction(argparse.Action):
    """Argparse function to handle --flag and --no-flag arguments."""

    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        """Initialize BooleanAction."""
        super(BooleanAction, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, _parser, namespace, values, option_string=None):
        """Call BooleanAction instance."""
        setattr(namespace, self.dest,
                False if option_string.startswith('--no') else True)


# noinspection PyPep8Naming
class classproperty(object):  # pylint: disable=invalid-name
    """Class property implementation.

    Usage: Same as a @property decorator for instance properties,
    @classproperty can be used for class members.

    TODO: solve the setter and deleter by adding a metaclass
    """

    def __init__(self, fget=None):  # only getter needed atm
        """Initialize classproperty."""
        if fget is not None and hasattr(fget, '__doc__'):
            doc = fget.__doc__
        else:
            doc = None

        self.__get = fget
        self.__doc = doc

    def __get__(self, _instance, cls=None):
        """Return class property."""
        if cls is None:
            raise AttributeError("instance type not filled")
        if self.__get is None:
            raise AttributeError("unreadable attribute")
        return self.__get(cls)

    def __set__(self, _inst, _value):
        """Set class property - DISABLED."""
        raise AttributeError("can't set attribute")

    def __delete__(self, _inst):
        """Delete class property - DISABLED."""
        raise AttributeError("can't delete attribute")


def check_attributes(*attributes: typing.Any, allow_none=True):
    """Check whether `attribute_list` contains valid elements.

    :raises: TypeError

    """
    for attr_list in attributes:
        if attr_list is None and allow_none:
            continue

        if isinstance(attr_list, list) and all([
            isinstance(attr, str) for attr in attr_list
        ]):
            continue

        else:
            raise TypeError("Argument `feed_attributes` expected to be of type `{}`,"
                            " got `{}`".format(typing.List[str], type(attr_list)))


def has_reference(cve, url=None, pattern=None) -> bool:
    """Report reference existence based on {url, pattern} given.

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
    """Return reference (if exists) based on {url, pattern}.

    If multiple references are present, only the first encountered is returned.

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
    """Find the `word` in the `stream`.

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


def nvd_to_dataframe(cve_list: typing.Iterable,
                     handler=None):
    """Create a pandas DataFrame from nvdlib.NVD object.

    NOTE: This function requires `pandas` package to be installed
    (which is not a requirement for the standard installation or
    for production usage)

    :param cve_list: list of nvdlib.model.CVE objects
    :param handler: handler for cves, right now only GitHubHandler is supported

    :return: pandas.DataFrame
    """
    from pandas import DataFrame, Series

    data = list()
    projects = set()
    # type: typing.Tuple[str(username), str(project)]

    languages = set()
    project_langs = dict()
    # type: typing.Dict[tuple(username, project), dict{lang: int}]

    for cve in cve_list:
        # Get reference supported by the handler to gather information
        # about the CVE
        username = project = None
        handle = None

        if handler is not None:
            ref = get_reference(cve, pattern=handler.pattern)
            if ref is None:
                # does not match handlers pattern
                continue

            # initialize handler
            handle = handler(url=ref)

            username, project = handle.user, handle.project
            if (username, project) in projects:
                # Multiple CVEs are possible for a project, # but not important
                # to cover for package prediction
                continue
            projects.add((username, project))

            # query GitHub API for project languages
            cve_languages = handle.languages()
            if not cve_languages:
                cve_languages = dict()

            languages |= set(cve_languages)
            project_langs[(username, project)] = cve_languages

        data.append([
            cve.cve_id,
            cve.description,
            username, project,
            getattr(handle, 'repository', None)
        ])

    columns = ['id', 'description', 'username', 'project', 'url']

    # Create list of series representing each language and their data
    lang_data = list()
    for lang in languages:
        # create series of languages and their byte counts per project
        series_data = [project_langs[(u, p)].get(lang, 0) for u, p, _, __ in data]
        lang_series = Series(data=series_data, name=lang)

        lang_data.append(lang_series)

    # finalize and return the dataframe
    return DataFrame(data=data, columns=columns).assign(**{s.name: s.values for s in lang_data})


def clear(func):
    """Decorate with cleanup before function call."""
    # noinspection PyUnusedLocal,PyUnusedLocal
    def wrapper(*args, **kwargs):  # pylint: disable=unused-argument
        """Wrap inner function."""
        # perform cleanup
        Hook.clear_current_instances()
        ret_values = None

        # run the function
        try:
            ret_values = func(*args, **kwargs)
        except Exception as exc:
            raise exc
        finally:
            # cleanup again
            Hook.clear_current_instances()

        return ret_values

    return wrapper
