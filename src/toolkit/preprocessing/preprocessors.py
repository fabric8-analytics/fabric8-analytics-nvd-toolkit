"""This module contains preprocessors.

Those preprocessors are transformers which implement `transform` methods
called by sklearn pipeline and can be integrated with it.
"""

import re
import typing

import nltk
import nltk.corpus as corpus
import numpy as np

from collections import namedtuple

from sklearn.base import TransformerMixin

from toolkit.preprocessing import GitHubHandler
from toolkit.transformers import Hook
from toolkit import utils


class NVDFeedPreprocessor(TransformerMixin):
    """Preprocessor collecting relevant data from NVD feed.

    :param attributes: list or ndarray of attributes to extract,

        While `transform`, each attribute will be gathered from each element
        of the list fed to the method and return in the resulting tuple.

        If None, ('cve_id', 'references', 'description') are selected by default

    :param handler: data handler, by default GitHubHandler (ATM the only one supported)
    :param skip_duplicity: bool, whether to allow duplicit attributes

        Handler provides default handler properties which it extracts from
        the data, if any of attr in `attributes` intersects with those properties,
        it raises an error if `skip_duplicit` is False (default).
    """

    def __init__(self,
                 attributes: typing.Union[list, np.ndarray] = None,
                 handler=None,
                 skip_duplicity=False):

        if attributes and not any(isinstance(attributes, t) for t in (list, np.ndarray)):
            raise TypeError(
                "Argument `attributes` expected to be of type `{}`, got `{}`"
                .format(typing.Union[list, np.ndarray], type(attributes))
            )

        self._handler = handler
        if handler is None:
            self._handler = GitHubHandler

        attributes = tuple(attributes) if attributes else ('cve_id', 'references', 'description')
        for attr in attributes:
            if attr in self._handler.default_properties and not skip_duplicity:
                raise ValueError("Attribute `{}` is already present in handlers "
                                 "default properties.".format(attr))

        self._cve_attributes = attributes
        # leave only those not present in handlers default properties
        self._cve_attributes = tuple([
            attr for attr in self._cve_attributes
            if attr not in self._handler.default_properties
        ])

        self._use_filter = False

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self,
            X,  # pylint: disable=invalid-name,unused-argument
            y=None,  # pylint: disable=unused-argument
            **fit_params):
        """Auxiliary method to enable pipeline functionality.

        :param fit_params: optional parameters

            use_filter: whether to filter the data by handler, default True
                        *NOTE*: This argument is especially important when labeling
                                the data follows after this preprocessor
        """
        self._use_filter = fit_params.get('use_filter', True)

        return self

    # noinspection PyPep8Naming
    def transform(self, X: typing.Union[list, np.ndarray]):  # pylint: disable=invalid-name
        """Apply transformation to each element in `cves`.

        This transformation outputs list of tuples

        :param X:
            list or ndarray, each element is assumed to be of type "nvdlib.model.CVE"
            or an object implementing attributes given in `attr_list`

        :returns: list of shape (len(x), len(attr_list))
            Each element of the resulting list is a namedtuple of cve attributes
        """
        if self._use_filter:
            # filter the cves by the given handler
            cve_iter = self._filter_by_handler(X)
        else:
            cve_iter = iter(X)

        return [
            self._get_cve_attributes(t) for t in cve_iter
        ]

    def _filter_by_handler(self, cves: typing.Union[list, np.ndarray]) -> typing.Iterator:
        """Filter the given elements by specified handler.

        NOTE: The only supported handler ATM is `GitHubHandler`, hence
        cves are filtered by reference pattern.

        :returns: list of tuples of type (cve, reference)
        """
        # TODO: modify this approach by creating handler-specific filter method
        #   such method would take CVE and returned bool whether
        #   it could operate on it or not
        filtered_cves = filter(
            lambda cve: utils.has_reference(cve, pattern=self._handler.pattern),
            cves
        )

        return filtered_cves

    def _get_cve_attributes(self, cve):
        """Get the selected attributes from the CVE object."""
        # initialize handler
        ref = utils.get_reference(cve, pattern=self._handler.pattern)

        data = list()
        if ref is not None:
            handler = self._handler(url=ref)

            # attribute creator
            Series = namedtuple(
                'Series',
                handler.default_properties + self._cve_attributes
            )

            # initialize with handlers default data
            for prop in handler.default_properties:
                attr = getattr(handler, prop)
                if attr is not None:
                    data.append(attr)

        elif not self._use_filter:
            # if the use_filter was False, the handler may have not extracted anything
            Series = namedtuple(
                'Series',
                self._cve_attributes
            )

        else:
            # something is wrong, since the cve passed the filter but handler
            # did not extracted any attributes
            raise ValueError("Unexpected error occurred. `use_filter=True`, but"
                             " reference `{}` did not match handlers pattern."
                             .format(ref))

        data.extend([
            getattr(cve, attr) for attr in self._cve_attributes
        ])

        return Series(*data)


class LabelPreprocessor(TransformerMixin):
    """Preprocessor implementing `tranform` method for scikit pipelines.

    This preprocessor assign labels to the given set by creating.

    :param feed_attributes: list, attributes to be extracted from the `X`
    while `transform` or `fit_transform` call and fed to the hook.

    NOTE: attributes of `X` are extracted by `getattr` function, make
    sure that the `X` implements __get__ method.

    :param hook: Hook, hook to be called on the each element of `X`
    while `fit_transform` call and will be fed the element of `X` (
    which should be a `namedtuple`).

    :param output_attributes: list, attributes to be returned

        By default output_attributes are the same as feed_attributes.
    """

    def __init__(self,
                 hook: "Hook",
                 feed_attributes: list,
                 output_attributes: list = None,
                 allow_nan_labels=False):

        if not isinstance(feed_attributes, typing.Iterable):
            raise TypeError("Argument `feed_attributes` expected to be of type `{}`,"
                            " got `{}`".format(typing.Iterable, type(feed_attributes)))

        self._feed_attributes = feed_attributes
        self._output_attributes = output_attributes or self._feed_attributes

        if not isinstance(self._output_attributes, typing.Iterable):
            raise TypeError("Argument `output_attributes` expected to be of type `{}`,"
                            " got `{}`".format(typing.Iterable, type(self._output_attributes)))

        if not isinstance(hook, Hook):
            raise TypeError("Argument `hook` expected to be of type `{}`, got `{}"
                            .format(Hook, type(hook)))
        self._hook = hook
        self._labels = None

        self._allow_nan_labels = allow_nan_labels

    @property
    def labels(self):
        """Processed labels."""
        return self._labels

    # noinspection PyPep8Naming
    def fit(self, X: typing.Union[list, np.ndarray], y=None, **fitparams):  # pylint: disable=invalid-name
        """Fit the preprocessor to the given data."""
        Series = namedtuple('Attributes', field_names=self._feed_attributes)

        self._labels = [None] * len(X)
        for i, x in enumerate(X):
            # noinspection PyTypeChecker
            self._labels[i] = self._hook(*Series(
                *tuple(getattr(x, attr) for attr in self._feed_attributes)
            ))

        return self

    # noinspection PyPep8Naming
    def transform(self,
                  X: typing.Union[list, np.ndarray]):  # pylint: disable=invalid-name
        """Transform the data provided in `X` by extracting output attributes."""
        def allow_label(l):
            if l is None:
                return self._allow_nan_labels

            return True

        Series = namedtuple('Series', [*self._output_attributes, 'label'])

        if self._labels is None:
            self._labels = [None] * len(X)

        # noinspection PyTypeChecker
        result = [
            [getattr(x, attr) for attr in self._output_attributes]
            for x, label in zip(X, self._labels) if allow_label(label)
        ]

        # create relevant labels
        labels = [label for label in self._labels if allow_label(label)]

        # in case no label was found, return Series of empty values
        if not result:
            result = [[]] * len(self._output_attributes)

        assert len(result) == len(labels)

        # assign label to each set of attributes
        for i in range(len(result)):
            result[i].append(labels[i])

        return [
            Series(*r) for r in result
        ]


class NLTKPreprocessor(TransformerMixin):
    """Preprocessor implementing `transform` method for scikit pipelines.

    This preprocessor performs tokenization, stemming and lemmatization
    by default. Processors used for these operations are customizable.

    Other text processing operations are not mandatory and can be optimized
    by user.

        :param lemmatizer: nltk lemmatizer, defaults to nltk.WordNetLemmatizer
        :param stemmer: nltk stemmer, defaults to nltk.SnowballStemmer
        :param tokenizer: nltk tokenizer, defaults to nltk.TreebankWordTokenizer
        :param stopwords: bool, whether to strip stopwords
        :param tag_dict: dictionary of (pattern, correct_tag) used for tag correction

            If provided, each tag is matched to a pattern in this dictionary
            and corrected, if applicable.
        :param lower: bool, whether to transform tokens to lowercase
        :param strip: bool, whether to strip tokens
    """

    def __init__(self,
                 feed_attributes: list = None,
                 output_attributes: list = None,
                 lemmatizer=None,
                 stemmer=None,
                 tokenizer=None,
                 stopwords=False,
                 tag_dict=None,
                 token_dict=None,
                 lower=False,
                 strip=False,
                 lang='english'):
        self._feed_attributes = feed_attributes or []
        self._output_attributes = output_attributes or []

        if not isinstance(self._feed_attributes, typing.Iterable):
            raise TypeError("Argument `feed_attributes` expected to be of type `{}`,"
                            " got `{}`".format(typing.Iterable, type(self._feed_attributes)))

        if not isinstance(self._output_attributes, typing.Iterable):
            raise TypeError("Argument `output_attributes` expected to be of type `{}`,"
                            " got `{}`".format(typing.Iterable, type(self._output_attributes)))

        self._tokenizer = tokenizer or nltk.TreebankWordTokenizer()
        self._lemmatizer = lemmatizer  # or nltk.WordNetLemmatizer()
        self._stemmer = stemmer  # or nltk.SnowballStemmer(language=lang)

        self._lower = lower
        self._strip = strip
        self._stopwords = corpus.stopwords.words(lang) if stopwords else set()

        self._lang = lang
        self._tag_dict = tag_dict or dict()
        self._token_dict = token_dict or dict()

        # update the token dict with a default version pattern
        self._token_dict.update({
            u'(\d[.]?)+[-_]?(\w)*': '<VERSION>'
        })

        # prototyped
        self._y = None

    # noinspection PyPep8Naming
    @staticmethod
    def inverse_transform(
            X: typing.Union[list, np.ndarray]) -> list:  # pylint: disable=invalid-name
        """Inverse operation to the `transform` method.

        Returns list of shape (len(X),) with the tokens stored in X.

        Note that this does not return the original data provided
        to `transform` method, since lemmatization and stemming
        are not reversible operations and for memory sake, lowercase changes
        are not stored in memory either.
        """
        return [
            x[0] for x in X
        ]

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self,
            X: typing.Iterable,  # pylint: disable=invalid-name
            y=None,  # pylint: disable=unused-argument
            **fit_params):
        """Fits the preprocessor to the given data.

        :param X: Iterable, each element should be a string to be tokenized
        :param y: Iterable, labels for each element in X (must be the same
        length as `X`)
        :param fit_params: kwargs, optional arguments to be used during fitting
        and transformation

            feed_attributes: # TODO
            output_attributes:
        """
        # allow defining attributes in the fit function as well, since
        # user might want to specify it directly in the pipeline
        self._feed_attributes = fit_params.get('feed_attributes', []) or self._feed_attributes
        self._output_attributes = fit_params.get('output_attributes', []) or self._output_attributes

        if not isinstance(self._feed_attributes, typing.Iterable):
            raise TypeError("Argument `feed_attributes` expected to be of type `{}`,"
                            " got `{}`".format(typing.Iterable, type(self._feed_attributes)))

        if not isinstance(self._output_attributes, typing.Iterable):
            raise TypeError("Argument `output_attributes` expected to be of type `{}`,"
                            " got `{}`".format(typing.Iterable, type(self._output_attributes)))
        if y is not None:
            assert len(list(X)) == len(list(y)), "len(X) != len(y)"

            self._output_attributes = [*self._output_attributes, 'labels']

        self._y = y

        return self

    # noinspection PyPep8Naming
    def transform(self,
                  X: typing.Iterable) -> typing.Any:  # pylint: disable=invalid-name
        """Fit to each element in X.

        This transformation outputs list of the shape (len(X), 2)
        where each element of the list is a tuple of (token, tag).

        :param X: Iterable, each element should be a string to be tokenized

        :returns: namedtuple of the same shape as `X` if `y` is None, otherwise
        adds one dimension for `y` elements
        """
        if self._feed_attributes:
            X_feed = [
                getattr(x, attr) for attr in self._feed_attributes
                for x in X
            ]

            intermediate_result = [
                self.tokenize(x) for x in X_feed
            ]

        else:
            intermediate_result = [
                self.tokenize(sent) for sent in X
            ]

        Series = namedtuple('Series', ['values'] + self._output_attributes)

        additional_output = [
            getattr(x, attr) for attr in self._output_attributes
            for x in X
        ]

        if additional_output:
            result = [
                Series(res, adds) for res, adds in zip(intermediate_result, additional_output)
            ]
        else:
            result = intermediate_result

        return result

    def tokenize(self, stream: str):
        """Perform tokenization of each sentence given in the list."""
        if not isinstance(stream, str):
            raise TypeError("Tokenization process expects input of type `{}`,"
                            "got `{}`".format(str, type(stream)))

        tokenized = self._tokenizer.tokenize(stream)

        result = list()
        for token, tag in nltk.pos_tag(tokenized, tagset='universal'):
            # Check and correct (if applicable) the tag against given patterns
            for pattern, correction in self._tag_dict.items():
                if re.match(pattern, tag):
                    tag = correction
                    # do not allow ambiguity of tags (assume user took care of this)
                    break

            # Check and correct (if applicable) the token against given patterns
            for pattern, correction in self._token_dict.items():
                if re.match(pattern, token):
                    token = correction
                    # do not allow ambiguity of tags (assume user took care of this)
                    break

            # Apply pre-processing to each token and tag
            token = token.lower() if self._lower else token
            token = token.strip() if self._strip else token

            # If stop word, ignore token and continue
            if token in self._stopwords:
                continue

            # Punctuation will not be yielded
            if tag == '.':
                continue

            if self._stemmer:
                token = self.stem(token)

            if self._lemmatizer:
                try:
                    token = self.lemmatize(token, tag)
                except KeyError:
                    # skip if the token can not be lemmatized (eg. tags 'PRT')
                    continue

            result.append((token, tag))

        return result

    def stem(self, token: str):
        """Stem the word and return the stem."""
        return self._stemmer.stem(token)

    def lemmatize(self, token: str, tag: str):
        """Lemmatize the token based on its tag and return the lemma."""
        # The lemmatizer expects the `pos` argument to be first letter
        # of positional tag of the universal set (which we use by default)
        return self._lemmatizer.lemmatize(token, pos=tag[0].lower())
