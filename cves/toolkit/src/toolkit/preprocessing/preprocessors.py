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
from toolkit.pipeline import Hook
from toolkit import utils


class NVDFeedPreprocessor(TransformerMixin):
    """Preprocessor collecting relevant data from NVD feed.

    :param attributes: Iterable of attributes to extract,

        While `fit`, each attribute will be gathered from each element
        of the list fed to the method and return in the resulting tuple.

        If None, ('cve_id', 'references', 'description') are selected by default

    :param handler: data handler, by default GitHubHandler (ATM the only one supported)
    :param skip_duplicity: bool, whether to allow duplicit attributes

        Handler provides default handler properties which it extracts from
        the data, if any of attr in `attributes` intersects with those properties,
        it raises an error if `skip_duplicit` is False (default).
    """

    def __init__(self,
                 attributes: typing.Iterable = None,
                 handler=GitHubHandler,
                 skip_duplicity=False):
        if attributes and not isinstance(attributes, typing.Iterable):
            raise TypeError(
                "Argument `attributes` expected to be of type `{}`, got `{}`"
                .format(typing.Iterable, type(attributes))
            )

        attributes = tuple(attributes) if attributes else ('cve_id', 'references', 'description')
        for attr in attributes:
            if attr in handler.default_properties and not skip_duplicity:
                raise ValueError("Attribute `{}` is already present in handlers "
                                 "default properties.".format(attr))

        self._cve_attributes = attributes
        # leave only those not present in handlers default properties
        self._cve_attributes = tuple([
            attr for attr in self._cve_attributes
            if attr not in handler.default_properties
        ])

        self._handler = handler

    # noinspection PyPep8Naming
    def transform(self, X: typing.Iterable):  # pylint: disable=invalid-name
        """Apply transformation to each element in `cves`.

        This transformation outputs list of tuples

        :param X:
            Iterable, each element is assumed to be of type "nvdlib.model.CVE"
            or an object implementing attributes given in `attr_list`

        :returns: list of shape (len(x), len(attr_list))
            Each element of the resulting list is a namedtuple of cve attributes
        """

        # filter the cves by the given handler
        cve_tuples = self._filter_by_handler(X)

        return [
            self._get_cve_attributes(t) for t in cve_tuples
        ]

    def _filter_by_handler(self, cves: typing.Iterable) -> typing.List[tuple]:
        """Filter the given elements by specified handler.

        NOTE: The only supported handler ATM is `GitHubHandler`, hence
        cves are filtered by reference pattern.

        :returns: list of tuples of type (cve, reference)
        """
        filtered_cves = list()
        for cve in cves:
            # TODO: modify this approach by creating handler-specific filter method
            #   such method would take CVE and returned bool whether
            #   it could operate on it or not
            ref = utils.get_reference(cve, pattern=self._handler.pattern)
            if ref is None:
                continue

            # include the ref in the tuple instead iterating over
            # the references later again
            filtered_cves.append((cve, ref))

        return filtered_cves

    def _get_cve_attributes(self, cve_tuple):
        """Get the selected attributes from the CVE object."""
        cve, ref = cve_tuple

        # initialize handler
        handler = self._handler(url=ref)

        # attribute creator
        Attributes = namedtuple(
            'Attributes',
            handler.default_properties + self._cve_attributes
        )

        # initialize with handlers default data
        data = list()
        for prop in handler.default_properties:
            attr = getattr(handler, prop, None)
            if attr is not None:
                data.append(attr)

        data.extend([
            getattr(cve, attr, None) for attr in self._cve_attributes
        ])

        return Attributes(*data)


class LabelPreprocessor(TransformerMixin):
    """Preprocessor implementing `tranform` method for scikit pipelines.

    This preprocessor assign labels to the given set by creating.

    :param attributes: list, attributes to be extracted from the `X`
    while `fit` or `fit_transform` call and fed to the hook.

    NOTE: attributes of `X` are extracted by `getattr` function, make
    sure that the `X` implements __get__ method.

    :param hook: Hook, hook to be called on the each element of `X`
    while `fit_transform` call and will be fed the element of `X` (
    which should be a `namedtuple`).
    """

    def __init__(self, attributes: list, hook: "Hook"):
        self._attributes = attributes
        if not isinstance(hook, Hook):
            raise TypeError("Argument `hook` expected to be of type `{}`, got `{}"
                            .format(Hook, type(hook)))
        self._hook = hook

    # noinspection PyPep8Naming
    def fit(self,
            X: typing.Iterable):  # pylint: disable=invalid-name
        """Fit the data provided in `X` by extracting attributes specified
        while initializaiton."""
        Attribute = namedtuple('Attributes', field_names=self._attributes)

        transformed = list()
        for x in X:
            transformed.append(Attribute(
                *tuple(getattr(x, attr) for attr in self._attributes)
            ))

        return transformed

    # noinspection PyPep8Naming
    def transform(self, X: typing.Iterable):  # pylint: disable=invalid-name
        """Apply transformation by applying provided hook to each element."""

        return np.array([
            (t, self._hook(*t)) for t in X
        ])

    # noinspection PyPep8Naming
    def fit_transform(self,
                      X: typing.Iterable,  # pylint: disable=invalid-name
                      y=None,
                      **fit_params):

        # perform fit
        fit = self.fit(X)

        # transform and return
        return self.transform(fit)


class NLTKPreprocessor(TransformerMixin):
    """Preprocessor implementing `fit` method for scikit pipelines.

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
        :param lower: bool, whether to fit tokens to lowercase
        :param strip: bool, whether to strip tokens
    """

    def __init__(self,
                 lemmatizer=None,
                 stemmer=None,
                 tokenizer=None,
                 stopwords=False,
                 tag_dict=None,
                 lower=False,
                 strip=False,
                 lang='english'):
        self._tokenizer = tokenizer or nltk.TreebankWordTokenizer()
        self._lemmatizer = lemmatizer or nltk.WordNetLemmatizer()
        self._stemmer = stemmer or nltk.SnowballStemmer(language=lang)

        self._lower = lower
        self._strip = strip
        self._stopwords = corpus.stopwords.words(lang) if stopwords else set()

        self._lang = lang
        self._tag_dict = tag_dict or dict()

    # noinspection PyPep8Naming
    @staticmethod
    def inverse_transform(X: typing.Iterable) -> np.ndarray:  # pylint: disable=invalid-name
        """Inverse operation to the `fit` method.

        Returns list of shape (len(X),) with the tokens stored in X.

        Note that this does not return the original data provided
        to `fit` method, since lemmatization and stemming
        are not reversible operations and for memory sake, lowercase changes
        are not stored in memory either.
        """
        return np.array([
            list(x[0] for x in X)
        ])

    # noinspection PyPep8Naming
    def transform(self, X: typing.Iterable, y: typing.Iterable = None) -> np.ndarray:  # pylint: disable=invalid-name
        """Apply transformation to each element in X.

        This transformation outputs list of the shape (len(X), 2)
        where each element of the list is a tuple of (token, tag).

        :param X: Iterable, each element should be a string to be tokenized
        :param y: Iterable, labels for each element in X (must be the same
        length as `X`)

        :returns: list of shape (len(x), 2)
        """
        if not y:
            y = [None] * len(list(X))
        assert len(list(X)) == len(list(y)), "len(X) != len(y)"

        return np.array([
            [np.array(list(self.tokenize(sent))), label] for sent, label in zip(X, y)
        ])

    def tokenize(self, sentence: str):
        """Performs tokenization of a given sentence.

        This is key method for transformation process."""
        # Tokenize the sentence with the given tokenizer
        tokenized = self._tokenizer.tokenize(sentence)

        for token, tag in nltk.pos_tag(tokenized, tagset='universal'):
            # Check and correct (if applicable) the tag against given patterns
            for pattern, correction in self._tag_dict.items():
                if re.match(pattern, tag):
                    tag = correction
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

            token = self.stem(token)
            try:
                token = self.lemmatize(token, tag)
            except KeyError:
                # skip if the token can not be lemmatized (eg. tags 'PRT')
                continue

            yield token, tag

    def stem(self, token: str):
        """Stem the word and return the stem."""
        return self._stemmer.stem(token)

    def lemmatize(self, token: str, tag: str):
        """Lemmatize the token based on its tag and return the lemma."""
        # The lemmatizer expects the `pos` argument to be first letter
        # of positional tag of the universal set (which we use by default)
        return self._lemmatizer.lemmatize(token, pos=tag[0].lower())
