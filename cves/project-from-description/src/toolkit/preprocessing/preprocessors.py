"""This module contains preprocessors of raw text.

Those preprocessors are transformers and estimators which implement methods
called by sklearn pipeline and can be integrated with it.
"""

import re
import typing
import weakref

import nltk
import nltk.corpus as corpus

from collections import namedtuple
from sklearn.base import TransformerMixin

from toolkit import utils
from toolkit.preprocessing import GitHubHandler


class NVDFeedPreprocessor(TransformerMixin):
    """Preprocessor collecting relevant data from NVD feed.

    :param attributes: Iterable of attributes to extract,

        While `transform`, each attribute will be gathered from each element
        of the list fed to the method and return in the resulting tuple.

        If None, ('cve_id', 'references', 'description') are selected by default
    """

    def __init__(self,
                 attributes: typing.Iterable = None,
                 handler=GitHubHandler):
        if attributes and not isinstance(attributes, typing.Iterable):
            raise TypeError(
                "Argument `attributes` expected to be of type `{}`, got `{}`"
                .format(typing.Iterable, type(attributes))
            )

        self._cve_attributes = attributes or ('cve_id', 'references', 'description')
        self._handler = handler

    # noinspection PyPep8Naming
    def transform(self, cves: typing.Iterable):  # pylint: disable=invalid-name
        """Apply transformation to each element in `cves`.

        This transformation outputs list of tuples

        :param cves:
            Iterable, each element is assumed to be of type "nvdlib.model.CVE"
            or an object implementing attributes given in `attr_list`

        :returns: list of shape (len(x), len(attr_list))
            Each element of the resulting list is a namedtuple of cve attributes
        """

        # filter the cves by the given handler
        cve_tuples = self._filter_by_handler(cves)

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
    def inverse_transform(X: typing.Iterable) -> list:  # pylint: disable=invalid-name
        """Inverse operation to the `transform` method.

        Returns list of shape (len(X),) with the tokens stored in X.

        Note that this does not return the original data provided
        to `transform` method, since lemmatization and stemming
        are not reversible operations and for memory sake, lowercase changes
        are not stored in memory either.
        """
        return [
            list(x[0] for x in X)
        ]

    # noinspection PyPep8Naming
    def transform(self, X: typing.Iterable) -> list:  # pylint: disable=invalid-name
        """Apply transformation to each element in X.

        This transformation outputs list of the shape (len(X), 2)
        where each element of the list is a tuple of (token, tag).

        :param X: Iterable, each element should be a string to be tokenized

        :returns: list of shape (len(x), 2)
        """
        return [
            list(self.tokenize(sent)) for sent in X
        ]

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


# noinspection PyTypeChecker
class FeatureExtractor(TransformerMixin):
    """Feature extractor implementing `transform` for scikit pipelines.

    By default, constructs vanilla feature extractor with basic feature_keys
    and positional context information. Use `feature_keys` argument to specify
    custom feature_keys.
    """

    def __init__(self,
                 features=None):
        if isinstance(features, dict):
            # create hooks from the dictionary
            features = [Hook(k, v) for k, v in features.items()]

        self._extractor = _FeatureExtractor().update(features or [])

    @property
    def feature_keys(self):
        """Return list of hooks of selected feature_keys."""
        return self._extractor.keys

    # noinspection PyPep8Naming
    def transform(self,
                  X: typing.Iterable,  # pylint: disable=invalid-name
                  skip=False) -> typing.List[typing.List[dict]]:
        """Apply transformation to each element in X.

        This transformation outputs list of the shape (len(X),)
        where each element of the list is a dictionary of (feature_key, value).

        :param X: Iterable, each element should be tokenized sentence

            Ie. an input could be a list of tuples (token, tag), which
            is expected to be the output of NLTKPreprocessor or custom
            tokenization process.

        :param skip: bool, whether to skip unfed hooks

        :returns: list of lists of dictionaries

            Each element of the list represents a sentence representing by another list,
            which is a list of features per each word in the sentence.

            The keys of those features (dictionaries), are names of the feature_keys, ie.
            hooks and the values are the values of those extracted feature_keys.
        """
        transformed = [None] * len(list(X))

        for i, tagged_sent in enumerate(X):
            transformed[i] = [
                self._extract_features(tagged_sent, word_pos=j, skip=skip)
                for j in range(len(tagged_sent))
            ]

        # pycharm is confused about the `None` initialization
        # noinspection PyPep8Naming
        return transformed

    def _extract_features(self,
                          tagged_sent: list,
                          word_pos: int,
                          skip=False,
                          **kwargs) -> dict:
        """Feeds the hooks and extract feature_keys based on those hooks."""
        feed_dict = {
            'tagged': tagged_sent,
            'pos': word_pos,
        }
        feed_dict.update(kwargs)

        return self._extractor.feed(feed_dict=feed_dict, skip=skip)


class Hook(object):
    """Convenient class for handling hooks."""

    __INSTANCES = weakref.WeakSet()

    def __init__(self, key: str, func, **kwargs):
        if key in Hook.get_current_keys():
            raise ValueError("Hook with key `%s` already exists" % key)

        # attr initialization
        self._key = key
        self._func = func
        self._default_args = kwargs

        # add the key to the class
        Hook.__INSTANCES.add(self)

    @property
    def key(self):
        return self._key

    @property
    def default_args(self):
        return self._default_args

    def __call__(self, **kwargs):

        return self._func(**kwargs)

    @classmethod
    def get_current_hooks(cls) -> list:
        """Returns instances of this class."""
        return list(cls.__INSTANCES)

    @classmethod
    def get_current_keys(cls) -> set:
        """Returns keys to the instances of this class."""
        return set([hook.key for hook in cls.__INSTANCES])

    @classmethod
    def clear_current_instances(cls):
        """Clean up the references held by the class.

        This function is not usually called by user, mainly used for tests
        where cleanup is needed.
        """
        cls.__INSTANCES.clear()


class _FeatureExtractor(object):
    """Core of the FeatureExtractor handling hook operations."""

    def __init__(self):
        # default hooks to be called by FeatureExtractor
        self._hooks = [
            Hook('prev-word', self._prev_ngram, n=1),
            Hook('prev-tag', self._prev_ngram, n=1),
            Hook('prev-bigram', self._prev_ngram, n=2),
            Hook('next-bigram', self._next_ngram, n=2),
            Hook('prev-bigram-tags', self._prev_ngram_tags, n=2),
            Hook('next-bigram-tags', self._next_ngram_tags, n=2)
        ]

    @property
    def keys(self) -> list:
        """List of hook keys."""
        return [hook.key for hook in self._hooks]

    def update(self, hooks: typing.Union[list, Hook]):
        """Updates the hooks used for feature extraction.

        :param hooks: list[Hook], custom hooks used for feature extraction

        :returns: self
        """
        if isinstance(hooks, Hook):
            # make it a list if user provided single hook
            hooks = [hooks]

        if not all([isinstance(hook, Hook) for hook in hooks]):
            print(hooks)
            raise ValueError("`hooks` elements expected to be of type `%r`"
                             % Hook)

        # extend current hooks
        self._hooks.extend(hooks)

        return self

    def feed(self, feed_dict: dict, skip=False) -> dict:
        """Calls each hook with the arguments given by values of `feed_dict`.

        :param feed_dict: dict of arguments to be fed into hooks

            `feed_dict` will be passed to the hook as **kwargs, where each
            element of the dict is key, value pair where key is the arguments
            name, value is the arguments value.

        :param skip: bool, False by default

            If True, allows skipping unfed hooks, otherwise raises AttributeError

        :returns: dict
            Where the key is the hook key and value is the value returned
            by the hook.
        """
        result = dict()
        for hook in self._hooks:
            # check that all arguments are provided
            try:
                result[hook.key] = hook(**feed_dict, **hook.default_args)
            except TypeError as e:
                if skip:
                    continue
                else:
                    raise e

        return result

    @staticmethod
    def _prev_ngram(tagged: list, pos: int, n: int):
        """Extract contextual information about previous n-gram words."""
        if n == 0:
            return ''
        if pos == 0:
            return '<start>'
        word = tagged[pos - 1][0]
        return " ".join([
            _FeatureExtractor._prev_ngram(tagged, pos - 1, n - 1), word
        ]).strip()

    @staticmethod
    def _prev_ngram_tags(tagged, pos, n):
        """Extract contextual information about previous n-gram tags."""
        if n == 0:
            return ''
        if pos == 0:
            return '<start>'
        tag = tagged[pos - 1][0]
        return " ".join([
            _FeatureExtractor._prev_ngram_tags(tagged, pos - 1, n - 1), tag
        ]).strip()

    @staticmethod
    def _next_ngram(tagged, pos, n):
        """Extract contextual information about following n-gram words."""
        if n == 0:
            return ''
        if pos == len(tagged) - 1:
            return '<end>'
        word = tagged[pos + 1][0]
        return " ".join([
            word, _FeatureExtractor._next_ngram(tagged, pos + 1, n - 1)
        ]).strip()

    @staticmethod
    def _next_ngram_tags(tagged, pos, n):
        """Extract contextual information about following n-gram tags."""
        if n == 0:
            return ''
        if pos == len(tagged) - 1:
            return '<end>'
        tag = tagged[pos + 1][1]
        return " ".join([
            tag, _FeatureExtractor._next_ngram_tags(tagged, pos + 1, n - 1)
        ]).strip()
