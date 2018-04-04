"""This module contains preprocessors of raw text.

Those preprocessors are transformers and estimators which implement methods
called by sklearn pipeline and can be integrated with it.
"""

import re
import typing
import weakref

import nltk
import nltk.corpus as corpus
import numpy as np

from collections import namedtuple
from sklearn.base import TransformerMixin

from toolkit import utils
from toolkit.preprocessing import GitHubHandler


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


# noinspection PyTypeChecker
class FeatureExtractor(TransformerMixin):
    """Feature extractor implementing `fit` for scikit pipelines.

    By default, constructs vanilla feature extractor with basic features
    and positional context information.

    :param features: dict, {feature_key: Hook}

        Specify features which should be extracted from the given set.
        The hooks are called for each element of the set and return
        corresponding features.
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
                  y: typing.Iterable = None,
                  skip=False) -> typing.List[typing.List[dict]]:
        """Apply transformation to each element in X.

        This transformation outputs list of the shape (len(X),)
        where each element of the list is a tupele of dictionary{feature_key: value},
        classification label (if provided). The classification label is a bool
        indicating whether the label provided by `y` matches the token.

        :param X: Iterable, each element should be tuple of (tagged_sentence, label)

            Ie. an input should be a list of tuples (List[(token, tag)], label),
            which is expected to be the output of NLTKPreprocessor or custom
            tokenization process.

        :param y: Iterable of labels for the given sentences (must be the same
        length as `X`)

        :param skip: bool, whether to skip unfed hooks

        :returns: List[Tuple[dict, label]]

            Each element of the list represents extracted features for the given token,
            which is a list of features per each word in the sentence.

            The keys of those features (dictionaries), are names of the feature_keys, ie.
            hooks and the values are the values of those extracted feature_keys.
        """
        transformed = list()

        if y is None:
            y = [None] * len(list(X))
        assert len(list(X)) == len(list(y)), "len(X) != len(y)"

        for tagged_sent, label in zip(X, y):
            transformed.extend([
                (
                    self._extract_features(tagged_sent, word_pos=j, skip=skip),
                    # whether the token matches the label
                    label == tagged_sent[j][0] if label else None
                ) for j in range(len(tagged_sent))
            ])

        # pycharm is confused about the `None` initialization
        # noinspection PyPep8Naming
        return np.array(transformed)

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
    """Convenient class for handling hooks.

    :param key: str, unique identifier of the hook
    :param func: function to be called by the hook

        The function can not modify any items fed by its arguments.
    :param default_kwargs: default `func` keyword argument values

        Example:
        def foo(x, verbose=False):
            if verbose:
                print('verbosity on')
            return x

        # init with default kwargs
        foo_hook = Hook('foo', foo, verbose=True)
        # and on the call
        foo_hook(x=None)  # prints 'verbosity on'
    """

    __INSTANCES = weakref.WeakSet()

    def __init__(self, key: str, func, **default_kwargs):
        if key in Hook.get_current_keys():
            raise ValueError("Hook with key `%s` already exists" % key)

        # attr initialization
        self._key = str(key)
        self._func = func
        self._default_kwargs = default_kwargs

        # add the key to the class
        Hook.__INSTANCES.add(self)

    @property
    def key(self):
        return self._key

    @property
    def default_kwargs(self):
        return self._default_kwargs

    def __call__(self, *args, **kwargs):

        return self._func(*args, **kwargs)

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
                result[hook.key] = hook(**feed_dict, **hook.default_kwargs)
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
