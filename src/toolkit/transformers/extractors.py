"""This module contains transformers and feature extractors.

The transformers are used in train and predict pipelines and implement
the `transform` or `fit_transform` methods for this purpose
"""

import typing

import numpy as np
from sklearn.base import TransformerMixin

from toolkit.transformers.hooks import Hook


# noinspection PyTypeChecker
class FeatureExtractor(TransformerMixin):
    """Feature extractor implementing `transform` for scikit pipelines.

    By default, constructs vanilla feature extractor with basic features
    and positional context information.

    :param feature_hooks: dict, {feature_key: Hook}

        Specify features which should be extracted from the given set.
        The hooks are called for each element of the set and return
        corresponding features.
    """

    def __init__(self,
                 feature_hooks=None,
                 share_hooks=False):
        """Initialize FeatureExtractor."""
        if isinstance(feature_hooks, dict):
            # create hooks from the dictionary
            feature_hooks = [Hook(k, v, reuse=share_hooks) for k, v in feature_hooks.items()]

        self._extractor = _FeatureExtractor(share_hooks=share_hooks).update(feature_hooks or [])

        # prototyped
        self._y = None
        self._skip_unfed_hooks = False

    @property
    def feature_keys(self):
        """Return list of hooks of selected feature_keys."""
        return self._extractor.keys

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None, **fit_params):  # pylint: disable=invalid-name,unused-argument
        """Fit the transformer to the given data.

        :param X: Iterable, each element should be a list of tuples (token, tag)

            Ie. an input should be a list of tuples List[(token, tag)],
            which is expected to be the output of NLTKPreprocessor or custom
            tokenization process.

        :param y: Iterable of len(X), target values

        :param fit_params:

            skip_unfed_hooks: bool, whether to skip unfed hooks
        """
        self._skip_unfed_hooks = fit_params.get('skip_unfed_hooks', False)

        if y is None:
            y = [None] * len(X)

        self._y = y

        return self

    # noinspection PyPep8Naming
    def transform(self,
                  X: typing.Iterable  # pylint: disable=invalid-name
                  ) -> list:
        """Apply transformation to each element in X.

        :param X: Iterable, each element should be a list of tuples (token, tag)

            Ie. an input should be a list of tuples List[(token, tag)],
            which is expected to be the output of NLTKPreprocessor or custom
            tokenization process.

        :returns: list

            This transformation outputs list of the shape (len(X),)
            where each element of the list is a nested list of tuples of type
            (tagged_word, dictionary{feature_key: value}, classification label).
            The classification label is a bool indicating whether the label
            provided by `y` to the `fit` matches the token.

            The keys of those features (dictionaries), are names of the feature_keys, ie.
            hooks and the values are the values of those extracted feature_keys.

        """
        transformed = list()

        for tagged_sent, label in zip(X, self._y):
            transformed.append(
                [
                    (
                        tagged_sent[j],
                        self._extract_features(tagged_sent, word_pos=j,
                                               skip_unfed_hooks=self._skip_unfed_hooks),
                        # whether the token matches the label
                        label == tagged_sent[j][0] if label else None
                    ) for j in range(len(tagged_sent))
                ]
            )

        return np.array(transformed)

    def _extract_features(self,
                          tagged_sent: list,
                          word_pos: int,
                          skip_unfed_hooks=False,
                          **kwargs) -> dict:
        """Feed the hooks and extract feature_keys based on those hooks."""
        feed_dict = {
            'tagged': tagged_sent,
            'pos': word_pos,
        }
        feed_dict.update(kwargs)

        return self._extractor.feed(feed_dict=feed_dict, skip_unfed_hooks=skip_unfed_hooks)


class _FeatureExtractor(object):
    """Core of the FeatureExtractor handling hook operations."""

    def __init__(self, share_hooks=False):
        # default hooks to be called by FeatureExtractor
        self._hooks = [
            Hook('prev-word', self._prev_ngram, reuse=share_hooks, n=1),
            Hook('prev-tag', self._prev_ngram, reuse=share_hooks, n=1),
            Hook('prev-bigram', self._prev_ngram, reuse=share_hooks, n=2),
            Hook('next-bigram', self._next_ngram, reuse=share_hooks, n=2),
            Hook('prev-bigram-tags', self._prev_ngram_tags, reuse=share_hooks, n=2),
            Hook('next-bigram-tags', self._next_ngram_tags, reuse=share_hooks, n=2)
        ]

    @property
    def keys(self) -> list:
        """List of hook keys."""
        return [hook.key for hook in self._hooks]

    def update(self, hooks: typing.Union[list, Hook]):
        """Update the hooks used for feature extraction.

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

    def feed(self, feed_dict: dict, skip_unfed_hooks=False) -> dict:
        """Call each hook with the arguments given by values of `feed_dict`.

        :param feed_dict: dict of arguments to be fed into hooks

            `feed_dict` will be passed to the hook as **kwargs, where each
            element of the dict is key, value pair where key is the arguments
            name, value is the arguments value.

        :param skip_unfed_hooks: bool, False by default

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
                if skip_unfed_hooks:
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
        tag = tagged[pos - 1][1]
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
