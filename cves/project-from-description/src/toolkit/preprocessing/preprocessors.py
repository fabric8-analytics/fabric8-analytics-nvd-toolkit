"""This module contains preprocessors of raw text.

Those preprocessors are transformers and estimators which implement methods
called by sklearn pipeline and can be integrated with it.
"""

import re
import typing

import nltk
import nltk.corpus as corpus

from sklearn.base import TransformerMixin


class NLTKPreprocessor(TransformerMixin):
    """Base class."""

    def __init__(self,
                 lemmatizer=None,
                 stemmer=None,
                 tokenizer=None,
                 stopwords=False,
                 tag_dict=None,
                 lower=False,
                 strip=False):
        """Initialize NLTK Preprocessor.

        This preprocessor performs tokenization, stemming and lemmatization
        by default. Processors used for these operations are customizable.

        Other text processing operations are not mandatory and can be optimized
        by user.
        """
        self._tokenizer = tokenizer or nltk.TreebankWordTokenizer()
        self._lemmatizer = lemmatizer or nltk.WordNetLemmatizer()
        self._stemmer = stemmer or nltk.SnowballStemmer(language='english')

        self._lower = lower
        self._strip = strip
        self._stopwords = corpus.stopwords.words('english') if stopwords else set()

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
        """Apply transformation to each sentence in X.

        This transformation outputs list of the shape (len(X), 2)
        where each element of the list is a tuple of (token, tag).
        """
        return [
            list(self.tokenize(sent)) for sent in X
        ]

    def tokenize(self, sentence: str):
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
            token = self.lemmatize(token, tag)

            yield token, tag

    def stem(self, token: str):
        """Stem the word and return the stem."""

        return self._stemmer.stem(token)

    def lemmatize(self, token: str, tag: str):
        """Lemmatize the token based on its tag and return the lemma."""
        # The lemmatizer expects the `pos` argument to be first letter
        # of positional tag of the universal set (which we use by default)
        return self._lemmatizer.lemmatize(token, pos=tag[0].lower())
