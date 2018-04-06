"""Naive Bayes classifier."""

import datetime
import os
import pickle
import typing

import numpy as np

from nltk import NaiveBayesClassifier
from nltk.probability import ELEProbDist
from sklearn.base import TransformerMixin

# Note: ATM try the approach with NLTK NaiveBayesClassifier, alternative
# to this approach would be MultinomialNB classifier from scikit library,
# which # would probably integrate in a better way with the classification
# pipeline.


class NBClassifier(TransformerMixin):
    """Naive Bayes classifier for part-of-text classification.

    The classifier creates a wrapper around NLTK NaiveBayesClassifier
    and implements `transform` and `fit_transform` methods suitable for
    pipeline integration.

        :param label_probdist:
            P(label), the probability distribution over labels.

            It is expressed as a ``ProbDistI`` whose samples are labels.
            I.e., P(label) = ``label_probdist.prob(label)``.

        :param feature_probdist:
            P(fname=fval|label), the probability distribution for feature values, given labels.

            It is expressed as a dictionary whose keys are ``(label, fname)``
            pairs and whose values are ``ProbDistI`` objects over feature values.
            I.e., P(fname=fval|label) = ``feature_probdist[label,fname].prob(fval)``.
            If a given ``(label,fname)`` is not a key in ``feature_probdist``,
            then it is assumed that the corresponding P(fname=fval|label)
            is 0 for all values of ``fval``.
    """

    def __init__(self,
                 label_probdist=None,
                 feature_probdist=None,
                 estimator=ELEProbDist):

        self._estimator = estimator

        # in case arguments are specified (ie. when restoring the classifier)
        if all([label_probdist, feature_probdist]):
            self._classifier = NaiveBayesClassifier(
                label_probdist=label_probdist,
                feature_probdist=feature_probdist,
            )
        else:
            self._classifier = None

    @property
    def features(self):
        if self._classifier is None:
            return None

        return self._classifier.most_informative_features()

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X: typing.Union[list, np.ndarray], *args):  # pylint: disable=invalid-name,unused-argument
        """Fits the classifier to the given data set.

        :param X: list or ndarray of train data

            The X is expected to be an iterable of tuples (tagged_word, feature_set, label),
            where feature set is a dictionary of evaluated features.
            The format of X matches the output of `FeatureExtractor`.
        """

        if isinstance(X, list):
            X = np.array(X)

        # initialize the NLTK classifier
        self._classifier = NaiveBayesClassifier.train(
            labeled_featuresets=X[:, 1:],
            estimator=self._estimator
        )

        return self

    # noinspection PyPep8Naming, PyUnusedLocal
    def transform(self, X):  # pylint: disable=invalid-name,unused-argument
        """Auxiliary function to be used in pipeline."""

        return self

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit_predict(self, X, y=None, **fit_params):  # pylint: disable=invalid-name,unused-argument
        """Makes prediction about the given data.

        :param X: list or ndarray of prediction data

            The prediction data is expected to be of type (feature_set, None),
            where feature_set corresponds to the output of FeatureExtractor and labels
            should be None (will be ignored anyway).

        :param y: auxiliary
        :param fit_params: kwargs, fit parameters

            n: number of candidates to output
            sample: one of labels to get the prediction for (for example,
            if labels are ['class_A', 'class_B', 'class_C'], the sample
            could be 'class_A'.
        """

        # get fit parameters
        n = fit_params.get('n', 3)
        sample = fit_params.get('sample', None)

        # do not allow sample to be `None` (wouldn't be possible to sort
        # the candidates in a logical way)
        if sample is None:
            raise ValueError("`fit_parameter` `sample` was not specified."
                             " This is not allowed in `fit_predict` method")

        candidates = [None] * len(X)
        for i, x in enumerate(X):
            name_tuple, features, label = x
            candidates[i] = (name_tuple, self.predict(features, sample=sample))

        candidates = sorted(candidates, key=lambda t: t[1], reverse=True)

        return np.array(candidates[:n])

    def predict(self, features, sample=None) -> typing.Any:
        """Make predictions based on given features.

        :returns: Union[float, dict]

            If `sample` is specified, returns P(sample|features),
            ie the probability of `sample` given features,
            where `sample` is one of labels.
            Otherwise returns dict of (label: max_prob) for all
            known labels.
        """
        if self._classifier is None:
            raise ValueError("Unable to make predictions. "
                             "Classifier has not been trained yet!")

        prob_dist = self._classifier.prob_classify(features)
        # sort by the probability

        if sample is not None:
            probs = prob_dist.prob(sample)
        else:
            probs = {s: prob_dist.prob(s) for s in self._classifier.labels()}

        return probs

    def show_most_informative_features(self):
        if self._classifier is None:
            return

        self._classifier.show_most_informative_features()

    def export(self, export_dir=None, export_name=None) -> str:
        """Exports timestamped pickled classifier to the given directory.

        :returns: path to the timestamped .pickle file
        """
        export_dir = export_dir or 'export/'
        export_name = export_name or 'classifier'

        if not export_name.endswith('.pickle'):
            export_name += '.pickle'

        time_stamped_dir = os.path.join(export_dir, str(datetime.datetime.now().timestamp()))
        # create timestamped export directory
        os.makedirs(time_stamped_dir)

        time_stamped_fname = os.path.join(time_stamped_dir, export_name)
        # pickle and export the classifier
        with open(time_stamped_fname, 'wb') as exp_file:
            pickle.dump(self, exp_file)

        return time_stamped_fname

    @staticmethod
    def restore(export_file) -> "NBClassifier":
        """Restores the classifier from a pickled file."""
        with open(export_file, 'rb') as exp_file:
            # load the exported classifier
            classifier = pickle.load(exp_file)

        return classifier
