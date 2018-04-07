"""Naive Bayes classifier."""

import copy
import datetime
import os
import pickle
import re
import typing

import numpy as np

from collections import namedtuple

from nltk import NaiveBayesClassifier
from nltk.probability import ELEProbDist

from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold

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
    def fit(self, X: typing.Union[list, np.ndarray], y=None, **fit_params):  # pylint: disable=invalid-name,unused-argument
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
    def evaluate(self,
                 X, y,  # pylint: disable=invalid-name,unused-argument
                 sample,
                 n=3):
        """Perform evaluation of the classifier instance.

        :param X: list or ndarray, prediction tuples of type (name_tuple, feature_set [,feature_label)

            Same as for `fit_predict` method

        :param y: list or ndarray of labels
        :param sample:

        one of labels to get the prediction for (for example,
                                                 if labels are ['class_A', 'class_B', 'class_C'], the sample
        could be 'class_A'.

        :param n: int, number of candidates to output
        """
        if len(X) != len(y):
            raise ValueError("`X` and `y` must be of the same length.")

        if isinstance(X, list):
            X = np.array(X)

        if isinstance(y, list):
            y = np.array(y)

        candidate_arr = [
            self.fit_predict(x, n=n, sample=sample) for x in X
        ]

        # pre-initialize prediction array which will hold correct predictions
        y_pred = np.empty(shape=y.shape, dtype=np.bool)

        correctly_predicted = 0
        for i, candidates in enumerate(candidate_arr):
            pred = self._valid_candidates(candidates, y[i])
            correctly_predicted += int(pred)

        # return the accuracy score
        return precision(total=len(y), correct=correctly_predicted)
    # noinspection PyPep8Naming, PyUnusedLocal

    def fit_predict(self, X, y=None, **fit_params):  # pylint: disable=invalid-name,unused-argument
        """Makes prediction about the given data.

        :param X: list or ndarray of prediction data

            The prediction data is expected to be of type List[(name_tuple, feature_set [,feature,label)]
            where feature_set corresponds to the output of FeatureExtractor and feature labels (if provided)
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
            if len(x) == 3:
                # feature label was provided as part of X set (usual case), ignore it
                name_tuple, features, _ = x
            else:
                name_tuple, features = x
            candidates[i] = (name_tuple, self.predict(features, sample=sample))

        candidates = sorted(candidates, key=lambda t: t[1], reverse=True)

        return np.array(candidates)[:n, 0]

    def predict(self, features, sample=None) -> typing.Any:
        """Make predictions based on given features.

        :params features:

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

    @staticmethod
    def _valid_candidates(candidates: list, label):
        """Check whether the correct label is among candidates."""
        for candidate, tag in candidates:
            # FIXME: a bug here, nltk lets weird things like '**' go through -> causes crash
            try:
                if re.search(candidate, label, flags=re.IGNORECASE):
                    return True
            except:
                return False

        return False


# noinspection PyPep8Naming, PyUnusedLocal
def cross_validate(classifier,
                   X, y,  # pylint: disable=invalid-name,unused-argument
                   sample,
                   n=3,
                   folds=10,
                   shuffle=True):
    """Evaluate cross-validation accuracy of the classifier.

    **Note:** this method DOES NOT evaluate the INSTANCE of the classifier.
    Instead, it trains a shadow classifier of the same parameters as the
    given `classifier` and evaluates its accuracy.

    :param classifier: NBClassifier instance to be evaluated
    :param X: list or ndarray of train data

        The same as is provided to `fit` method.

    :param y: list or ndarray of labels
    :param sample:

        one of labels to get the prediction for (for example,
        if labels are ['class_A', 'class_B', 'class_C'], the sample
        could be 'class_A'.

    :param n: int, number of candidates to output
    :param folds: int, number of folds to be used for cross-validation
    :param shuffle: whether to shuffle the data

        If None, no cross-validaiton is performed
    """
    if not isinstance(classifier, NBClassifier):
        raise TypeError("`classifier` expected to be of type `{}`, got `{}`"
                        .format(NBClassifier, type(classifier)))

    if len(X) != len(y):
        raise ValueError("`X` and `y` must be of the same length.")

    if isinstance(X, list):
        X = np.array(X)

    if isinstance(y, list):
        y = np.array(y)

    # copy the classifier to avoid corrupting the original one
    clf = copy.copy(classifier)

    accuracy = list()
    # perform KFold
    kf = KFold(n_splits=folds, shuffle=shuffle)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # collapse along the first axis as `fit` expects continuous stream
        X_fit = np.vstack(X_train)

        # fit with the collapsed data
        clf.fit(X_fit)

        # make predictions
        score = clf.evaluate(X_test, y_test, n=n, sample=sample)

        # compute the accuracy
        accuracy.append(score)

    # return the accuracy score
    accuracy = np.array(accuracy)
    Score = namedtuple('Score', 'values mean std')

    return Score(accuracy, accuracy.mean(), accuracy.std())


def precision(total: int, correct: int) -> float:
    """Calculate precision."""

    return float(correct / total)


def weighted_precision(y_true, y_pred, weights):
    """Calculate weighted precision."""
    raise NotImplementedError
