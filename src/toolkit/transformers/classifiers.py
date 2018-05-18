"""Naive Bayes classifier."""

import copy
import datetime
import os
import pickle
import re
import typing

from collections import namedtuple

import numpy as np

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

    :param label_probdist: P(label)

        The probability distribution over labels.

        It is expressed as a ``ProbDistI`` whose samples are labels.
        I.e., P(label) = ``label_probdist.prob(label)``.

    :param feature_probdist: P(fname=fval|label)

        The probability distribution for feature values, given labels.

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
        """Initialize NBClassifier."""
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
        """Return features most informative for classification."""
        if self._classifier is None:
            return None

        return self._classifier.most_informative_features()

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self,
            X: typing.Iterable,  # pylint: disable=invalid-name
            y=None,  # pylint: disable=unused-argument
            **fit_params):
        """Fits the classifier to the given data set.

        :param X: Iterable, output of FeatureExtractor

            The X is expected to be an iterable of tuples (tagged_word, feature_set, label),
            where feature set is a dictionary of evaluated features.
            The format of X matches the output of `FeatureExtractor`.

        :param y: redundant (included to preserve base class method definition)
        """
        # NLTK classifier expects stacked featuresets for the training,
        # so we need to reduce the dimenstionality
        labeled_featuresets = list()
        for entry in X:
            labeled_featuresets.extend([
                (featureset, feature_label)
                for _, featureset, feature_label in entry
            ])

        # initialize the NLTK classifier
        self._classifier = NaiveBayesClassifier.train(
            labeled_featuresets,
            estimator=self._estimator
        )

        return self

    # noinspection PyPep8Naming, PyUnusedLocal
    def transform(self, X):  # pylint: disable=invalid-name,unused-argument
        """Auxiliary function to be used in pipeline."""
        return self

    # noinspection PyPep8Naming
    def evaluate(self,
                 X: typing.Iterable,  # pylint: disable=invalid-name
                 y: typing.Iterable,
                 sample,
                 n=3,
                 filter_hooks=None):
        """Perform evaluation of the classifier instance.

        :param X: Iterable, test data

            Same shape as for `fit` and `fit_predict` methods

        :param y: Iterable, of labels
        :param sample: one of labels to get the prediction for

            For example, if labels are ['class_A', 'class_B', 'class_C'], the sample
            could be 'class_A'.

        :param n: int, number of candidates to output

        :param filter_hooks: list of hooks, will be used to filter predictions

            The hook should take a tuple of ((word, tag), score) as its parameter
            and output boolean whether or not it passes the filter.
        """
        # noinspection PyTypeChecker,PyTypeChecker
        if len(X) != len(y):
            raise ValueError("`X` and `y` must be of the same length.")

        candidate_arr = self.fit_predict(X, n=n, sample=sample, filter_hooks=filter_hooks or [])

        correctly_predicted = 0
        for candidates, label in zip(candidate_arr, y):
            pred = self._valid_candidates(candidates, label)
            correctly_predicted += int(pred)

        # return the accuracy score
        # noinspection PyTypeChecker
        return precision(total=len(y), correct=correctly_predicted)

    # noinspection PyPep8Naming
    def fit_predict(self,
                    X: typing.Iterable,  # pylint: disable=invalid-name
                    y=None,  # pylint: disable=unused-argument
                    **fit_params):
        """Make prediction about the given data.

        :param X: Iterable, prediction data

            The prediction data is expected to be of type
            List[(name_tuple, feature_set [,feature,label)] where feature_set
            corresponds to the output of FeatureExtractor and feature
            labels (if provided) should be None (will be ignored anyway).

        :param y: redundant (included to preserve bace class method definition)
        :param fit_params: kwargs, fit parameters

            n: number of candidates to output
            sample: one of labels to get the prediction for (for example,
            if labels are ['class_A', 'class_B', 'class_C'], the sample
            could be 'class_A'.
            filter_hooks: list of hooks, will be used to filter predictions

                The hook should take a tuple of ((word, tag), score) as its parameter
                and output boolean whether or not it passes the filter.
        """
        # get fit parameters
        n = fit_params.get('n', 3)
        sample = fit_params.get('sample', None)

        # do not allow sample to be `None` (wouldn't be possible to sort
        # the candidates in a logical way)
        if sample is None:
            raise ValueError("`fit_parameter` `sample` was not specified."
                             " This is not allowed in `fit_predict` method")

        if not all([hasattr(var, '__len__') for var in [X, y or []]]):
            raise TypeError("`X` and `y` must implement `__len__` method")

        # noinspection PyTypeChecker
        predictions = [None] * len(X)
        for i, x in enumerate(X):
            candidate_pred = [None] * len(x)
            for j, candidate in enumerate(x):
                if len(candidate) == 3:
                    # feature label was provided as part of X set (usual case), ignore it
                    name_tuple, features, _ = candidate
                else:
                    name_tuple, features = candidate
                candidate_pred[j] = (name_tuple, self.predict(features, sample=sample))

            sorted_pred = sorted(candidate_pred, key=lambda t: t[1], reverse=True)

            for hook in fit_params.get('filter_hooks', []):
                sorted_pred = list(filter(hook, sorted_pred))

            predictions[i] = sorted_pred[:n]

        return np.array(predictions)

    def predict(self, features: dict, sample=None) -> typing.Any:
        """Make predictions based on given features.

        :param features: dict, features to be used for prediction

            Dictionary of (feature_key, feature_value)

        :param sample:

            one of labels to get the prediction for (for example,
            if labels are ['class_A', 'class_B', 'class_C'], the sample
            could be 'class_A'.

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
        """Print features most informative for classification."""
        if self._classifier is None:
            return

        self._classifier.show_most_informative_features()

    def export(self, export_dir=None, export_name=None) -> str:
        """Export timestamped pickled classifier to the given directory.

        :returns: path to the timestamped .checkpoint file
        """
        export_dir = export_dir or 'export/'
        export_name = export_name or 'classifier'

        if export_name.endswith('.checkpoint'):
            export_name = ".".join(export_name.split('.')[:-1])

        time_stamp = str(datetime.datetime.now().timestamp())

        # create export directory
        os.makedirs(export_dir, exist_ok=True)

        time_stamped_fname = ".".join([export_name, time_stamp, 'checkpoint'])
        time_stamped_fpath = os.path.join(export_dir, time_stamped_fname)

        # pickle and export the classifier
        with open(time_stamped_fpath, 'wb') as exp_file:
            pickle.dump(self, exp_file)

        return time_stamped_fpath

    @staticmethod
    def restore(checkpoint) -> "NBClassifier":
        """Restores the classifier from a checkpoint file.

        :param checkpoint: path to directory or specific checkpoint

            If path to directory provided, the newest checkpoint
            is restored.
        """
        def _restore_checkpoint(fp):
            with open(fp, 'rb') as checkpoint_file:
                # load the exported classifier
                return pickle.load(checkpoint_file)

        if os.path.isdir(checkpoint):
            checkpoint_dir = checkpoint
            checkpoints = [
                os.path.join(checkpoint_dir, f)
                for f in os.listdir(checkpoint) if f.endswith('.checkpoint')
            ]
            # find the latest
            if not checkpoints:
                raise ValueError("No checkpoints were found in `{}`."
                                 .format(checkpoint))
            latest_checkpoint = sorted(checkpoints)[-1]
            clf = _restore_checkpoint(latest_checkpoint)

        else:
            clf = _restore_checkpoint(checkpoint)

        return clf

    @staticmethod
    def _valid_candidates(candidates: typing.Iterable, label):
        """Check whether the correct label is among candidates."""
        for candidate, _ in candidates:
            # FIXME: a bug here, NLTK lets weird things like '**' go through -> causes crash
            candidate_name, _ = candidate
            try:
                if re.search(candidate_name, label, flags=re.IGNORECASE):
                    return True
            except Exception:
                return False

        return False


# noinspection PyPep8Naming, PyUnusedLocal
def cross_validate(classifier,
                   X: typing.Iterable,  # pylint: disable=invalid-name
                   y: typing.Iterable,
                   sample,
                   n=3,
                   folds=10,
                   filter_hooks=None,
                   shuffle=True):
    """Evaluate cross-validation accuracy of the classifier.

    **Note:** this method DOES NOT evaluate the INSTANCE of the classifier.
    Instead, it trains a shadow classifier of the same parameters as the
    given `classifier` and evaluates its accuracy.

    :param classifier: NBClassifier instance to be evaluated
    :param X: Iterable of train data

        The same as is provided to `fit` method.

    :param y: Iterable of labels
    :param sample: one of labels

        one of labels to get the prediction for (for example,
        if labels are ['class_A', 'class_B', 'class_C'], the sample
        could be 'class_A'.

    :param n: int, number of candidates to output
    :param folds: int, number of folds to be used for cross-validation
    :param filter_hooks: list of hooks, will be used to filter predictions

        The hook should take a tuple of ((word, tag), score) as its parameter
        and output boolean whether or not it passes the filter.

    :param shuffle: whether to shuffle the data

        If None, no cross-validaiton is performed
    """
    if not isinstance(classifier, NBClassifier):
        raise TypeError("`classifier` expected to be of type `{}`, got `{}`"
                        .format(NBClassifier, type(classifier)))

    # disable inspection and let it fail if x and y are not seized
    # noinspection PyTypeChecker
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
        _, y_test = y[train_index], y[test_index]

        # fit with the collapsed data
        clf.fit(X_train)

        # make predictions
        score = clf.evaluate(X_test, y_test, n=n, sample=sample,
                             filter_hooks=filter_hooks or [])

        # compute the accuracy
        accuracy.append(score)

    # return the accuracy score
    accuracy = np.array(accuracy)
    Score = namedtuple('Score', 'values mean std')

    return Score(accuracy, accuracy.mean(), accuracy.std())


def precision(total: int, correct: int) -> float:
    """Calculate precision."""
    return float(correct / total)


def weighted_precision(y_true, y_pred, weights):  # pylint: disable=unused-argument
    """Calculate weighted precision."""
    raise NotImplementedError("The feature has not been implemented yet."
                              " Sorry for the inconvenience.")
