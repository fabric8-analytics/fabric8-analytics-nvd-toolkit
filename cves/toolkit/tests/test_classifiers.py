"""Tests for classifier module."""

import os
import tempfile
import unittest

import numpy as np

from toolkit.transformers import NBClassifier, cross_validate

# generate random binary data to test classifier
TEST_PROJECT_LABEL = 'candidate'
TEST_FEATURES = np.array(
    [
        (
            (TEST_PROJECT_LABEL, 'tag'),
            {'bin_1': 1, 'bin_0': 0}
        )
    ] * 10
)
TEST_FEATURE_LABELS = np.random.choice([True, False], size=10)
TEST_TRAIN_DATA = np.array([(n, c, l) for (n, c), l in zip(TEST_FEATURES, TEST_FEATURE_LABELS)])


class TestClassifier(unittest.TestCase):
    """Tests for NBClassifier class."""

    def test_fit(self):
        """Test NBClassifier `fit` method."""
        classifier = NBClassifier()

        self.assertIsNotNone(classifier)
        self.assertIsInstance(classifier, NBClassifier)

        classifier = classifier.fit(X=TEST_TRAIN_DATA)

        self.assertIsNotNone(classifier)
        self.assertIsInstance(classifier, NBClassifier)

        # check that classifier was created
        self.assertIsNotNone(classifier._classifier)  # pylint: disable=protected-access
        self.assertIsNotNone(classifier.features)

    def test_predict(self):
        classifier = NBClassifier()

        features = TEST_FEATURES[0, 1]
        # should raise if wasn't fit before
        with self.assertRaises(ValueError):
            classifier.predict(features=features)

        classifier = classifier.fit(TEST_TRAIN_DATA)

        # make prediction
        prediction = classifier.predict(features=features, sample=True)

        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, float)

        # check with no sample specified
        prediction = classifier.predict(features=features, sample=None)
        self.assertIsInstance(prediction, dict)

    def test_fit_predict(self):
        classifier = NBClassifier()
        classifier = classifier.fit(TEST_TRAIN_DATA)

        num_candidates = 3
        candidates = classifier.fit_predict(TEST_FEATURES,
                                            n=num_candidates,
                                            sample=True)

        self.assertIsInstance(candidates, np.ndarray)
        # check correct number of candidates
        self.assertEqual(len(candidates), num_candidates)
        # check that all predictions are the same (no flaw appeared)
        self.assertTrue(
            all([pred == candidates[0] for pred in candidates])
        )

    def test_export(self):
        """Test NBClassifier `export` method."""
        classifier = NBClassifier()
        classifier = classifier.fit(TEST_TRAIN_DATA)

        tmp_dir = tempfile.mkdtemp(prefix='test_export_')
        pickle_path = classifier.export(export_dir=tmp_dir)

        self.assertIsNotNone(pickle_path)

        # check that the tmp_dir contains only the pickled file
        files = None
        for root, _, walkfiles in os.walk(tmp_dir):
            files = [f for f in walkfiles]

        f, = files
        self.assertIsNotNone(files)
        self.assertTrue(f.endswith('.pickle'))

    def test_restore(self):
        """Test NBClassifier `restore` method."""
        classifier = NBClassifier()
        classifier = classifier.fit(TEST_TRAIN_DATA)

        pickle_path = classifier.export()

        restored_classifier = NBClassifier.restore(pickle_path)

        self.assertIsInstance(restored_classifier, NBClassifier)
        self.assertEqual(classifier.features, restored_classifier.features)

    def test_features(self):
        """Test NBClassifier `features` property."""
        classifier = NBClassifier()

        self.assertIsNone(classifier.features)

    def test_evaluate(self):
        """Test NBClassifier `evaluate` method."""
        classifier = NBClassifier()
        classifier.fit(TEST_TRAIN_DATA)

        # NOTE: `astype` is a workaround for the dtype incompatibility,
        # which was caused by prototyping the TEST_FEATURES for the test
        # purposes
        score = classifier.evaluate(
            [TEST_FEATURES],
            [TEST_PROJECT_LABEL],
            sample=True
        )

        self.assertIsNotNone(score)
        self.assertEqual(score, 1.0)

    def test_cross_validate(self):
        """Test NBClassifier `cross_validate` method."""
        classifier = NBClassifier()

        score = cross_validate(
            classifier,
            [TEST_TRAIN_DATA] * 10,  # simulate data
            [TEST_PROJECT_LABEL] * 10,
            folds=5,
            shuffle=True,
            sample=True,
        )

        self.assertIsNotNone(score)
        self.assertEqual(score.mean, 1.0)
