"""Tests for classifier module."""

import os
import tempfile
import unittest

import numpy as np

from toolkit.transformers import NBClassifier, cross_validate
from toolkit.pipelines import extract_labeled_features


class TestClassifier(unittest.TestCase):
    """Tests for NBClassifier class."""

    @classmethod
    def setUpClass(cls):
        """Return preprocessed extracted labeled features."""
        from nvdlib.nvd import NVD

        feed = NVD.from_feeds(feed_names=['recent'])
        self.assertIsNotNone(feed)

        # download and update
        feed.update()

        # get the sample cves
        __cve_iter = feed.cves()

        data = list(__cve_iter)

        data, labels = extract_labeled_features(
            data=data,
            nvd_attributes=['project', 'description'],
            nltk_feed_attributes=['description']
        )

        cls.data, cls.labels = data, labels

    def test_fit(self):
        """Test NBClassifier `fit` method."""
        classifier = NBClassifier()

        self.assertIsNotNone(classifier)
        self.assertIsInstance(classifier, NBClassifier)

        classifier = classifier.fit(X=self.data)

        self.assertIsNotNone(classifier)
        self.assertIsInstance(classifier, NBClassifier)

        # check that classifier was created
        self.assertIsNotNone(classifier._classifier)  # pylint: disable=protected-access
        self.assertIsNotNone(classifier.features)

    def test_features(self):
        """Test NBClassifier `features` property."""
        classifier = NBClassifier()

        self.assertIsNone(classifier.features)

        # after fit
        classifier = classifier.fit(X=self.data)

        self.assertIsNotNone(classifier.features)

    def test_predict(self):
        """Test NBClassifier `predict` method."""
        classifier = NBClassifier()

        features = self.data[0][1][1]  # sample features
        # should raise if wasn't fit before
        with self.assertRaises(ValueError):
            classifier.predict(features=features)

        classifier = classifier.fit(self.data)

        # make prediction
        prediction = classifier.predict(features=features, sample=True)

        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, float)

        # check with no sample specified
        prediction = classifier.predict(features=features, sample=None)
        self.assertIsInstance(prediction, dict)

    def test_fit_predict(self):
        """Test NBClassifier `fit_predict` method."""
        classifier = NBClassifier()

        classifier = classifier.fit(self.data)
        self.assertIsNotNone(classifier)

        pred_data = self.data[:10]

        num_candidates = 3
        candidates = classifier.fit_predict(pred_data,
                                            n=num_candidates,
                                            sample=True)

        self.assertIsInstance(candidates, np.ndarray)
        # check correct number of candidates
        self.assertEqual(candidates.shape[1], num_candidates)
        self.assertEqual(candidates.shape[2], 2)  # (candidate, proba)

    def test_export(self):
        """Test NBClassifier `export` method."""
        classifier = NBClassifier()
        classifier = classifier.fit(self.data)
        self.assertIsNotNone(classifier)

        tmp_dir = tempfile.mkdtemp(prefix='test_export_')
        pickle_path = classifier.export(export_dir=tmp_dir)

        self.assertIsNotNone(pickle_path)

        # check that the tmp_dir contains only the pickled file
        files = None
        for root, _, walkfiles in os.walk(tmp_dir):
            files = [f for f in walkfiles]

        f, = files
        self.assertIsNotNone(files)
        self.assertTrue(f.endswith('.checkpoint'))

    def test_restore(self):
        """Test NBClassifier `restore` method."""
        classifier = NBClassifier()
        classifier = classifier.fit(self.data)
        self.assertIsNotNone(classifier)

        tmp_dir = tempfile.mkdtemp(prefix='test_export_')
        pickle_path = classifier.export(export_dir=tmp_dir)

        restored_classifier = NBClassifier.restore(pickle_path)

        self.assertIsInstance(restored_classifier, NBClassifier)
        self.assertEqual(classifier.features, restored_classifier.features)

    def test_evaluate(self):
        """Test NBClassifier `evaluate` method."""
        classifier = NBClassifier()
        classifier.fit(self.data)
        self.assertIsNotNone(classifier)

        # NOTE: `astype` is a workaround for the dtype incompatibility,
        # which was caused by prototyping the TEST_FEATURES for the test
        # purposes
        zero_score_labels = [None] * len(self.data)
        score = classifier.evaluate(
            X=self.data,
            y=zero_score_labels,
            sample=True
        )

        self.assertIsNotNone(score)
        self.assertEqual(score, 0.0)

        score = classifier.evaluate(
            X=self.data,
            y=self.labels,
            sample=True
        )

        self.assertIsNotNone(score)
        self.assertGreater(score, 0.0)

    def test_cross_validate(self):
        """Test NBClassifier `cross_validate` method."""
        classifier = NBClassifier()
        self.assertIsNotNone(classifier)

        zero_score_labels = [None] * len(self.data)
        score = cross_validate(
            classifier,
            self.data,
            zero_score_labels,
            folds=5,
            shuffle=True,
            sample=True,
        )

        self.assertIsNotNone(score)
        self.assertEqual(score.mean, 0.0)

        score = cross_validate(
            classifier,
            self.data,
            self.labels,
            folds=5,
            shuffle=True,
            sample=True,
        )

        self.assertIsNotNone(score)
        self.assertGreater(score.mean, 0.0)


def _get_extracted_test_data():
    """Return preprocessed data.

    Note: used for tests only.
    """
    from nvdlib.nvd import NVD

    feed = NVD.from_feeds(feed_names=['recent'])
    assert feed is not None

    # download and update
    feed.update()

    # get the sample cves
    __cve_iter = feed.cves()

    data = list(__cve_iter)

    data, labels = extract_labeled_features(
        data=data,
        nvd_attributes=['project', 'description']
    )

    return data, labels
