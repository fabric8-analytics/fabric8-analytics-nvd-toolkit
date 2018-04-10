"""Tests for classifier module."""

import os
import tempfile
import unittest

import numpy as np

from toolkit.transformers import FeatureExtractor, NBClassifier, cross_validate
from toolkit.pipelines import get_preprocessing_pipeline


class TestClassifier(unittest.TestCase):
    """Tests for NBClassifier class."""

    def test_fit(self):
        """Test NBClassifier `fit` method."""
        classifier = NBClassifier()

        self.assertIsNotNone(classifier)
        self.assertIsInstance(classifier, NBClassifier)

        train_data = _get_extracted_test_data()

        classifier = classifier.fit(X=train_data)

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
        train_data = _get_extracted_test_data()
        classifier = classifier.fit(X=train_data)

        self.assertIsNotNone(classifier.features)

    def test_predict(self):
        classifier = NBClassifier()

        test_data = _get_extracted_test_data()

        features = test_data[0][1][1]  # sample features
        # should raise if wasn't fit before
        with self.assertRaises(ValueError):
            classifier.predict(features=features)

        classifier = classifier.fit(test_data)

        # make prediction
        prediction = classifier.predict(features=features, sample=True)

        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, float)

        # check with no sample specified
        prediction = classifier.predict(features=features, sample=None)
        self.assertIsInstance(prediction, dict)

    def test_fit_predict(self):
        classifier = NBClassifier()

        test_train_data = _get_extracted_test_data()
        classifier = classifier.fit(test_train_data)

        pred_data = [t[1] for t in test_train_data]
        num_candidates = 3
        candidates = classifier.fit_predict(pred_data,
                                            n=num_candidates,
                                            sample=True)

        self.assertIsInstance(candidates, np.ndarray)
        # check correct number of candidates
        self.assertEqual(len(candidates), num_candidates)
        # check that all predictions are the same (no flaw appeared)

    def test_export(self):
        """Test NBClassifier `export` method."""
        test_train_data = _get_extracted_test_data()

        classifier = NBClassifier()
        classifier = classifier.fit(test_train_data)

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
        test_train_data = _get_extracted_test_data()

        classifier = NBClassifier()
        classifier = classifier.fit(test_train_data)

        pickle_path = classifier.export()

        restored_classifier = NBClassifier.restore(pickle_path)

        self.assertIsInstance(restored_classifier, NBClassifier)
        self.assertEqual(classifier.features, restored_classifier.features)

    def test_evaluate(self):
        """Test NBClassifier `evaluate` method."""
        test_train_data = _get_extracted_test_data()

        classifier = NBClassifier()
        classifier.fit(test_train_data)

        # NOTE: `astype` is a workaround for the dtype incompatibility,
        # which was caused by prototyping the TEST_FEATURES for the test
        # purposes
        score = classifier.evaluate(
            X=test_train_data,
            y=[None] * len(test_train_data),
            sample=True
        )

        self.assertIsNotNone(score)
        self.assertEqual(score, 0.0)

    def test_cross_validate(self):
        """Test NBClassifier `cross_validate` method."""
        test_train_data = _get_extracted_test_data()

        classifier = NBClassifier()

        labels = [None] * len(test_train_data)
        score = cross_validate(
            classifier,
            test_train_data,
            labels,
            folds=5,
            shuffle=True,
            sample=True,
        )

        self.assertIsNotNone(score)
        self.assertEqual(score.mean, 0.0)


def _get_extracted_test_data():
    """Return preprocessed data.

    Note: used for tests only."""
    from nvdlib.nvd import NVD

    feed = NVD.from_feeds(feed_names=['recent'])
    # download and update
    feed.update()

    # get the sample cves
    __cve_iter = feed.cves()
    __records = 500

    data = [next(__cve_iter) for _ in range(__records)]  # only first n to speed up tests
    pipeline = get_preprocessing_pipeline()
    steps, preps = list(zip(*pipeline.steps))

    # set up fit parameters (see sklearn fit_params notation)
    fit_params = {
        "%s__feed_attributes" % steps[2]: ['description'],
        "%s__output_attributes" % steps[2]: ['label']
    }

    prep_data = pipeline.fit_transform(
        X=data,
        **fit_params
    )

    prep_data = np.array(prep_data)
    prep_data, prep_labels = prep_data[:, 0], prep_data[:, 1]

    data = FeatureExtractor().fit_transform(prep_data, prep_labels)
    print(data)

    return data
