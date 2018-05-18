"""Tests for integration of components into pipelines."""

import os
import unittest

import numpy as np

from toolkit.preprocessing import preprocessors
from toolkit.transformers import extractors, classifiers
from toolkit import pipelines

from toolkit.pipelines import evaluation
from toolkit.pipelines import predict
from toolkit.pipelines import train


class TestPipeline(unittest.TestCase):
    """Test classification pipeline."""

    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline."""
        # default prep pipeline
        pipeline = pipelines.get_preprocessing_pipeline()

        # check that the pipeline contains correct steps
        steps, preps = list(zip(*pipeline.steps))
        self.assertIsInstance(preps[0], preprocessors.NVDFeedPreprocessor)
        self.assertIsInstance(preps[1], preprocessors.LabelPreprocessor)
        self.assertIsInstance(preps[2], preprocessors.NLTKPreprocessor)

        # set up fit parameters (see sklearn fit_params notation)
        fit_params = {
            "%s__feed_attributes" % steps[2]: ['description'],
            "%s__output_attributes" % steps[2]: ['label']
        }

        test_data = _get_test_data()

        prep_data = pipeline.fit_transform(
            X=test_data,
            **fit_params
        )

        # sanity check
        self.assertLessEqual(len(prep_data), len(test_data))

        # check that prep_data is not empty
        # NOTE: this is a bit risky since there is no assurance that there
        # are suitable cves in the first n records
        self.assertTrue(any(prep_data))
        self.assertTrue(hasattr(prep_data[0], 'values'))  # default output
        self.assertTrue(hasattr(prep_data[0], 'label'))  # custom attribute

    def test_training_pipeline(self):
        """Test training pipeline."""
        test_data = _get_test_data()
        prep_pipeline = pipelines.get_preprocessing_pipeline()

        steps, preps = list(zip(*prep_pipeline.steps))
        fit_params = {
            "%s__feed_attributes" % steps[2]: ['description'],
            "%s__output_attributes" % steps[2]: ['label']
        }

        prep_data = prep_pipeline.fit_transform(
            X=test_data,
            **fit_params
        )

        # split the data
        prep_data = np.array(prep_data)

        features, labels = prep_data[:, 0], prep_data[:, 1]

        train_pipeline = pipelines.get_training_pipeline()
        _, trains = list(zip(*train_pipeline.steps))

        self.assertIsInstance(trains[0], extractors.FeatureExtractor)
        self.assertIsInstance(trains[1], classifiers.NBClassifier)

        clf = train_pipeline.fit_transform(
            X=features, y=labels
        )

        self.assertIsNotNone(clf)
        self.assertIsNotNone(clf.features)

    def test_prediction_pipeline(self):
        """Test pipeline prediction."""
        test_data = _get_test_data()
        train_data, _ = pipelines.extract_labeled_features(
            test_data,
            attributes=['description'],
        )

        clf = classifiers.NBClassifier().fit(train_data)

        pred_data = [
            'Sample project name prediction',
            'Sample project name prediction',
            'Sample project name prediction',
        ]

        pred_pipeline = pipelines.get_prediction_pipeline(
            clf,
        )

        n_candidates = 3
        predictions = pred_pipeline.fit_predict(pred_data,
                                                classifier__n=n_candidates,
                                                classifier__sample=True)

        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.shape[1], n_candidates)
        self.assertEqual(predictions.shape[-1], 2)  # candidate - proba

    def test_extract_features(self):
        """Test feature extraction."""
        test_data = _get_test_data()
        featuresets = pipelines.extract_features(
            test_data,
            ['description']
        )

        self.assertTrue(any(featuresets))
        self.assertEqual(len(featuresets), len(test_data))

    def test_extract_labeled_features(self):
        """Test labeled feature extraction."""
        test_data = _get_test_data()
        featuresets, labels = pipelines.extract_labeled_features(
            data=test_data,
            attributes=['description'],
        )

        self.assertTrue(any(featuresets))
        self.assertTrue(any(labels))

    def test_evaluation(self):
        """Test evaluation of extracted features."""
        test_data = _get_test_data()
        featuresets, _ = pipelines.extract_labeled_features(
            data=test_data,
            attributes=['description'],
        )

        clf = classifiers.NBClassifier().fit(featuresets)
        self.assertIsNotNone(clf)

        # evaluation == 0.0
        zero_labels = [None] * len(featuresets)
        score = clf.evaluate(featuresets, zero_labels, sample=True)

        self.assertIsNotNone(score)
        self.assertEqual(score, 0.0)

        score = classifiers.cross_validate(
            clf,
            featuresets,
            zero_labels,
            sample=True
        )

        self.assertIsNotNone(score)
        self.assertEqual(score.mean, 0.0)


class TestEvaluation(unittest.TestCase):
    """Tests for evaluation module."""

    def test_main_help(self):
        """Test argument parser's help."""
        argv = ['--help']

        with self.assertRaises(SystemExit) as exc:
            evaluation.main(argv)

        self.assertEqual(exc.exception.code, 0)

    def test_main_no_args(self):
        """Test main function with no arguments."""
        argv = []
        with self.assertRaises(SystemExit) as exc:
            evaluation.main(argv)

        self.assertNotEqual(exc.exception.code, 0)

    def _test_main_default(self):
        """Test main function with default arguments."""
        argv = [
            '--from-feeds', 'recent',
            '-clf',
            os.path.join(os.path.dirname(__file__), 'export/')
        ]
        ret_val = evaluation.main(argv)

        self.assertIsNone(ret_val)


class TestPredict(unittest.TestCase):
    """Tests for evaluation module."""

    def test_main_help(self):
        """Test argument parser's help."""
        argv = ['--help']

        with self.assertRaises(SystemExit) as exc:
            predict.main(argv)

        self.assertEqual(exc.exception.code, 0)

    def test_main_no_args(self):
        """Test main function with no arguments."""
        argv = []
        with self.assertRaises(SystemExit) as exc:
            predict.main(argv)

        self.assertNotEqual(exc.exception.code, 0)

    def _test_main_default(self):
        """Test main function with default arguments."""
        argv = [
            '-clf',
            os.path.join(os.path.dirname(__file__), 'export/'),
            'Sample description.'
        ]
        ret_val = predict.main(argv)

        self.assertIsNone(ret_val)


class TestTrain(unittest.TestCase):
    """Tests for evaluation module."""

    def test_main_help(self):
        """Test argument parser's help."""
        argv = ['--help']

        with self.assertRaises(SystemExit) as exc:
            train.main(argv)

        self.assertEqual(exc.exception.code, 0)

    def test_main_no_args(self):
        """Test main function with no arguments."""
        argv = []
        with self.assertRaises(SystemExit) as exc:
            train.main(argv)

        self.assertNotEqual(exc.exception.code, 0)

    def _test_main_default(self):
        """Test main function with default arguments."""
        argv = ['--from-feeds', 'recent']
        ret_val = train.main(argv)

        self.assertIsNone(ret_val)


def _get_test_data(n_records=500):
    """Return preprocessed data.

    Note: used for tests only.
    """
    from nvdlib.nvd import NVD

    feed = NVD.from_feeds(feed_names=['recent'])
    # download and update
    feed.update()

    # get the sample cves
    __cve_iter = feed.cves()
    __records = n_records

    data = [next(__cve_iter) for _ in range(__records)]  # only first n to speed up tests

    return data
