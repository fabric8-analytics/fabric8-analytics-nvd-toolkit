"""Tests for integration of components into pipelines."""

import tempfile
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

    @classmethod
    def setUpClass(cls):
        """Set up class level fixture."""
        cls.test_data = _get_test_data()

    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline."""
        # default prep pipeline
        pipeline = pipelines.get_preprocessing_pipeline(
            nvd_attributes=['project', 'description'],
            share_hooks=True
        )

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

        prep_data = pipeline.fit_transform(
            X=self.test_data,
            **fit_params
        )

        # sanity check
        self.assertLessEqual(len(prep_data), len(self.test_data))

        # check that prep_data is not empty
        # NOTE: this is a bit risky since there is no assurance that there
        # are suitable cves in the first n records
        self.assertTrue(any(prep_data))
        self.assertTrue(hasattr(prep_data[0], 'features'))  # default output
        self.assertTrue(hasattr(prep_data[0], 'label'))  # custom attribute

        # ---

        # custom attributes
        pipeline = pipelines.get_preprocessing_pipeline(
            nvd_attributes=['cve_id', 'project', 'description'],
            share_hooks=True  # reuse already existing hook
        )

        fit_params = {
            "%s__feed_attributes" % steps[2]: ['description'],
            "%s__output_attributes" % steps[2]: ['cve_id', 'label']
        }

        prep_data = pipeline.fit_transform(
            X=self.test_data,
            **fit_params
        )

        # sanity check
        self.assertLessEqual(len(prep_data), len(self.test_data))

        self.assertTrue(any(prep_data))
        self.assertTrue(hasattr(prep_data[0], 'features'))  # default output
        self.assertTrue(hasattr(prep_data[0], 'label'))  # custom attribute

    def test_training_pipeline(self):
        """Test training pipeline."""
        prep_pipeline = pipelines.get_preprocessing_pipeline(
            nvd_attributes=['project', 'description'],
            share_hooks=True
        )

        steps, preps = list(zip(*prep_pipeline.steps))
        fit_params = {
            "%s__feed_attributes" % steps[2]: ['description'],
            "%s__output_attributes" % steps[2]: ['label']
        }

        prep_data = prep_pipeline.fit_transform(
            X=self.test_data,
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

    def test_full_training_pipeline(self):
        """Test full training pipeline."""
        train_pipeline = pipelines.get_full_training_pipeline()
        steps, preps = list(zip(*train_pipeline.steps))
        fit_params = {
            "%s__attributes" % steps[0]: ['description'],
            "%s__feed_attributes" % steps[1]: ['project', 'description'],
            "%s__feed_attributes" % steps[2]: ['description'],
            "%s__output_attributes" % steps[2]: ['label']
        }

        clf = train_pipeline.fit_transform(
            X=self.test_data,
            **fit_params
        )

        self.assertIsNotNone(clf)
        self.assertIsInstance(clf, classifiers.NBClassifier)
        self.assertIsNotNone(clf.features)

    def test_prediction_pipeline(self):
        """Test pipeline prediction."""
        train_data, _ = pipelines.extract_labeled_features(
            self.test_data,
            nvd_attributes=['project', 'description'],
            nltk_feed_attributes=['description']
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
        featuresets = pipelines.extract_features(
            self.test_data,
            ['description']
        )

        self.assertTrue(any(featuresets))
        self.assertEqual(len(featuresets), len(self.test_data))

    def test_extract_labeled_features(self):
        """Test labeled feature extraction."""
        featuresets, labels = pipelines.extract_labeled_features(
            data=self.test_data,
            nvd_attributes=['project', 'description'],
            nltk_feed_attributes=['description']
        )

        self.assertTrue(np.any(featuresets))
        self.assertTrue(any(labels))

    def test_evaluation(self):
        """Test evaluation of extracted features."""
        featuresets, _ = pipelines.extract_labeled_features(
            data=self.test_data,
            nvd_attributes=['project', 'description'],
            nltk_feed_attributes=['description']
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

    @classmethod
    def setUpClass(cls):
        """Set up class level fixture."""
        cls.clf_path = _export_classifier()

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

    def test_main_default(self):
        """Test main function with default arguments."""
        argv = [
            '--from-feeds', 'recent',
            '-clf', self.clf_path
        ]
        ret_val = evaluation.main(argv)

        self.assertIsNone(ret_val)


class TestPredict(unittest.TestCase):
    """Tests for evaluation module."""

    @classmethod
    def setUpClass(cls):
        """Set up class level fixture."""
        cls.clf_path = _export_classifier()

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

    def test_main_default(self):
        """Test main function with default arguments."""
        argv = [
            '-clf', self.clf_path,
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

    def test_main_default(self):
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
    cve_iter = feed.cves()
    records = n_records

    data = [next(cve_iter) for _ in range(records)]  # only first n to speed up tests

    return data


def _export_classifier():
    """Set up for unit tests by exporting classifier."""
    raw_data = _get_test_data()

    data, _ = pipelines.extract_labeled_features(
        data=raw_data,
        nvd_attributes=['project', 'description'],
        nltk_feed_attributes=['description']
    )

    classifier = classifiers.NBClassifier()
    classifier = classifier.fit(data)

    tmp_dir = tempfile.mkdtemp(prefix='test_export_')

    pickle_path = classifier.export(export_dir=tmp_dir)

    return pickle_path
