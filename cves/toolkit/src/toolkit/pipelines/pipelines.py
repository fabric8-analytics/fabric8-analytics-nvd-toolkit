"""Pipeline getters for API integration."""

import typing

import numpy as np

from sklearn.pipeline import Pipeline

from toolkit import preprocessing, transformers, utils


def get_preprocessing_pipeline(attributes: list = None,
                               labeling_func=None) -> Pipeline:
    """Build the preprocessing pipeline using existing classifier.

    The preprocessing pipeline takes as an input a list of CVE objects
    and outputs labeled data ready for feature extraction.

    *must be fit using `fit_transform` method.*

    :param attributes: list, attributes for NLTKPreprocessor

        List of attributes which will be extracted from NVD and passed to NLTK
        preprocessor.

    :param labeling_func: function object to be used for labeling

        The `labeling_func` is used to create a hook for `LabelPreprocessor`
        (see `LabelPreprocessor` documentation for more info).
        By default `toolkit.utils.find_` function is used for that purpose.
    """

    if labeling_func is None:
        labeling_func = utils.find_

    return Pipeline(
        steps=[
            (
                'nvd_feed_preprocessor',
                preprocessing.NVDFeedPreprocessor(attributes=attributes)
            ),
            (
                'label_preprocessor',
                preprocessing.LabelPreprocessor(
                    feed_attributes=['project', 'description'],
                    # output only description attribute for NLTK processing
                    output_attributes=attributes,
                    hook=transformers.Hook(key='label_hook', func=labeling_func)
                ),
            ),
            (
                'nltk_preprocessor',
                preprocessing.NLTKPreprocessor(
                    feed_attributes=attributes
                )
            )
        ]
    )


def get_training_pipeline(feature_hooks=None) -> Pipeline:
    """Build the training pipeline from FeatureExtractor and NBClassifier.

    The training pipeline expects as an input preprocessed data
    and trains NBClassifier on that data.

    *must be fit using `fit_transform` method.*

    :param feature_hooks: dict, {feature_key: Hook}
        to be used as an argument to `FeatureExtractor`

        Specify features which should be extracted from the given set.
        The hooks are called for each element of the set and return
        corresponding features.
    """

    return Pipeline(
        steps=[
            (
                'feature_extractor',
                transformers.FeatureExtractor(
                    feature_hooks=feature_hooks,
                    # make hooks sharable (useful if training pipeline was used before)
                    share_hooks=True
                )
            ),
            (
                'classifier',
                transformers.NBClassifier()
            )
        ]
    )


def get_prediction_pipeline(classifier: transformers.NBClassifier,
                            attributes: list = None,
                            feature_hooks: list = None) -> Pipeline:
    """Build the prediction pipeline using existing classifier.

    *must be fit using `fit_predict` method.*

    :param classifier: pre-trained NBClassifier
    :param attributes: list, attributes for NLTKPreprocessor

        List of attributes which will be extracted from NVD and passed to NLTK
        preprocessor.

    :param feature_hooks: dict, {feature_key: Hook}
        to be used as an argument to `FeatureExtractor`

        Specify features which should be extracted from the given set.
        The hooks are called for each element of the set and return
        corresponding features.
    """

    return Pipeline(
        steps=[
            (
                'nltk_preprocessor',
                preprocessing.NLTKPreprocessor(
                    feed_attributes=attributes
                )
            ),
            (
                'feature_extractor',
                transformers.FeatureExtractor(
                    feature_hooks=feature_hooks,
                    # make hooks sharable (useful if training pipeline was used before)
                    share_hooks=True
                )
            ),
            (
                'classifier',
                classifier
            )
        ]
    )


def get_extraction_pipeline(attributes,
                            feature_hooks: list = None) -> Pipeline:
    """Build the extraction pipeline.

    :param attributes: list, attributes for NLTKPreprocessor

        List of attributes which will be extracted from NVD and passed to NLTK
        preprocessor.

    :param feature_hooks: dict, {feature_key: Hook}
        to be used as an argument to `FeatureExtractor`

        Specify features which should be extracted from the given set.
        The hooks are called for each element of the set and return
        corresponding features.
    """

    return Pipeline(
        steps=[
            (
                'nvd_feed_preprocessor',
                preprocessing.NVDFeedPreprocessor(attributes=attributes)
            ),
            (
                'nltk_preprocessor',
                preprocessing.NLTKPreprocessor(
                    feed_attributes=attributes
                )
            ),
            (
                'feature_extractor',
                transformers.FeatureExtractor(
                    feature_hooks=feature_hooks,
                    # make hooks sharable (useful if training pipeline was used before)
                    share_hooks=True
                )
            ),
        ]
    )


def extract_features(
        data: typing.Union[list, np.ndarray],
        attributes: list,
        **kwargs):
    """Extract data by fitting the extraction pipeline.

    :returns: ndarray, featureset
    """

    feature_hooks = kwargs.get('feature_hooks', None)
    extraction_pipeline = get_extraction_pipeline(
        attributes=attributes,
        feature_hooks=feature_hooks
    )

    steps, _ = list(zip(*extraction_pipeline.steps))

    featureset = extraction_pipeline.fit_transform(data)

    return featureset


def extract_labeled_features(
        data: typing.Union[list, np.ndarray],
        attributes: list,
        **kwargs) -> tuple:
    """Extract data by concatenating and fitting
    the preprocessing and extraction pipeline.

    :returns: tuple, (featureset, classification labels)
    """

    labeling_func = kwargs.get('labeling_func', None)
    prep_pipeline = get_preprocessing_pipeline(
        labeling_func=labeling_func
    )

    steps, preps = list(zip(*prep_pipeline.steps))
    fit_params = {
        "%s__feed_attributes" % steps[2]: attributes,
        "%s__output_attributes" % steps[2]: ['label']
    }

    prep_data = prep_pipeline.fit_transform(
        X=data,
        **fit_params
    )
    del data

    # split the data
    prep_data = np.array(prep_data)
    features, labels = prep_data[:, 0], prep_data[:, 1]

    extractor = transformers.FeatureExtractor()

    featuresets = extractor.fit_transform(
        X=features, y=labels
    )

    return featuresets, labels
