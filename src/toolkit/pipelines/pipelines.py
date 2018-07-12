"""Pipeline utility functions for API integration.

This module contains predefined pipelines for API integration.
Those pipelines are made to be used simply and effectively, however,
for more complex cases, it is suggested to build and optimize your own
pipeline from the blocks provided in this toolkit.

"""

import typing

import numpy as np

from sklearn.pipeline import Pipeline

from toolkit import preprocessing, transformers, utils


def get_preprocessing_pipeline(
        nvd_attributes: list,
        nltk_feed_attributes: list = None,
        labeling_func: typing.Callable = None,
        share_hooks=False) -> Pipeline:
    """Build the preprocessing pipeline using existing classifier.

    The preprocessing pipeline takes as an input a list of CVE objects
    and outputs labeled data ready for feature extraction.

    *must be fit using `fit_transform` method.*

    :param nvd_attributes: list, attributes to output by NVDPreprocessor

        The attributes are outputed by NVDPreprocessor and passed
        to FeatureExtractor.

    :param nltk_feed_attributes: list, attributes for NLTKPreprocessor

        List of attributes which will be fed to NLTKPreprocessor.

    :param labeling_func: callable object to be used for labeling

        The `labeling_func` is used to create a hook for `LabelPreprocessor`
        (see `LabelPreprocessor` documentation for more info).
        By default `toolkit.utils.find_` function is used for that purpose.

    :param share_hooks: boolean, whether to reuse hooks
    """
    if labeling_func is None:
        labeling_func = utils.find_

    return Pipeline(
        steps=[
            (
                'nvd_feed_preprocessor',
                preprocessing.NVDFeedPreprocessor(attributes=nvd_attributes)
            ),
            (
                'label_preprocessor',
                preprocessing.LabelPreprocessor(
                    feed_attributes=['project', 'description'],
                    # output only description attribute for NLTK processing
                    output_attributes=nvd_attributes,
                    hook=transformers.Hook(key='label_hook',
                                           func=labeling_func,
                                           reuse=share_hooks)
                ),
            ),
            (
                'nltk_preprocessor',
                preprocessing.NLTKPreprocessor(
                    feed_attributes=nltk_feed_attributes,
                )
            )
        ]
    )


def get_training_pipeline(feature_hooks=None) -> Pipeline:
    """Build the simple training pipeline from FeatureExtractor and NBClassifier.

    The pipeline expects as an input preprocessed data
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


def get_full_training_pipeline(labeling_func: typing.Callable = None,
                               feature_hooks=None,
                               share_hooks=False) -> Pipeline:
    """Build the full training pipeline with no predefined attributes.

    The pipeline accepts raw data, performs preprocessing and feature
    extraction and trains NBClassifier on that data.

    The customization of feed and output attributes is fully left to user.
    It is necessary to provide `fit_params` when fitting, as this pipeline
    does not contain any predefined arguments.

    *must be fit using `fit_transform` method with `fit_params`*

    :param feature_hooks: dict, {feature_key: Hook}
        to be used as an argument to `FeatureExtractor`

        Specify features which should be extracted from the given set.
        The hooks are called for each element of the set and return
        corresponding features.

    :param labeling_func: callable object to be used for labeling

        The `labeling_func` is used to create a hook for `LabelPreprocessor`
        (see `LabelPreprocessor` documentation for more info).
        By default `toolkit.utils.find_` function is used for that purpose.

    :param share_hooks: boolean, whether to reuse hooks

    :returns: Pipeline
    """
    if labeling_func is None:
        labeling_func = utils.find_

    return Pipeline(
        steps=[
            (
                'nvd_feed_preprocessor',
                preprocessing.NVDFeedPreprocessor()
            ),
            (
                'label_preprocessor',
                preprocessing.LabelPreprocessor(
                    hook=transformers.Hook(key='label_hook',
                                           reuse=share_hooks,
                                           func=labeling_func)
                )
            ),
            (
                'nltk_preprocessor',
                preprocessing.NLTKPreprocessor()
            ),
            (
                'feature_extractor',
                transformers.FeatureExtractor(
                    feature_hooks=feature_hooks,
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


def get_extraction_pipeline(attributes=None,
                            feature_hooks: list = None,
                            share_hooks=False) -> Pipeline:
    """Build the extraction pipeline.

    :param attributes: list, attributes for NLTKPreprocessor

        List of attributes which will be extracted from NVD and passed to NLTK
        preprocessor.

    :param feature_hooks: dict, {feature_key: Hook}
        to be used as an argument to `FeatureExtractor`

        Specify features which should be extracted from the given set.
        The hooks are called for each element of the set and return
        corresponding features.

    :param share_hooks: boolean, whether to reuse hooks
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
                    share_hooks=share_hooks
                )
            ),
        ]
    )


def extract_features(data: typing.Union[list, np.ndarray],
                     attributes: list = None,
                     nvd_attributes: list = None,
                     nltk_feed_attributes: list = None,
                     share_hooks=True,
                     **kwargs):
    """Extract data by fitting the extraction pipeline.

    :param data: input data to the pipeline
    :param attributes: list, attributes for NLTKPreprocessor

        List of attributes which will be extracted from NVD and passed to NLTK
        preprocessor.

    :param nvd_attributes: list, attributes to output by NVDPreprocessor

        The attributes are outputed by NVDPreprocessor and passed
        to FeatureExtractor.

        By default same as `attributes`.

    :param nltk_feed_attributes: list, attributes for NLTKPreprocessor

        List of attributes which will be fed to NLTKPreprocessor.

        By default same as `attributes`.

    :param share_hooks: bool, whether to reuse hooks
    :param kwargs: optional, key word arguments

        :feature_hooks: list of feature hooks to be used for feature extraction

    :returns: ndarray, featureset
    """
    if not any([attributes, nvd_attributes, nltk_feed_attributes]):
        raise ValueError("No attributes were provided.")

    feature_hooks = kwargs.get('feature_hooks', None)

    extraction_pipeline = get_extraction_pipeline(
        feature_hooks=feature_hooks,
        share_hooks=share_hooks
    )

    featureset = extraction_pipeline.fit_transform(
        data,
        # it is important not to filter the data by the handler here
        nvd_feed_preprocessor__attributes=nvd_attributes or attributes,
        nvd_feed_preprocessor__use_filter=False,
        nltk_preprocessor__feed_attributes=nltk_feed_attributes or attributes,
        nltk_preprocessor__output_attributes=nvd_attributes
    )

    return featureset


def extract_labeled_features(data: typing.Union[list, np.ndarray],
                             nvd_attributes: list,
                             nltk_feed_attributes: list = None,
                             feature_hooks: list = None,
                             labeling_func=None,
                             share_hooks=True) -> tuple:
    """Extract labeled features from input data.

     Extracts labeled features by concatenating and fitting the preprocessing
     and extraction pipeline.

     This is a wrapper for simplification of preprocessing and feature extraction.
     For full functionality it is suggested to build custom pipelines.

    :param data: input data to the preprocessing pipeline
    :param nvd_attributes: list, attributes to output by NVDPreprocessor

        The attributes are outputed by NVDPreprocessor and passed
        to FeatureExtractor.

    :param nltk_feed_attributes: list, attributes for NLTKPreprocessor

        List of attributes which will be fed to NLTKPreprocessor.

    :param feature_hooks: List[Hook], hooks used for feature extraction
    :param labeling_func: function used for labeling, passed to LabelPreprocessor
    :param share_hooks: bool, whether to reuse hooks

    :returns: tuple, (featureset, classification labels)
    """
    nltk_feed_attributes = nltk_feed_attributes or []

    prep_pipeline = get_preprocessing_pipeline(
        nvd_attributes=nvd_attributes,
        labeling_func=labeling_func,
        share_hooks=share_hooks
    )

    steps, _ = list(zip(*prep_pipeline.steps))
    fit_params = {
        "%s__feed_attributes" % steps[2]: nltk_feed_attributes,
        "%s__output_attributes" % steps[2]: nvd_attributes + ['label']
    }

    prep_data = prep_pipeline.fit_transform(
        X=data,
        **fit_params
    )

    # split the data
    extractor = transformers.FeatureExtractor(
        feature_hooks=feature_hooks
    )

    featuresets = extractor.fit_transform(X=prep_data)

    return featuresets, np.array(prep_data)[:, -1]
