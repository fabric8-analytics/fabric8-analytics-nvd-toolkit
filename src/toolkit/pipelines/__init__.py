"""Package containing pre-build pipelines and CLIs."""

__all__ = [
    "toolkit.pipelines.pipelines.get_preprocessing_pipeline",
    "toolkit.pipelines.pipelines.get_training_pipeline",
    "toolkit.pipelines.pipelines.get_full_training_pipeline",
    "toolkit.pipelines.pipelines.get_prediction_pipeline",
    "toolkit.pipelines.pipelines.extract_features",
    "toolkit.pipelines.pipelines.extract_labeled_features",
    "toolkit.pipelines.predict.restore_classifier",
    "toolkit.pipelines.train.FEATURE_HOOKS"
]

from toolkit.pipelines.pipelines import \
    get_preprocessing_pipeline,\
    get_training_pipeline,\
    get_full_training_pipeline,\
    get_prediction_pipeline,\
    extract_features,\
    extract_labeled_features

from toolkit.pipelines.train import FEATURE_HOOKS

# make all Python checkers happy
assert get_preprocessing_pipeline is not None
assert get_training_pipeline is not None
assert get_full_training_pipeline is not None
assert get_prediction_pipeline is not None
assert extract_features is not None
assert extract_labeled_features is not None
assert FEATURE_HOOKS is not None
