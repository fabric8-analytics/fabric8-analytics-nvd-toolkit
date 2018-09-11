"""Package containing transformers and classifiers."""

from toolkit.transformers.classifiers import NBClassifier, cross_validate
from toolkit.transformers.extractors import FeatureExtractor
from toolkit.transformers.hooks import Hook

# make all Python checkers happy
assert NBClassifier is not None
assert cross_validate is not None
assert FeatureExtractor is not None
assert Hook is not None
