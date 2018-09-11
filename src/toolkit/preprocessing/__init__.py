"""Package containing pre-processing tools."""

from toolkit.preprocessing.handlers import GitHubHandler
from toolkit.preprocessing.preprocessors import \
    NVDFeedPreprocessor,\
    LabelPreprocessor,\
    NLTKPreprocessor

# make all Python checkers happy
assert GitHubHandler is not None
assert NVDFeedPreprocessor is not None
assert LabelPreprocessor is not None
assert NLTKPreprocessor is not None
