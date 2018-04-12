#!/bin/env python3
"""This module contains prediction pipeline.

The pipeline restores pre-trained classifier to make predictions about
given data.
"""

import argparse
import os

from toolkit import pipelines
from toolkit.transformers import classifiers
from toolkit.pipelines.train import FEATURES


__parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
__parser.add_argument(
    '-n', '--num-candidates',
    type=int,
    default=3,
    help="Number of candidates to output by the classifier."
)

__parser.add_argument(
    '-clf', '--path-to-classifier',
    required=True,
    help="Path to the stored classifier checkpoints.",
)

__parser.add_argument(
    'description',
    help="The description to use for prediction.",
)


def restore_classifier(export_file: str) -> classifiers.NBClassifier:
    """Restores the classifier from given checkpoints."""
    if not os.path.isfile(export_file):
        raise FileNotFoundError("Incorrect path to classifier: File `{}` not found."
                                .format(export_file))

    return classifiers.NBClassifier.restore(export_file)


def main():
    args = __parser.parse_args()

    clf = restore_classifier(args.path_to_classifier)
    prediction_pipeline = pipelines.get_prediction_pipeline(
        classifier=clf,
        feature_hooks=FEATURES
    )

    prediction = prediction_pipeline.fit_predict(X=[args.description],
                                                 classifier__n=args.num_candidates,
                                                 classifier__sample=True)
    print(prediction)


if __name__ == '__main__':
    main()
