#!/bin/env python3
"""This module contains training pipeline.

The pipeline integrates preprocessors, transformers and classifier
to fit on the data.
"""

import argparse
import os
import sys

import numpy as np

from nvdlib.nvd import NVD
from sklearn.model_selection import train_test_split

from toolkit import pipelines
from toolkit.transformers import classifiers
from toolkit.pipelines.train import FEATURE_HOOKS
from toolkit.utils import BooleanAction


def parse_args(argv):
    """Parse arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    feed_group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument(
        '-clf', '--path-to-classifier',
        required=True,
        help="Path to the stored classifier checkpoints.",
    )

    feed_group.add_argument(
        '--from-feeds',
        type=str,
        nargs='+',
        dest='nvd_feeds',
        help="On ore more NVD Feeds to be chosen to train on."
    )

    feed_group.add_argument(
        '--from-csv',
        dest='csv',
        help="train on the custom data from `*.csv` format\n"
             "**NOTE:** The csv data must contain the relevant attributes infered "
             "by preprocessors."
    )

    parser.add_argument(
        '--eval', '--no-eval',
        action=BooleanAction,
        default=True
    )

    parser.add_argument(
        '-xval', '--cross-validate', '--no-cross-validate',
        action=BooleanAction,
        default=True
    )

    parser.add_argument(
        '-n', '--num-candidates',
        type=int,
        default=3,
        help="Number of candidates to output by the classifier."
    )

    parser.add_argument(
        '-xvn', '--cross-validation-folds',
        type=int,
        default=10,
    )
    return parser.parse_args(args=argv)


def main(argv):
    """Run."""
    args = parse_args(argv)

    if args.csv:
        # TODO
        raise NotImplementedError("The feature has not been implemented yet."
                                  " Sorry for the inconvenience.")
    else:
        print("Getting NVD Feed...")
        feed = NVD.from_feeds(feed_names=args.nvd_feeds)
        feed.update()
        data = feed.cves()  # generator

    # transform and transform the data with the pre-processing pipeline
    print("Preprocessing...")
    features, labels = pipelines.extract_labeled_features(
        data=data,
        feature_hooks=FEATURE_HOOKS,
        nvd_attributes=['description'],
    )
    print("Preprocessing done.")

    if not data:
        print("No data left after preprocessing. Check the data provided"
              " or modify preprocessing pipeline.", file=sys.stderr)
        exit(1)

    path_to_classifier = os.path.join(os.getcwd(), args.path_to_classifier)
    classifier = classifiers.NBClassifier.restore(path_to_classifier)

    # noinspection PyPep8Naming
    X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
        features, labels,
        test_size=0.2,
        random_state=np.random.randint(0, 100),
        shuffle=True
    )

    if args.eval:
        score = classifier.evaluate(X_test, y_test, sample=True, n=args.num_candidates)

        print("Evaluation accuracy:", score)

    if args.cross_validate:
        score = classifiers.cross_validate(
            classifier,
            X_train,
            y_train,
            sample=True,
            n=args.num_candidates,
            folds=args.cross_validation_folds,
            shuffle=True
        )

        print("Cross-validation results:")
        print("-------------------------")
        print("\tIntermediate results:\n")
        print(
            "\n".join("\t\tFold {}: {}".format(fold, np.round(value, 2))
                      for fold, value in enumerate(score.values))
        )
        print("\tAccuracy: %.2f (+/- %.4f)" % (np.round(score.mean, 2), np.round(score.std * 2, 4)))


if __name__ == '__main__':
    main(sys.argv[1:])
