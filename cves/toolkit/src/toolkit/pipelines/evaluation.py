#!/bin/env python3
"""This module contains training pipeline.

The pipeline integrates preprocessors, transformers and classifier
to fit on the data.
"""

import argparse
import sys

import numpy as np

from nvdlib.nvd import NVD
from sklearn.model_selection import train_test_split

from toolkit import pipelines
from toolkit.transformers import extractors, classifiers
from toolkit.pipelines.train import FEATURES

__parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
feed_group = __parser.add_mutually_exclusive_group(required=True)

__parser.add_argument(
    '-p', '--path-to-classifier',
    required=True,
    help="Path to the stored classifier checkpoints.",
)

feed_group.add_argument(
    '--from-feeds',
    type=int,
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

__parser.add_argument(
    '-eval', '--evaluate',
    action='store_true',
    default=False
)

__parser.add_argument(
    '-xval', '--cross-validate',
    action='store_true',
    default=False
)

__parser.add_argument(
    '-n', '--num-candidates',
    type=int,
    default=3,
    help="Number of candidates to output by the classifier."
)

__parser.add_argument(
    '-xvn', '--cross-validation-folds',
    type=int,
    default=10,
)


# noinspection PyUnusedLocal
def main():
    args = __parser.parse_args()

    if args.csv:
        raise NotImplementedError
        # TODO
    else:
        print("Getting NVD Feed...")
        feed = NVD.from_feeds(feed_names=args.nvd_feeds)
        feed.update()
        data = feed.cves()  # generator

    # transform and transform the data with the pre-processing pipeline
    print("Preprocessing...")
    data, labels = pipelines.extract_labeled_features(
        X=data,
        nltk_preprocessor__feed_attributes=['description'],
    )
    print("Preprocessing done.")

    if not data:
        print("No data left after preprocessing. Check the data provided"
              " or modify preprocessing pipeline.", file=sys.stderr)
        exit(1)

    classifier = classifiers.NBClassifier.restore(args.path_to_classifier)

    extractor = extractors.FeatureExtractor(
        features=FEATURES
    )
    featureset = extractor.fit_transform(X=features, y=labels)

    # noinspection PyPep8Naming
    X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
        featureset, labels,
        test_size=0.2,
        random_state=np.random.randint(0, 100),
        shuffle=True
    )

    if args.evaluate:
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
    main()
