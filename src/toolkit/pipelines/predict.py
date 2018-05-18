#!/bin/env python3
"""This module contains prediction pipeline.

The pipeline restores pre-trained classifier to make predictions about
given data.
"""

import argparse
import sys
import textwrap

from toolkit import pipelines
from toolkit.pipelines.train import FEATURE_HOOKS
from toolkit.transformers.classifiers import NBClassifier


def parse_args(argv):
    """Parse arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-clf', '--path-to-classifier',
        required=True,
        help="Path to the stored classifier checkpoints.",
    )
    parser.add_argument(
        '-n', '--num-candidates',
        type=int,
        default=3,
        help="Number of candidates to output by the classifier."
    )

    parser.add_argument(
        'description',
        help="The description to use for prediction.",
    )

    return parser.parse_args(args=argv)


def main(argv):
    """Run."""
    args = parse_args(argv)

    clf = NBClassifier.restore(args.path_to_classifier)
    prediction_pipeline = pipelines.get_prediction_pipeline(
        classifier=clf,
        feature_hooks=FEATURE_HOOKS
    )

    prediction, = prediction_pipeline.fit_predict(
        X=[args.description],
        feature_extractor__skip_unfed_hooks=True,
        classifier__n=args.num_candidates,
        classifier__sample=True
    )

    print("Prediction results:")
    print("-------------------")
    for (name, tag), score in prediction:
        formated_prediction = """\
        Candidate : {name}
        Tag       : {tag}
        Confidence: {score}
        """.format(name=name, tag=tag, score=score)

        print(textwrap.dedent(formated_prediction))


if __name__ == '__main__':
    main(sys.argv[1:])
