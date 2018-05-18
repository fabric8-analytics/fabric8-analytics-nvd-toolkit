#!/bin/env python3
"""This module contains training pipeline.

The pipeline integrates preprocessors, transformers and classifier
to fit on the data.
"""

import argparse
import sys

from collections import namedtuple
from sklearn.pipeline import Pipeline
from time import time

from nvdlib.nvd import NVD

from toolkit import preprocessing, transformers
from toolkit.transformers import feature_hooks
from toolkit import utils


# create this helper class for easier addressing of relevant hooks
__FeatureHooks = namedtuple('FeatureHooks', [  # pylint: disable=invalid-name
    'has_uppercase_hook',
    'is_alnum_hook',
    'vendor_product_match_hook',
    'ver_pos_hook',
    'word_len_hook',
])

FEATURE_HOOKS = __FeatureHooks(*[
    feature_hooks.has_uppercase_hook,
    feature_hooks.is_alnum_hook,
    feature_hooks.vendor_product_match_hook,
    feature_hooks.ver_pos_hook,
    feature_hooks.word_len_hook,
])


def parse_args(argv):
    """Parse arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    feed_group = parser.add_mutually_exclusive_group(required=True)
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
        '--export',
        action='store_true',
        default=False
    )

    parser.add_argument(
        '-', '--export-dir',
        default='export'
    )

    return parser.parse_args(args=argv)


def main(argv):
    """Run."""
    args = parse_args(argv=argv)

    if args.csv:
        # TODO
        raise NotImplementedError("The feature has not been implemented yet."
                                  " Sorry for the inconvenience.")
    else:
        print("Getting NVD Feed...")
        feed = NVD.from_feeds(feed_names=args.nvd_feeds)
        feed.update()
        data = list(feed.cves())  # generator

    cve_dict = {cve.cve_id: cve for cve in data}

    # set up default argument for vendor-product feature hook
    feature_hooks.vendor_product_match_hook.default_kwargs = {
        'cve_dict': cve_dict
    }

    training_pipeline = Pipeline(
        steps=[
            (
                'nvd_feed_preprocessor',
                preprocessing.NVDFeedPreprocessor(
                    attributes=['cve_id', 'description']
                )
            ),
            (
                'label_preprocessor',
                preprocessing.LabelPreprocessor(
                    feed_attributes=['project', 'description'],
                    output_attributes=['cve_id', 'description'],
                    hook=transformers.Hook(key='label_hook',
                                           reuse=True,
                                           func=utils.find_)
                )
            ),
            (
                'nltk_preprocessor',
                preprocessing.NLTKPreprocessor(
                    feed_attributes=['description'],
                    output_attributes=['cve_id', 'label']
                )
            ),
            (
                'feature_extractor',
                transformers.FeatureExtractor(
                    feature_hooks=FEATURE_HOOKS,
                    share_hooks=True
                )
            ),
            (
                'classifier',
                transformers.NBClassifier()
            )
        ]
    )

    start_time = time()
    print("Training started")

    try:
        classifier = training_pipeline.fit_transform(X=data)
    finally:
        print(f"Training finished in {time() - start_time} seconds")

    if args.export:
        classifier.export(args.export_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
