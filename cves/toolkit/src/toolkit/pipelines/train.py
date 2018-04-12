#!/bin/env python3
"""This module contains training pipeline.

The pipeline integrates preprocessors, transformers and classifier
to fit on the data.
"""

import argparse

from nvdlib.nvd import NVD

from toolkit import pipelines


__parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
feed_group = __parser.add_mutually_exclusive_group(required=True)
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
    '--export',
    action='store_true',
    default=False
)

FEATURES = None


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
    # print("Preprocessing...")
    # prep_pipeline = pipelines.get_preprocessing_pipeline()
    # steps, preps = list(zip(*prep_pipeline.steps))
    # fit_params = {
    #     "%s__feed_attributes" % steps[2]: ['description'],
    #     "%s__output_attributes" % steps[2]: ['label']
    # }
    #
    # prep_data = prep_pipeline.fit_transform(
    #     X=prep_data,
    #     **fit_params
    # )
    # print("Preprocessing done.")
    #
    # if not prep_data.size > 0:
    #     print("No data left after preprocessing. Check the data provided"
    #           " or modify preprocessing pipeline.", file=sys.stderr)
    #     exit(1)
    #
    # # split the data to labels
    # features, labels = prep_data[:, 0], prep_data[:, 1]
    print("Preprocessing...")
    features, labels = pipelines.extract_labeled_features(
        data=data,
        attributes=['description']
    )

    print("Preprocessing done.")

    print("Training...")
    # transform and transform the data with the training pipeline
    train_pipeline = pipelines.get_training_pipeline(feature_hooks=FEATURES)

    classifier = train_pipeline.fit_transform(
        X=features, y=labels
    )
    print("Training done.")

    if args.export:
        classifier.export()


if __name__ == '__main__':
    main()
