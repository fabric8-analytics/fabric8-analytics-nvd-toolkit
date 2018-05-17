"""Python setup script."""

import os
import sys

from setuptools import find_packages, setup

BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")

# When executing the setup.py, we need to be able to import ourselves, this
# means that we need to add the src/ directory to the sys.path.
sys.path.insert(0, SRC_DIR)

ABOUT = dict()
with open(os.path.join(SRC_DIR, 'toolkit', '__about__.py')) as f:
    exec(f.read(), ABOUT)

setup(
    name=ABOUT['__title__'],
    version=ABOUT['__version__'],

    author=ABOUT['__author__'],
    author_email=ABOUT['__email__'],
    url=ABOUT['__uri__'],

    license=ABOUT['__license__'],

    description=ABOUT['__summary__'],
    long_description=(
        "The toolkit currently aims to provide the possibility to construct custom "
        "processing and classification pipelines. To do that, various preprocessors, "
        "transformers and classifiers can be used and customized "
        "with Hooks and attribute extractors."
    ),

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License"
    ],

    package_dir={"": "src"},
    packages=find_packages(where='src'),

    # explicit setup requirement in order to install scipy
    install_requires=[
        "requests",
        "urllib3",
        "nltk",
        "scikit-learn",
    ],

    tests_require=[
        "unittest2",
        "pytest",
        "scikit-learn",
    ],
)
