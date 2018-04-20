# nvd-toolkit
 
###### Tools for processing National Vulnerability Database feeds. 

<br>

#### Installation steps and requirements

If you don't have [SciPy](https://www.scipy.org/) installed, issue the following command
```bash
pip3 install scipy
```

Which will install the SciPy along with its dependencies (like NumPy etc.).
This is because SciPy requires dependencies beyond PyPI (and python, actually),
hence it needs to be installed separately.

The toolkit currently uses [nvdlib] to download the [NVD feed]
and assumes CVE object to comply with the [CVE](https://github.com/msrb/nvdlib/blob/master/nvdlib/model.py)
model schema.
Since the [nvdlib] is not part of PyPI, it is necessary to install it manually.

To install [nvdlib], proceed with these steps:
```bash
git clone https://github.com/msrb/nvdlib
cd nvdlib
python3 setup.py install
```

[nvdlib]: (https://github.com/msrb/nvdlib)
Now you can easily proceed with a standard installation process.

```python3 setup.py install```


It is suggested to work in virtual environment. The whole setup can look like this:
```bash
# set up venv
python3 -m venv env
source env/bin/activate
# install scipy
pip install scipy
# install nvdlib
git clone https://github.com/msrb/nvdlib /tmp/nvdlib
pushd /tmp/nvdlib
python setup.py install
popd
# install the toolkit
python setup.py install
```

#### Getting started

The `toolkit` is currently under development, you can try it out, however,
by exploring [examples](/examples).
More information about the examples is provided by the [examples/README.md](/examples/README.md).

<br>

#### Key Concepts of the toolkit

Currently the toolkit aims at extracting CVE information from its description in the [NVD]
feed.
It provides a set of tools to pre-process the data, extract relevant information
and make inference (see [classifiers module](/src/toolkit/transformers/classifiers.py)).

Such tools can be assembled together in *pipelines* to improve performance, provide code
readability and ease of use.
The concept of the pipelines is inspired by [scikit-learn]
and is also used in the similar manner (see the [examples](/exampes) to get familiar with
the pipelines).

Some of the building blocks of the pipelines can be fed custom [hooks](/src/toolkit/transformers/hooks.py)
and therefore modified to the users purpose.

[NVD]: https://nvd.nist.gov/
[NVD feed]: https://nvd.nist.gov/vuln/data-feeds#JSON_FEED
[scikit-learn]: (http://scikit-learn.org/stable/)

<br>

> Marek Cermak <macermak@redhat.com>
