# nvd-toolkit &nbsp; [![Build Status](https://ci.centos.org/buildStatus/icon?job=devtools-fabric8-analytics-nvd-toolkit-fabric8-analytics)](https://ci.centos.org/job/devtools-fabric8-analytics-nvd-toolkit-fabric8-analytics/)
 
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

### Footnotes

#### Check for all possible issues

The script named `check-all.sh` is to be used to check the sources for all detectable errors and issues. This script can be run w/o any arguments:

```
./check-all.sh
```

Expected script output:

```
Running all tests and checkers
  Check all BASH scripts
    OK
  Check documentation strings in all Python source file
    OK
  Detect common errors in all Python source file
    OK
  Detect dead code in all Python source file
    OK
  Run Python linter for Python source file
    OK
  Unit tests for this project
    OK
Done

Overal result
  OK
```

An example of script output when one error is detected:

```
Running all tests and checkers
  Check all BASH scripts
    Error: please look into files check-bashscripts.log and check-bashscripts.err for possible causes
  Check documentation strings in all Python source file
    OK
  Detect common errors in all Python source file
    OK
  Detect dead code in all Python source file
    OK
  Run Python linter for Python source file
    OK
  Unit tests for this project
    OK
Done

Overal result
  One error detected!
```

Please note that the script creates bunch of `*.log` and `*.err` files that are temporary and won't be commited into the project repository.

#### Coding standards

- You can use scripts `run-linter.sh` and `check-docstyle.sh` to check if the code follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [PEP 257](https://www.python.org/dev/peps/pep-0257/) coding standards. These scripts can be run w/o any arguments:

```
./run-linter.sh
./check-docstyle.sh
```

The first script checks the indentation, line lengths, variable names, white space around operators etc. The second
script checks all documentation strings - its presence and format. Please fix any warnings and errors reported by these
scripts.

List of directories containing source code, that needs to be checked, are stored in a file `directories.txt`

#### Code complexity measurement

The scripts `measure-cyclomatic-complexity.sh` and `measure-maintainability-index.sh` are used to measure code complexity. These scripts can be run w/o any arguments:

```
./measure-cyclomatic-complexity.sh
./measure-maintainability-index.sh
```

The first script measures cyclomatic complexity of all Python sources found in the repository. Please see [this table](https://radon.readthedocs.io/en/latest/commandline.html#the-cc-command) for further explanation how to comprehend the results.

The second script measures maintainability index of all Python sources found in the repository. Please see [the following link](https://radon.readthedocs.io/en/latest/commandline.html#the-mi-command) with explanation of this measurement.

You can specify command line option `--fail-on-error` if you need to check and use the exit code in your workflow. In this case the script returns 0 when no failures has been found and non zero value instead.

#### Dead code detection

The script `detect-dead-code.sh` can be used to detect dead code in the repository. This script can be run w/o any arguments:

```
./detect-dead-code.sh
```

Please note that due to Python's dynamic nature, static code analyzers are likely to miss some dead code. Also, code that is only called implicitly may be reported as unused.

Because of this potential problems, only code detected with more than 90% of confidence is reported.

List of directories containing source code, that needs to be checked, are stored in a file `directories.txt`

#### Common issues detection

The script `detect-common-errors.sh` can be used to detect common errors in the repository. This script can be run w/o any arguments:

```
./detect-common-errors.sh
```

Please note that only semantical problems are reported.

List of directories containing source code, that needs to be checked, are stored in a file `directories.txt`

#### Check for scripts written in BASH

The script named `check-bashscripts.sh` can be used to check all BASH scripts (in fact: all files with the `.sh` extension) for various possible issues, incompatibilities, and caveats. This script can be run w/o any arguments:

```
./check-bashscripts.sh
```

Please see [the following link](https://github.com/koalaman/shellcheck) for further explanation, how the ShellCheck works and which issues can be detected.

#### Code coverage report

Code coverage is reported via the codecov.io. The results can be seen on the following address:

[code coverage report](https://codecov.io/gh/fabric8-analytics/fabric8-analytics-nvd-toolkit)

<br>

> Marek Cermak <macermak@redhat.com>
