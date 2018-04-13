# nvd-toolkit
 
###### Tools for processing National Vulnerability Database feeds. 

<br>

#### Installation steps and requirements

The toolkit currently uses [nvdlib] to download the [NVD feed]
and assumes CVE object to compy with the [CVE](https://github.com/msrb/nvdlib/blob/master/nvdlib/model.py)
model schema.

To install [nvdlib] first, proceed with these steps:
```bash
git clone https://github.com/msrb/nvdlib
cd nvdlib
python3 setup.py install
```

[nvdlib]: (https://github.com/msrb/nvdlib)

**NOTE:**
The `toolkit` is currently under development and setup is yet to be done,
you can try it out, however, by exploring [examples](/examples).
Some information is also provided by the [examples/README.md](/examples/README.md).

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
