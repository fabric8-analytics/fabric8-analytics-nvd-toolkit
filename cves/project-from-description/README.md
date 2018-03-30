# This repository contains POC for issue #2485

> Assignee: Marek Cermak <macermak@redhat.com>\
> Issue: https://github.com/openshiftio/openshift.io/issues/2485 

### Description

> Mapping CVE entries to actual package names is much easier when we at least know name of a project (e.g. "Apache NiFi", or "Apache POI") that is affected by given vulnerability. Knowing the project name will help us to get better results and less false positives.

#### Input

One [NVD] CVE record.


#### Output

The output of this task should be a function that takes one NVD CVE record
on input and returns list of possible project name candidates.

Having confidence score for each candidate would be nice, but is not necessary.

---

# POC

#### Initial intention:  
Since it is not evident whether the NVD descriptions evince a latent pattern,
the first part of the POC will focus on exploring whether a pattern is present and to what
extent it can be used to predict project name candidates.

If such pattern is discovered, proceed with implementation of classifier and evaluate
its accuracy.


#### Sub tasks
- [x] Have a set of labeled data to train, validate and test accuracy with.
- [x] Discover whether the data evinces latent pattern. 
- [x] Model selection based on the description pattern properties
- [x] Classifier implementation
- [x] Accuracy evaluation


#### Sub tasks evaluation
- The data used for this task was a sub set of the [NVD] record which directly references
GitHub. This allows for labeling the data with the project name infered from the GitHub repository

- To discover a latent pattern, a Naive Bayes Classifier for chosen for a model.
Vanilla feature extractors were based on simple text feature_keys such as positional tags.
[NLTK] was used for these purposes.

- Naive Bayes Classifier provided decent results on the toy data set and hence with feature
extractor improvements, it was selected as a base model for this POC.

#### Conclusions

# TODO

[NLTK]: https://www.nltk.org/
[NVD]: https://nvd.nist.gov/


