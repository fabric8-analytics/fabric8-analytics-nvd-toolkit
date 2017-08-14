# fabric8-analytics-POCs
This repository contains POCs conducted for fabric8-analytics.

## Hyperion: a graph based model for stack analysis recommendations

Training involves ingesting the training-stacks into a graph. So, we need to start graph database as follows:
```
docker-compose -f docker-compose-hyperion.yml up dynamodb gremlin-http
```

Example scenario of training and scoring a Hyperion model is provided in test/unit_tests.py file.
