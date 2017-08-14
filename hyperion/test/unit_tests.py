from hyperion.util.data_store.local_filesystem import LocalFileSystem
from hyperion.src.graph_based_reco import HyperionModel


def test_train_and_score_hyperion():
    input_data_store = LocalFileSystem(src_dir='vertx_data_small')
    assert (input_data_store is not None)

    hyperion = HyperionModel(graph_db_url="http://localhost:8181")
    hyperion.train(data_store=input_data_store)

    list_packages = [
        "com.englishtown.vertx:vertx-curator",
        "com.datastax.cassandra:cassandra-driver-core",
        "io.vertx:vertx-core",
        "com.englishtown.vertx:vertx-hk2",
        "io.vertx:vertx-service-proxy"
    ]
    output = hyperion.score(list_packages=list_packages)

    # Companions
    assert (set(output['companion_packages']) == {
        'com.englishtown.vertx:vertx-when',
        'com.englishtown.vertx:vertx-guice'
    })

    # Alternates
    assert (len(output['alternate_packages']) == 5)
    alternates = (output['alternate_packages'])
    assert(set(alternates['com.englishtown.vertx:vertx-curator']) == {
        "io.vertx:vertx-core",
        "com.englishtown.vertx:vertx-cassandra"
    })
    assert(len(alternates['com.datastax.cassandra:cassandra-driver-core']) == 0)
    assert(len(alternates['io.vertx:vertx-service-proxy']) == 0)
    assert(set(alternates['com.englishtown.vertx:vertx-hk2']) == {
        "com.englishtown.vertx:vertx-guice"
    })
    assert(set(alternates['io.vertx:vertx-core']) == {
        "com.englishtown.vertx:vertx-cassandra",
        "com.englishtown.vertx:vertx-curator"
    })

    # Outliers
    assert (set(output['outlier_packages']) == {
        'io.vertx:vertx-service-proxy'
    })
