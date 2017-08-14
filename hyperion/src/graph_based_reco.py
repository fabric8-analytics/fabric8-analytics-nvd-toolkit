# imports ...
import json
import requests
from uuid import uuid4

gremlin_method_insert_pkg_version =  """
        def insert_package_version(g, ecosystem, name, version) {
        def pred_pkg = g.V().has('vertex_label', 'Package').has('name', name).has('ecosystem', ecosystem);
        def pkg_vertex = (pred_pkg.hasNext()) ? pred_pkg.next() : g.addV('vertex_label', 'Package', 'name', name, 'ecosystem', ecosystem).next()
        def pred_version = g.V().has('vertex_label', 'Version').has('pecosystem', ecosystem).has('pname', name).has('version', version);
        if (!pred_version.hasNext()) {
            def version_vertex = g.addV('vertex_label', 'Version', 'pecosystem', ecosystem, 'pname', name, 'version', version).next();
            pkg_vertex.addEdge('has_version', version_vertex);
        }
    }
    """

gremlin_method_insert_ref_stack = """
    def insert_ref_stack(g, sid, sname, secosystem, usage, source, is_ref_stack, dependencies) {
        def pred_stack = g.V().has('vertex_label', 'Stack').has('sname', sname).has('secosystem', secosystem)
        if (!pred_stack.hasNext()) {
            def stack_vertex = g.addV('vertex_label','Stack','sname', sname, 'secosystem', secosystem, 'usage', usage, 'source', source, 'is_ref_stack', is_ref_stack, 'sid', sid).next();
            for (k in dependencies.keySet()) {
                def package_vertex = g.V().has('vertex_label', 'Package').has('ecosystem', secosystem).has('name', k).next();
                stack_vertex.addEdge('has_dependency', package_vertex);
            }
        }
    }
    """


def fire_gremlin(graph_db_url, str_gremlin, print_result=False):
    payload = {'gremlin': str_gremlin}
    response = requests.post(graph_db_url, data=json.dumps(payload))

    # TODO: check for error and raise exception
    if response.status_code != 200:
        print ("ERROR %d: %s") % (response.status_code, response.reason)

    json_response = response.json()
    if print_result:
        print json.dumps(json_response, indent=4)
    return json_response


def gremlin_str_pkg_version(ecosystem, pkg_name, version):
    return """insert_package_version(g, '{}', '{}', '{}');
    """.format(ecosystem, pkg_name, version)


def gremlin_str_ref_stack(ref_stack):
    stack_name = ref_stack.get('name')
    eco_system = ref_stack.get('ecosystem')
    usage = ref_stack.get('usage')
    source = 'some-source'
    is_ref_stack = ref_stack.get('is_ref_stack')
    sid = uuid4().hex
    tmp_list = map(lambda x: "'{}':'{}'".format(x[0], x[1]), ref_stack.get('dependencies').items())
    dependencies = "[" + ','.join(tmp_list) + "]"
    return """insert_ref_stack(g, '{}', '{}', '{}', '{}', '{}', '{}', {});
    """.format(sid, stack_name, eco_system, usage, source, is_ref_stack, dependencies)


def gremlin_str_package_tags(ecosystem, pkg_name, list_tags):
    gremlin_str = \
        """
        def pred_pkg = g.V().has('vertex_label', 'Package').has('name', '{name}').has('ecosystem', '{ecosystem}');
        def pkg_vertex = (pred_pkg.hasNext()) ? pred_pkg.next() : g.addV('vertex_label', 'Package', 'name', '{name}', 'ecosystem', '{ecosystem}').next();
        """.format(name=pkg_name, ecosystem=ecosystem)

    for tag in list_tags:
        gremlin_str += "pkg_vertex.property(set, 'tags', '{tag}'); ".format(tag=tag)

    return gremlin_str


def gremlin_str_stack_tags(stk_name, list_tags):
    gremlin_str = \
    """
    def pred_stack = g.V().has('vertex_label', 'Stack').has('sname', '{name}')
    def stk_vertex = pred_stack.next();
    """.format(name=stk_name)

    for tag in list_tags:
        gremlin_str += "stk_vertex.property(set, 'tags', '{tag}'); ".format(tag=tag)

    return gremlin_str


def create_reference_stack(dependencies):
    ref_stack = {}  # Empty map
    if dependencies is not None:
        list_list_pv = map(lambda x: x.split(' '), dependencies)
        pv_list = map(lambda x: (x[0], x[1]), list_list_pv)
        pk_names = map(lambda x: x[0], pv_list)
        stack_name = "@@".join(pk_names)
        stack_deps = dict(pv_list)
        stack_license = "Unknown"
        stack_ecosystem = "vertx"
        ref_stack = {
            "name": stack_name,
            "ecosystem": stack_ecosystem,
            "dependencies": stack_deps,
            "usage": 1,
            "is_ref_stack": 'true'
        }
    return ref_stack


class HyperionModel(object):
    """
    Graph based recommendation for stack analysis.
    """
    def __init__(self, graph_db_url):
        self.graph_db_url = graph_db_url
        self._init_graph_schema()

    def _init_graph_schema(self):
        str_gremlin = \
        """
        mgmt = graph.openManagement(); 
        tags = mgmt.getPropertyKey('tags'); 
        if(tags == null) { 
            tags = mgmt.makePropertyKey('tags').dataType(String.class).cardinality(Cardinality.SET).make();
        }
        mgmt.commit();
        """
        fire_gremlin(self.graph_db_url, str_gremlin)

    def train(self, data_store):
        """
        Train the Hyperion model by ingesting the training data that consists of
        known stacks and known package-topics.

        In the end of training, the graph will have the following vertices:
          a. packages with appropriate tags
          b. versions of a package
          c. stacks with appropriate tags

        Note: It's a POC work ! Various error scenarios are not handled.

        :param data_store: data store to read the input data from
        :return: None
        """
        stacks_data = data_store.read_json_file('training_stacks.json')
        topics_data = data_store.read_json_file('package_topics.json')

        # Insert vertices for packages and versions
        for stk in stacks_data['stacks']:
            list_list_pv = map(lambda x: x.split(' '), stk['dependencies'])
            list_gremlin_epv = map(lambda x: gremlin_str_pkg_version('vertx', x[0], x[1]), list_list_pv)
            str_gremlin = gremlin_method_insert_pkg_version + ' '.join(list_gremlin_epv)
            fire_gremlin(self.graph_db_url, str_gremlin)

        # Insert known training stacks
        # Also, add tags for packages and stacks
        for stk in stacks_data['stacks']:
            ref_stack = create_reference_stack(stk['dependencies'])
            str_gremlin = gremlin_method_insert_ref_stack + gremlin_str_ref_stack(ref_stack)
            fire_gremlin(self.graph_db_url, str_gremlin)

            stack_tags = []
            list_list_pv = map(lambda x: x.split(' '), stk['dependencies'])
            for list_pv in list_list_pv:
                if list_pv[0] in topics_data['package_topic_map']:
                    list_tags = topics_data['package_topic_map'][list_pv[0]]
                    str_gremlin = gremlin_str_package_tags('vertx', list_pv[0], list_tags)
                    fire_gremlin(self.graph_db_url, str_gremlin)

                    stack_tags += list_tags  # i.e. stack tags = union of package tags

            stack_name = ref_stack.get('name')
            stack_tags = list(set(stack_tags))
            str_gremlin = gremlin_str_stack_tags(stack_name, stack_tags)
            fire_gremlin(self.graph_db_url, str_gremlin)

    def score(self, list_packages):
        """
        Score the Hyperion model by doing graph traversals. It can make the
        following recommendations for the given user stack:
          a. Companion packages:
            - u1 = union of tags for input list of packages
            - s1 = stacks in the graph that have some of the tags in u1
            - s2 = stacks in s1 that have some of the packages from input
            - m  = the stack in s2 that closely matches with input
            - companion packages = the ones in m that are not in input
          b. Outlier packages:
            - the ones in input that are not in m ( above )
          c. Alternate packages:
            - for each input package, find a package from graph whose tags
            closely match with the input package tags

        Note: It's a POC work ! Various error scenarios are not handled.

        :param list_packages:
        :return:
        """
        str_packages = ','.join(map(lambda x: "'" + x + "'", list_packages))

        # collect tags for each package and union of them is input stack tags
        str_gremlin = "g.V().has('vertex_label', 'Package').has('name', within(" + str_packages + ")).values('tags');"
        input_stack_tags = fire_gremlin(self.graph_db_url, str_gremlin)
        input_stack_tags = input_stack_tags['result']['data']
        input_stack_tags = list(set(input_stack_tags))
        input_stack_tags = map(lambda x: x.encode('utf-8'), input_stack_tags)

        # find those stacks from graph that have some of the input stack tags
        str_gremlin = \
            """
            def find_stack_with_tags(list_tags) {{
                def pred_stack = g.V().has('vertex_label', 'Stack');
                def list_stacks = []
                while (pred_stack.hasNext()) {{
                    def stack_vertex = pred_stack.next();
                    def cur_tags = [];
                    stack_vertex.values('tags').each {{ tag ->
                        cur_tags.add(tag)
                    }}
                    if (!list_tags.disjoint(cur_tags)) {{
                        list_stacks.add(stack_vertex.values('sid'));
                    }}
                }}
                return list_stacks
            }}
            find_stack_with_tags({list_input_tags})
            """.format(list_input_tags=input_stack_tags)
        stack_ids = fire_gremlin(self.graph_db_url, str_gremlin)
        stack_ids = stack_ids['result']['data']
        str_stack_ids = ','.join(map(lambda x: "'" + x[0] + "'", stack_ids))

        # find those stacks from graph that have some of the packages from input
        str_gremlin = "g.V().has('vertex_label', 'Package').has('name', within(" + str_packages + ")).in().has('sid', within(" + str_stack_ids + ")).dedup().valueMap(true);"
        refstacks = fire_gremlin(self.graph_db_url, str_gremlin)

        # find a stack that closely matches with input stack
        set_pkg = set(list_packages)
        list_stacks = refstacks['result']['data']
        list_set_pkg = map(
            lambda x: {'usage': x['usage'][0], 'intersection': set_pkg.intersection(set(x['sname'][0].split('@@'))),
                       'stack_pkg': x['sname'][0].split('@@')}, list_stacks)

        max_intersection = 0
        for i in list_set_pkg:
            if len(i['intersection']) > max_intersection:
                max_intersection = len(i['intersection'])

        list_max_pkg = filter(lambda x: len(x['intersection']) >= max_intersection, list_set_pkg)
        max_stack_size = 0
        max_stack = None  # i.e. the stack that has maximal matching with input stack
        for i in list_max_pkg:
            if len(i['stack_pkg']) > max_stack_size:
                max_stack_size = len(i['stack_pkg'])
                max_stack = i

        companion_packages = list(set(max_stack['stack_pkg']) - set_pkg)
        outlier_packages = list(set_pkg - set(max_stack['stack_pkg']))

        # for each input package, find packages from graph whose tags match closely with input package tags
        alternate_packages = {}
        for pkg in list_packages:
            str_gremlin = \
                """
                def find_alternate_packages(pkg_name, threshold) {{
                    def list_tags = [];
                    g.V().has('vertex_label', 'Package').has('name', pkg_name).values('tags').each {{ tag ->
                        list_tags.add(tag);
                    }}
    
                    def pred_pkg = g.V().has('vertex_label', 'Package').has('name', neq(pkg_name));
                    def list_packages = [];
                    while (pred_pkg.hasNext()) {{
                        def pkg_vertex = pred_pkg.next();
                        def cur_tags = []
                        pkg_vertex.values('tags').each {{ tag ->
                            cur_tags.add(tag);
                        }}
                        similarity = list_tags.intersect(cur_tags).unique().size()/list_tags.plus(cur_tags).unique().size();
                        if (similarity > threshold) {{
                            list_packages.add(pkg_vertex.values('name'));
                        }}
                    }}
                    return list_packages
                }}
                find_alternate_packages('{pkg_name}', {threshold})
                """.format(pkg_name=pkg, threshold=0.3)
            alternates = fire_gremlin(self.graph_db_url, str_gremlin)
            alternates = alternates['result']['data']
            alternates = map(lambda x: x[0], alternates)
            alternate_packages[pkg] = alternates

        return {
            'companion_packages': companion_packages,
            'outlier_packages': outlier_packages,
            'alternate_packages': alternate_packages
        }
