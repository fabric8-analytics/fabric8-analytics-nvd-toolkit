#!/bin/env python3

"""Module containing handlers for command line utilities and scripts."""

import argparse
import json
import os
import re

from toolkit.preprocessing.handlers import GitHandler
from toolkit.preprocessing.ecos import Maven
from toolkit.utils import BooleanAction


# NOTE: only maven is supported for now
ECO_NAMESPACE = {
    'maven': Maven
}


__parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
__parser.add_argument(
    '-repo', '--repository',
    required=True,
    help="Path to local git repository or url to remote git repository."
)
__parser.add_argument(
    '-c', '--commit',
    required=True,
    help="Commit hash to search the modified files by."
)
__parser.add_argument(
    '-n', '--package-limit',
    default=1,
    type=int,
    help="Limit number of packages returned per modified file.\n"
         "If all packages should be returned, "
         "pass 0 or None (default 1)."
)
__parser.add_argument(
    '-eco', '--ecosystem',
    default='maven',
    help="One of {maven}, the ecosystem the repository belongs to."
)

excl_grp = __parser.add_mutually_exclusive_group()
excl_grp.add_argument(
    '--json', '--nojson',
    action=BooleanAction,
    default=True,
    help="Dump the result as JSON (NOTE: cannot be used with --format argument)."
)
excl_grp.add_argument(
    '--format',
    dest='format_str',
    help="Python format string to be formatted with package attributes.\n"
         "Possible package attributes are: ecosystem, name, owner, version, "
         "description, url[, aid, gid].\n"
         "Example: Name: {name}, owner: {owner}, gid: {gid}, aid: {aid}}"
)


def main():
    """Main function."""
    args = __parser.parse_args()

    packages = get_packages_by_commit(
        repository=args.repository,
        commit=args.commit,
        package_limit=args.package_limit,
        ecosystem=args.ecosystem
    )
    if args.json:
        print(json.dumps(
            [p.get_attributes(skip_none=True) for p in packages],
            indent=4,
            sort_keys=True
        ))

        exit(0)

    if args.format_str is not None:
        format_str = re.sub(r"{(\w+)}", r"{self.\1}", args.format_str)
        print("\n".join(format_str.format(self=p) for p in packages))
    else:
        print("\n".join("{!s}".format(p) for p in packages))

    exit(0)


def get_packages_by_commit(
        repository: str,
        commit: str,
        package_limit=1,
        ecosystem='maven') -> list:
    """Get package name from git repository and commit hash.

    A git handler is created and modified files are searched
    by the given commit. Package is inferred based on those
    modified files.
    There can be multiple packages, by default only one child
    package is returned.

    *NOTE:* Only usable for git repositories.

    :param repository: str, path to local repository or url

        If url is provided, to repository will be cloned into
        a temporary folder (at /tmp)

    :param commit: commit hash to search the modified files by
    :param package_limit: int or None, limit number of packages

        The limit is applied per modified file.
        If all packages found in the path should be listed, provide None or 0

    :param ecosystem: ecosystem the repository belongs to

        {maven, npm, python}, by default 'maven' is assumed
    """
    if repository.startswith('http'):
        handler = GitHandler.clone(url=repository)
    else:
        handler = GitHandler(path=repository)

    with handler as git:
        mod_files = git.get_modified_files(commit=commit)

    mod_files = sorted(mod_files, key=len, reverse=True)
    eco_namespace = _get_namespace_by_eco(ecosystem)

    packages = set()
    for mod_file_path in mod_files:
        root_dir = os.path.dirname(str(mod_file_path))
        found_packages = eco_namespace.find_packages(root_dir, topdown=False)

        for p in found_packages[:[None, package_limit][package_limit]]:
            packages.add(p)

    # the first found package should be the child package belonging to the file
    # which has been modified
    return list(packages)


def _get_namespace_by_eco(ecosystem: str):
    """Return correct handler for given ecosystem."""

    if not ecosystem.lower() in ECO_NAMESPACE:
        raise ValueError("Ecosystem `{}` is not supported.\n"
                         .format(ecosystem),
                         "Supported ecosystems are: {}"
                         .format(ECO_NAMESPACE.keys()))

    return ECO_NAMESPACE.get(ecosystem.lower())


if __name__ == '__main__':
    main()
