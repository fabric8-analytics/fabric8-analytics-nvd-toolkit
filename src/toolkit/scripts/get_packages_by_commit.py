#!/bin/env python3

"""Module containing handlers for command line utilities and scripts."""

import argparse
import json
import os
import re
import sys

from toolkit.preprocessing.handlers import GitHandler
from toolkit.preprocessing.ecos import Maven
from toolkit.utils import BooleanAction


# NOTE: only maven is supported for now
ECO_NAMESPACE = {
    'maven': Maven
}


def parse_args(argv):
    """Parse arguments passed to the script."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-repo', '--repository',
        required=True,
        help="Path to local git repository or url to remote git repository."
    )
    parser.add_argument(
        '-c', '--commits',
        nargs='+',
        type=str,
        required=True,
        help="List of commit hashes to search the modified files by."
    )
    parser.add_argument(
        '-n', '--package-limit',
        default=1,
        type=int,
        help="Limit number of packages returned per modified file.\n"
             "If all packages should be returned, "
             "pass 0 or None (default 1)."
    )
    parser.add_argument(
        '-eco', '--ecosystem',
        default='maven',
        help="One of {maven}, the ecosystem the repository belongs to."
    )

    excl_grp = parser.add_mutually_exclusive_group()
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

    return parser.parse_args(argv)


def main(argv):
    """Run."""
    args = parse_args(argv)

    packages = get_packages_by_commits(
        repository=args.repository,
        commits=args.commits,
        package_limit=args.package_limit,
        ecosystem=args.ecosystem
    )

    if args.json:
        print(json.dumps(
            [p.get_attributes(skip_none=False) for p in packages],
            indent=4,
            sort_keys=True
        ))

        return

    if args.format_str is not None:
        format_str = re.sub(r"{(\w+)}", r"{self.\1}", args.format_str)
        print("\n".join(format_str.format(self=p) for p in packages))
    else:
        print("\n".join("{!s}".format(p) for p in packages))

    return


def get_packages_by_commits(
        repository: str,
        commits: list,
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

    :param commits: list, commit hashes to search the modified files by
    :param package_limit: int or None, limit number of packages

        The limit is applied per modified file.
        If all packages found in the path should be listed,
        provide None or 0

    :param ecosystem: ecosystem the repository belongs to

        {maven, npm, python}, by default 'maven' is assumed
    """
    if repository.startswith('http'):
        print('\nCloning repository...\n', file=sys.stderr)
        handler = GitHandler.clone(url=repository)
    else:
        handler = GitHandler(path=repository)

    with handler as git:
        mod_files = git.get_modified_files(commits=commits)

    eco_namespace = _get_namespace_by_eco(ecosystem)

    packages = set()
    for commit, files in mod_files.items():

        stdout, _ = handler.exec_cmd(
            cmd='git checkout %s' % commit,
            chdir=handler.repository
        )

        for mod_file_path in sorted(files, key=len, reverse=True):
            root_dir = os.path.dirname(str(mod_file_path))
            found_packages = eco_namespace.find_packages(root_dir, topdown=False)

            for p in found_packages[:[None, package_limit][package_limit]]:
                packages.add(p)

    stdout, _ = handler.exec_cmd(
        cmd='git checkout master',
        chdir=handler.repository
    )

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
    main(sys.argv[1:])
