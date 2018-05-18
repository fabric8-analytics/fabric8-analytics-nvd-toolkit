"""Feature hooks for extractors."""

import operator
import typing

from itertools import chain

from toolkit.transformers import Hook


# Hook Functions
# --------------

def __has_uppercase(features: list, pos: int, **kwargs) -> bool:
    """Return whether the word in `tagged` contains uppercase letter.

    :param features: list of tuples (word, tag)
    :param pos: int, position of the word in `tagged`
    """
    word, _ = features[pos]  # type: str

    return any([w.isupper() for w in word])


def __is_alnum(features: list, pos: int, **kwargs) -> bool:
    """Return whether the word contains only alphanumeric characters.

    :param features: list of tuples (word, tag)
    :param pos: int, position of the word in `tagged`
    """
    word, _ = features[pos]  # type: str

    return word.isalnum()


def __vendor_product_match(features: list,
                           pos: int,
                           cve_dict: dict,
                           cve_id: str,
                           **kwargs) -> bool:
    """Return whether the given word matches vendor or product of given CVE.

    :param features: list of tuples (word, tag)
    :param pos: int, position of the word in `tagged`
    :param cve_dict: dict of signature (CVE_ID: str, cve: nvdlib.model.CVE)
    :param cve_id: str, CVE_ID as stated in NVD feed
    """
    word, _ = features[pos]  # type: str
    word = word.lower()

    match = False

    cve = cve_dict.get(cve_id, None)

    if cve is not None:
        nodes: list = cve.configurations

        if not nodes:
            match = False

        else:
            cpes = list(chain.from_iterable([node.cpe for node in nodes]))

            for cpe in cpes:
                if not cpe.is_application():
                    continue
                try:
                    from nvdlib.model import CVE
                    vendor, = cpe.vendor  # type: str
                    product, = cpe.product  # type: str

                except ValueError:
                    break

                if any([s.lower().find(word) != -1 for s in [vendor, product]]):
                    match = True
                    break

    return match


def __ver_follows(features: list, pos: int, **kwargs) -> bool:
    """Return whether the given word is followed by a version string."""
    version_tag = '<VERSION>'
    ver_pos = [
        p for p, (_, t) in enumerate(features)
        if t == version_tag
    ]

    if not ver_pos:
        return False

    return any([p > pos for p in ver_pos])


def __ver_pos(features: list, pos: int, **kwargs) -> typing.Union[int, None]:
    """Return version position in the `tagged` w.r.t the given word.

    :param features: list of tuples (word, tag)
    :param pos: int, position of the word in `tagged`
    """
    version_tag = '<VERSION>'
    ver_pos = [
        p - pos for p, (_, t) in enumerate(features)
        if t == version_tag
    ]

    # noinspection PyTypeChecker
    return min(ver_pos, key=lambda x: abs(x), default=None)


def __word_len(features: list, pos: int, cmp=None, limit=3, **kwargs) -> bool:
    """Compare length of word in `tagged` to the `limit`.

    :param features: list of tuples (word, tag)
    :param pos: int, position of the word in `tagged`
    :param cmp: func, comparator function, default `operator.gt`

        The cmp function must be of signature (a: int, b: int) -> bool

    :param limit: int, limit to compare to (the `b` argument in `cmp`)
    """
    word, _ = features[pos]  # type: str

    if cmp is None:
        cmp = operator.gt

    return cmp(len(word), limit)


# Feature Hooks
# -------------

has_uppercase_hook: Hook = Hook(key="has_uppercase", func=__has_uppercase)
"""Hook: Return whether the word in `tagged` contains uppercase letter."""

is_alnum_hook: Hook = Hook(key="is_alnum", func=__is_alnum)
"""Hook: Return whether the word contains only alphanumeric characters."""

ver_follows_hook: Hook = Hook(key="ver_follows", func=__ver_follows)
"""Hook: Return whether the given word is followed by a version string."""

ver_pos_hook: Hook = Hook(key="ver_pos", func=__ver_pos)
"""Hook: Return version position in the `tagged` w.r.t the given word."""

vendor_product_match_hook: Hook = Hook(
    key="vendor_product_match",
    func=__vendor_product_match
)
"""Hook: Return whether the given word matches vendor or product."""

word_len_hook: Hook = Hook(key="word_len", func=__word_len, limit=3)
"""Hook: Compare length of word in `tagged` to the `limit`."""
