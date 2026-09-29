#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""PipelineItems for interacting with
`NeXus <https://www.nexusformat.org>`__ file objects.
"""

# Local modules
from CHAP.processor import Processor


def nxs_match(nxobject_a, nxobject_b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Return true if the two nxobjects "match" -- they must be
    exactly identical, with one exception: any attributes or fields
    related to code version or file datetime may differ and still be
    considered a "match". Values in corresponding data arrays between
    nxobjects are compared with
    [`np.allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).

    :param nxobject_a, nxobject_b: Input NeXus objects to compare.
    :type nxobject_a, nxobject_b: nexus.nexusformat.NXobject
    :param rtol: Relative tolerance parameter used with `np.allclose`.
    :type rtol: array_like
    :param atol: Absolute tolerance parameter used with `np.allclose`.
    :type atol: array_like
    :param equal_nan: Whether to compare NaN’s as equal. Used with
        `np.allclose`.
    :type equal_nan: bool
    :rtype: bool
    """
    # Third party modules
    import numpy as np
    from nexusformat.nexus import NXfield, NXgroup

    _SKIP_ATTRS = frozenset({
        'date', 'datetime', 'timestamp',
        'file_time', 'file_update_time',
        'version', 'program_version', 'CHAP_version',
    })
    _SKIP_FIELDS = frozenset({
        'start_time', 'end_time', 'duration', 'timestamp',
        'program_name', 'program_version', 'configuration',
    })

    def _filtered_attrs(obj):
        return {k: v for k, v in obj.attrs.items()
                if k not in _SKIP_ATTRS}

    def _match(a, b):
        if type(a) is not type(b):
            return False
        a_attrs = _filtered_attrs(a)
        b_attrs = _filtered_attrs(b)
        if set(a_attrs) != set(b_attrs):
            return False
        for k, va in a_attrs.items():
            vb = b_attrs[k]
            if isinstance(va, np.ndarray):
                if not np.allclose(va, vb, rtol=rtol, atol=atol, equal_nan=equal_nan):
                    return False
            elif va != vb:
                return False
        if isinstance(a, NXfield):
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            return np.allclose(
                a.nxdata, b.nxdata, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        if isinstance(a, NXgroup):
            a_keys = {k for k in a.keys() if k not in _SKIP_FIELDS}
            b_keys = {k for k in b.keys() if k not in _SKIP_FIELDS}
            if a_keys != b_keys:
                return False
            for k in a_keys:
                if not _match(a[k], b[k]):
                    return False
        return True

    return _match(nxobject_a, nxobject_b)


class NexusMakeLinkProcessor(Processor):
    """Processor to run
    `makelink <https://nexpy.github.io/nexpy/treeapi.html#nexusformat.nexus.tree.NXgroup.makelink>`__
    within a given NeXus style
    `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
    object.
    """

    def process(self, data, link_from, link_to,
                nxname=None, abspath=False):
        """Create links between Nexus objects within the given
        PipelineData.

        This method takes a NeXus style
        `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
        object and creates links from the objects specified in
        `link_from` to those in `link_to`. If the underlying file is
        read-only, a copy is made before modifying. Returns the
        modified
        `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
        object containing both the targets and their linked
        counterparts.

        :param data: Input data.
        :type data: list[PipelineData]
        :param link_from: Path(s) within the NXroot whose objects
            should be linked. Can be a single path (str) or a list of
            paths.
        :type link_from: str | list[str]
        :param link_to: Path(s) within the NXroot that serve as link
            targets.  Can be a single path (str) or a list of paths.
        :type link_to: str | list[str]
        :param nxname: Name to assign to the created link. If `None`
            (default), the default naming rules from `makelink` are
            applied.
        :type nxname: str | None, optional
        :param abspath: Whether to create an absolute link path
            (`True`) or a relative one (`False`), defaults to `False`.
        :type abspath: bool, optional
        :returns: The modified
            `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
            object containing the new links.
        :rtype: nexusformat.nexus.NXroot
        """
        # Local modules
        from CHAP.utils.general import nxcopy

        root = self.get_data(data)
        self.logger.debug(f'root.nxfile.mode = {root.nxfile.mode}')
        if root.nxfile.mode == 'r':
            # root belongs to a readonly file, copy to proceed.
            root = nxcopy(root)

        if isinstance(link_from, str):
            link_from = [link_from]
        if isinstance(link_to, str):
            link_to = [link_to]

        for _from in link_from:
            for _to in link_to:
                origin = root[_from]
                target = root[_to]
                self.logger.debug(f'linking to {_to} from {_from}')
                origin.makelink(target, name=nxname, abspath=abspath)

        return root
