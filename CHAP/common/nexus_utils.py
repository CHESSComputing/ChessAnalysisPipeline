#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""PipelineItems for interacting with
`NeXus <https://www.nexusformat.org>`__ file objects.
"""

# Local modules
from CHAP.processor import Processor


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
