"""PipelineItems for interacting with NeXus file objects"""

from CHAP import Processor


class NexusMakeLinkProcessor(Processor):
    """Processor to run
    [`nexusformat.nexus.tree.NXgroup.makelink`](https://nexpy.github.io/nexpy/treeapi.html#nexusformat.nexus.tree.NXgroup.makelink)
    within a given `NXroot`"""
    def process(self, data, link_from, link_to,
                name=None, abspath=False):
        """Create links between Nexus objects within the given
        PipelineData.

        This method takes an `NXroot` and creates links from the
        objects specified in `link_from` to those in `link_to`. If the
        underlying file is read-only, a copy is made before
        modifying. Returns the modified `NXroot` object containing
        both the targets and their linked counterparts.

        :param data: Data from previous PipelineItems.
        :type data: list[PipelineData]

        :param link_from: Path(s) within the NXroot whose objects
            should be linked. Can be a single path (str) or a list of
            paths.
        :type link_from: str | list[str]

        :param link_to: Path(s) within the NXroot that serve as link
            targets.  Can be a single path (str) or a list of paths.
        :type link_to: str | list[str]

        :param name: Optional name to assign to the created link. If
            None, the default naming rules from `makelink` are
            applied.
        :type name: str | None

        :param abspath: Whether to create an absolute link path (True)
            or a relative one (False). Defaults to False.
        :type abspath: bool

        :returns: The modified NXroot object containing the new links.
        :rtype: NXroot
        """
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
                origin.makelink(target, name=name, abspath=abspath)

        return root
