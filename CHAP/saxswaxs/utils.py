# Local modules
from CHAP.runner import set_logger

def dict_to_nexus(tree, logger=None):
    """Create a
    `NXgroup <https://nexpy.github.io/nexpy/treeapi.html#nexusformat.nexus.tree.NXgroup>`__
    object based on a dictionary representing a tree of groups and
    arrays.

    :param tree: Nested dictionary representing a tree of groups and
        arrays.
    :type tree: dict[str, Any]
    :return: NeXus style object corresponding to the contents of
        `tree`.
    :rtype: nexusformat.nexus.NXgroup
    """
    def create_nexus_object(node, *args, **kwargs):
        """Create and return a NeXus style object matching the node's
        definition.

        :param node: Tree group or dataset.
        :type node: dict[str, Any]
        :return: NeXus style object matching the `node`'s definition.
        :rtype: nexusformat.nexus.NXgroup
        """
        # Third party modules
        import nexusformat.nexus as nx

        attrs = node.get('attributes', {})
        attrs.pop('__nxlinks__', None)
        nxclass_string = attrs.pop('NX_class', None)
        if nxclass_string is None:
            nxclass_string = 'NXroot' if 'root' in node else 'NXgroup'
        try:
            nxclass = getattr(nx, nxclass_string)
        except AttributeError:
            raise ValueError(f'Invalid NeXus class string ({nxclass_string})')
        return nxclass(attrs=attrs if attrs else None, *args, **kwargs)

    def create_group_or_dataset(node, parent):
        """Create all 'children' objects of a node under its parent.

        :param node: Child tree group or dataset.
        :type node: dict[str, Any]
        :param parent: Parent tree group.
        :type parent: nexusformat.nexus.NXgroup
        """
        # Third party modules
        from nexusformat.nexus import NXfield

        # Set attributes if present
        if 'attributes' in node:
            for key, value in node['attributes'].items():
                parent.attrs[key] = value
        # Create children (groups or datasets)
        for name, child in node.get('children', {}).items():
            if 'shape' in child or 'data' in child:
                # It's a dataset
                if logger is not None:
                    logger.debug(f'Adding dataset {name} to {parent}')
                attrs = child.pop('attributes', {})
                if 'NX_class' in attrs:
                    assert attrs.pop('NX_class') == 'NXfield'
                child.pop('chunks', None) # FIX
                #if 'chunks' in child:
                #    child['chunks'] = tuple(child['chunks'])
                parent[name] = NXfield(name=name, attrs=attrs, **child)
            else:
                # It's a group
                if logger is not None:
                    logger.debug(f'Adding group {name} to {parent}')
                parent[name] = create_nexus_object(child, name=name)
                create_group_or_dataset(child, parent[name])

    results = create_nexus_object(tree)
    create_group_or_dataset(tree, results)
    return results

def dict_to_zarr(tree, logger=None):
    """Create a
    `Zarr group <https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group>`__
    object based on a dictionary representing a tree of groups and
    arrays.

    :param tree: Nested dictionary representing a tree of groups and
        arrays.
    :type tree: dict[str, Any]
    :return: Zarr group corresponding to the contents of `tree`.
    :rtype: zarr.Group
    """
    # Third party modules
    # pylint: disable=import-error
    import zarr
    from zarr.storage import MemoryStore

    def create_group_or_dataset(node, parent, indent=0):
        """Create all 'children' objects of a node under its parent.

        :param node: Child tree group or dataset.
        :type node: dict[str, Any]
        :param parent: Parent tree group.
        :type parent: zarr.Group
        :param indent: Indentation level, defaults to 0.
        :type indent: int, optional
        """
        # Set attributes if present
        if 'attributes' in node:
            for key, value in node['attributes'].items():
                parent.attrs[key] = value
        # Create children (groups or datasets)
        if 'children' in node:
            for name, child in node['children'].items():
                if 'data' in child:
                    # It's a dataset with values specified
                    if logger is not None:
                        logger.debug(f'Adding dset: {name}')
                    #parent.create_array(name, data=child['data'])
                    parent[name] = child['data']
                    # Set dataset attributes
                    if 'attributes' in child:
                        for key, value in child['attributes'].items():
                            parent[name].attrs[key] = value
                elif 'shape' in child or 'dtype' in child:
                    # It's a dataset, but no values specified
                    if logger is not None:
                        logger.debug(f'Adding dset: {name}')
                    parent.create_array(name, **child)
                    # Set dataset attributes
                    if 'attributes' in child:
                        for key, value in child['attributes'].items():
                            parent[name].attrs[key] = value
                else:
                    # It's a group
                    group = parent.create_group(name)
                    create_group_or_dataset(child, group, indent=indent+2)
    results = zarr.create_group(store=MemoryStore({}))
    create_group_or_dataset(tree, results)
    return results
