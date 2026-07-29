# Local modules
from CHAP.runner import set_logger

logger, _ = set_logger(log_level='DEBUG')

def dict_to_zarr(tree, logger=logger):
    """Create a
    `Zarr group <https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group>`__
    object based on a dictionary representing a Zarr tree of groups
    and arrays.

    :param tree: Nested dictionary representing a Zarr tree of
        groups and arrays.
    :type tree: dict[str, Any]
    :return: Zarr group corresponding to the contents of `tree`.
    :rtype: zarr.Group
    """
    # Third party modules
    # pylint: disable=import-error
    import zarr
    from zarr.storage import MemoryStore

    def create_group_or_dataset(node, zarr_parent, indent=0):
        """Create and return a
        `Zarr group <https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group>`__
        `Zarr dataset <https://zarr.readthedocs.io/en/stable/api/zarr/array/#zarr.Array>`__.

        :param node: Child Zarr tree group.
        :type node: zarr.Group or zarr.Array
        :param zarr_parent: Parent Zarr tree group.
        :type zarr_parent: zarr.Group
        :param indent: Indentation level, defaults to 0.
        :type indent: int, optional
        """
        # Set attributes if present
        if 'attributes' in node:
            for key, value in node['attributes'].items():
                zarr_parent.attrs[key] = value
        # Create children (groups or datasets)
        if 'children' in node:
            for name, child in node['children'].items():
                if 'shape' in child or 'data' in child:
                    # It's a dataset
                    logger.debug(f'Adding dset: {name}')
                    zarr_parent.create_dataset(
                        name,
                        **child,
                    )
                    # Set dataset attributes
                    if 'attributes' in child:
                        for key, value in child['attributes'].items():
                            zarr_parent[name].attrs[key] = value
                else:
                    # It's a group
                    group = zarr_parent.create_group(name)
                    create_group_or_dataset(child, group, indent=indent+2)
    results = zarr.create_group(store=MemoryStore({}))
    create_group_or_dataset(tree, results)
    return results
