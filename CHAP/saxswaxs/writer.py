#!/usr/bin/env python
"""SAXSWAXS command line writer."""

# Local modules
from CHAP import Writer


class ZarrSetupWriter(Writer):
    """Writer for creating an intital zarr file with empty datasets
    for filling in by `saxswaxs.PyfaiIntegrationProcessor` and
    `saxswaxs.ZarrResultsWriter`.
    """
    def process(self, data, filename, dataset_shape, dataset_chunks):
        """Create, write, and return a `zarr.group` to hold processed
        SAXS/WAXS data processed by
        `saxswaxs.PyfaiIntegrationProcessor`.

        :param data:
        `'saxswaxs.models.PyfaiIntegrationProcessorConfig`
        configuration which will be used to process the data later on.
        :type data: list[PipelineData]
        :param filename: Location of zarr file to be written.
        :type filename: str
        :param dataset_shape: Shape of the completed dataset that will
            be processed later on (shape of the measurement itself,
            _not_ including the dimensions of any signals collected at
            each point in that measurement).
        :type dataset_shape: list[int]
        :param dataset_chunks: Extent of chunks along each dimension
            of the completed dataset / measurement. Choose this
            according to how you will process your data -- for
            example, if your `dataset_shape` is `[m, n]`, and you are
            planning to process each of the `m` rows as chunks,
            `dataset_chunks` should be `[1, n]`. But if you plan to
            process each of the `n` columns as chunks,
            `dataset_chunks` should be `[m, 1]`.
        :type dataset_chunks: list[int]
        :return: Empty structure for filling in SAXS/WAXS data
        :rtype: zarr.group
        """
        # Get PyfaiIntegrationProcessorConfig
        try:
            config = self.get_config(
                data=data,
                schema='saxswaxs.models.PyfaiIntegrationProcessorConfig')
        except:
            self.logger.info(
                'No valid PyfaiIntegrationProcessorConfig in input '
                'pipeline data, using config parameter instead')
            try:
                from CHAP.saxswaxs.models import (
                    PyfaiIntegrationProcessorConfig)
                config = PyfaiIntegrationProcessorConfig(**config)
            except Exception as exc:
                raise RuntimeError from exc
        # Get zarr tree as dict from the
        # PyfaiIntegrationProcessorConfig
        tree = config.zarr_tree(dataset_shape, dataset_chunks)
        # Write & return the root zarr.group
        return self.zarr_setup_writer(tree, filename)

    def zarr_setup_writer(self, tree, filename):
        """Create, write, and return a `zarr.group` based on a
        dictionary representing a zarr tree of groups and arrays.

        :param tree: Nested dictionary representing a zarr tree of
            groups and arrays.
        :type tree: dict[str, object]
        :param filename: Location of zarr file to be written.
        :type filename: str
        :return: Zarr group corresponding to the contents of `tree`.
        :rtype: zarr.group
        """
        # Third party modules
        import zarr

        def create_group_or_dataset(node, zarr_parent, indent=0):
            # Set attributes if present
            if 'attributes' in node:
                for key, value in node['attributes'].items():
                    zarr_parent.attrs[key] = value
            # Create children (groups or datasets)
            if 'children' in node:
                for name, child in node['children'].items():
                    if 'shape' in child or 'data' in child:
                        # It's a dataset
                        self.logger.debug(f'Adding dset: {name}')
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
        #root = zarr.open(filename, mode='w')
        from zarr.storage import MemoryStore
        root = zarr.create_group(store=MemoryStore({}))
        create_group_or_dataset(tree['root'], root)
        return root


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
