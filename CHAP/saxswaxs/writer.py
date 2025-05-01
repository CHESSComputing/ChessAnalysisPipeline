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
                schema=f'saxswaxs.models.PyfaiIntegrationProcessorConfig')
        except:
            self.logger.info(
                f'No valid PyfaiIntegrationProcessorConfig in input '
                'pipeline data, using config parameter instead')
            try:
                from CHAP.saxswaxs.models import (
                    PyfaiIntegrationProcessorConfig)
                config = PyfaiIntegrationProcessorConfig(
                    **config, inputdir=inputdir)
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
        root = zarr.open(filename, mode='w')
        create_group_or_dataset(tree['root'], root)
        return root


class ZarrResultsWriter(Writer):
    """Writer for placing results of
    `saxswaxs.PyfaiIntegrationProcessor` into a zarr file (usually one
    created by `saxswaxs.ZarrSetupWriter`)."""
    def write(self, data, filename):
        """Write `data` (from `saxswaxs.PyfaiIntegrationProcessor`) to
        the Zarr file `filename`.

        :param data: Results from `saxswaxs.PyfaiIntegrationProcessor`.
        :type data: list[PipelineData]
        :param filename: Name of Zarr file to which to write.
        :type filename: str
        :returns: Original results from
            `saxswaxs.PyfaiIntegrationProcessor`.
        :rtype: list[dict[str, object]]
        """
        # Third party modules
        import zarr

        # Open file in append mode to allow modifications
        zarrfile = zarr.open(filename, mode='a')

        # Get list of PyfaiIntegrationProcessor results to write
        data = self.unwrap_pipelinedata(data)[0]
        for d in data:
            self.zarr_writer(zarrfile=zarrfile, **d)

        return data

    def zarr_writer(self, zarrfile, path, idx, data):
        """Write data to a specific Zarr dataset.

        This method writes `data` to a specified dataset within a Zarr
        file at the given index (`idx`). If the dataset does not
        exist, an error is raised. The method ensures that the shape
        of `data` matches the shape of the target slice before
        writing.

        :param zarrfile: Path to the Zarr file.
        :type zarrfile: zarr.core.group.Group
        :param path: Path to the dataset inside the Zarr file.
        :type path: str
        :param idx: Index or slice where the data should be written.
        :type idx: tuple or int
        :param data: Data to be written to the specified slice in the
            dataset.
        :type data: numpy.ndarray or compatible array-like object
        :return: The written data.
        :rtype: numpy.ndarray or compatible array-like object
        :raises ValueError: If the specified dataset does not exist or
            if the shape of `data` does not match the target slice.
        """
        self.logger.info(f'Writing to {path} at {idx}')

        # Check if the dataset exists
        if path not in zarrfile:
            raise ValueError(
                f'Dataset "{path}" does not exist in the Zarr file.')

        # Access the specified dataset
        dataset = zarrfile[path]

        # Check that the slice shape matches the data shape
        if dataset[idx].shape != data.shape:
            raise ValueError(
                f'Data shape {data.shape} does not match the target slice '
                f'shape {dataset[idx].shape}.')

        # Write the data to the specified slice
        dataset[idx] = data
        self.logger.info(f'Data written to "{path}" at slice {idx}.')


class NexusResultsWriter(Writer):
    """Writer for placing results of
    `saxswaxs.PyfaiIntegrationProcessor` into a NeXus file.
    """
    def write(self, data, filename):
        """Write `data` (from `saxswaxs.PyfaiIntegrationProcessor`) to
        the NeXus file `filename`.

        :param data: Results from `saxswaxs.PyfaiIntegrationProcessor`.
        :type data: list[PipelineData]
        :param filename: Name of NeXus file to which to write.
        :type filename: str
        :returns: Original results from
            `saxswaxs.PyfaiIntegrationProcessor`.
        :rtype: list[dict[str, object]]
        """
        from nexusformat.nexus import NXFile

        # Get list of PyfaiIntegrationProcessor results to write
        data = self.unwrap_pipelinedata(data)[0]
        for d in data:
            with NXFile(filename, 'a') as nxroot:
                self.nxs_writer(nxroot=nxroot, **d)

        return data

    def nxs_writer(self, nxroot, path, idx, data):
        """Write data to a specific NeXus file.

        This method writes `data` to a specified dataset within a NeXus
        file at the given index (`idx`). If the dataset does not
        exist, an error is raised. The method ensures that the shape
        of `data` matches the shape of the target slice before
        writing.

        :param nxroot: NeXus root object.
        :type nxroot: nexusformat.nexus.NXroot
        :param path: Path to the dataset inside the NeXus file.
        :type path: str
        :param idx: Index or slice where the data should be written.
        :type idx: tuple or int
        :param data: Data to be written to the specified slice in the
            dataset.
        :type data: numpy.ndarray or compatible array-like object
        :return: The written data.
        :rtype: numpy.ndarray or compatible array-like object
        :raises ValueError: If the specified dataset does not exist or
            if the shape of `data` does not match the target slice.
        """
        self.logger.info(f'Writing to {path} at {idx}')

        # Check if the dataset exists
        if path not in nxroot:
            raise ValueError(
                f'Dataset "{path}" does not exist in the NeXus file.')

        # Access the specified dataset
        dataset = nxroot[path]

        # Check that the slice shape matches the data shape
        if dataset[idx].shape != data.shape:
            raise ValueError(
                f'Data shape {data.shape} does not match the target slice '
                f'shape {dataset[idx].shape}.')

        # Write the data to the specified slice
        dataset[idx] = data
        self.logger.info(f'Data written to "{path}" at slice {idx}.')


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
