#!/usr/bin/env python
"""SAXSWAXS command line writer."""
from CHAP import Writer

class ZarrSetupWriter(Writer):
    def process(self, data, filename, dataset_shape, dataset_chunks):
        # Get config for PyfaiIntegrationProcessor from data
        # Using config & experiment_shape, setup tree dict to pass to
        # common.ZarrSetupWriter
        # call common.ZarrSetupWriter
        import json
        try:
            config = self.get_config(
                data, f'saxswaxs.models.PyfaiIntegrationProcessorConfig')
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
        tree = config.zarr_tree(dataset_shape, dataset_chunks)
        self.logger.debug(f'zarr tree:\n{json.dumps(tree, indent=2)}')
        return self.zarr_setup_writer(tree, filename)

    def zarr_setup_writer(self, tree, filename):
        import zarr
        import numpy as np

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
    def write(self, data, filename):
        # Get list of PyfaiIntegrationProcessor results to write
        data = self.unwrap_pipelinedata(data)[0]
        for d in data:
            self.zarr_writer(filename=filename, **d)
        return data
    def zarr_writer(self, filename, path, idx, data):
        """Write data to a specific Zarr dataset.

        This method writes `data` to a specified dataset within a Zarr
        file at the given index (`idx`). The Zarr file is opened in
        append mode to allow modifications.  If the dataset does not
        exist, an error is raised. The method ensures that the shape
        of `data` matches the shape of the target slice before
        writing.

        :param filename: Path to the Zarr file.
        :type filename: str
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
        import zarr
        self.logger.info(f'Writing to {path} at {idx}')

        # Open the Zarr file in append mode to allow modifications
        root = zarr.open(filename, mode='a')

        # Check if the dataset exists
        if path not in root:
            raise ValueError(
                f'Dataset "{path}" does not exist in the Zarr file.')

        # Access the specified dataset
        dataset = root[path]

        # Check that the slice shape matches the data shape
        if dataset[idx].shape != data.shape:
            raise ValueError(
                f'Data shape {data.shape} does not match the target slice '
                + f'shape {dataset[idx].shape}.')

        # Write the data to the specified slice
        dataset[idx] = data
        self.logger.info(
            f'Data written to "{path}" at slice {idx}.')
        return data



if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
