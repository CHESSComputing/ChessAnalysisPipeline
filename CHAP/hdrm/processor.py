#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Rolf Verberg
Description: Module for Processors used only by HDRM experiments
"""
# System modules
import os

# Third party modules
import numpy as np

# Local modules
from CHAP.processor import Processor


class HdrmPeakfinderProcessor(Processor):
    """A processor for finding peaks in HDRM images."""
    def process(self, data, config=None):
        """Process the stacked input images and return the peaks above
        a given height cirteria.

        :param data: Results of `common.MapProcessor` or other suitable
            preprocessor of the raw detector data containing the map of
            input images.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            hdrm.models.HdrmPeakfinderConfig.
        :type config: dict
        :return: Peaks.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXentry,
            NXlink,
            NXprocess,
            NXroot,
        )

        # Local modules
        from CHAP.edd.processor import get_axes

        # Load the detector data
        try:
            nxobject = self.get_data(data)
            if isinstance(nxobject, NXroot):
                nxroot = nxobject
            elif isinstance(nxobject, NXentry):
                nxroot = NXroot()
                nxroot[nxobject.nxname] = nxobject
                nxobject.set_default()
            else:
                raise ValueError(
                    f'Invalid nxobject in data pipeline ({type(nxobject)}')
        except Exception as exc:
            raise RuntimeError(
                'No valid detector data in input pipeline data') from exc

        # Load the validated HDRM image stacking configuration
        try:
            config = self.get_config(data, 'hdrm.models.HdrmPeakfinderConfig')
        except Exception:
            self.logger.info('No valid conversion config in input pipeline '
                             'data, using config parameter instead.')
            if config is None:
                config = {}
            try:
                # Local modules
                from CHAP.hdrm.models import HdrmPeakfinderConfig

                config = HdrmPeakfinderConfig(**config)
            except Exception as exc:
                raise RuntimeError from exc

        # Add the NXprocess object to the NXroot
        nxprocess = NXprocess()
        try:
            nxroot[f'{nxroot.default}_peaks'] = nxprocess
        except Exception:
            # Local imports
            from CHAP.utils.general import nxcopy

            # Copy nxroot if nxroot is read as read-only
            nxroot = nxcopy(nxroot)
            nxroot[f'{nxroot.default}_peaks'] = nxprocess
        nxprocess.peakfinder_config = config.model_dump_json()

        # Load the detector images
        nxentry = nxroot[nxroot.default]
        nxdata = nxentry[nxentry.default]
        axes = get_axes(nxdata)
        intensities = {k:v for k, v in nxdata.items() if k not in axes}

        # Create the NXdata object to store the peaks data
        nxprocess.data = NXdata()
        if 'axes' in nxdata.attrs:
            nxprocess.data.attrs['axes'] = nxdata.attrs['axes']
        elif 'unstructured_axes' in nxdata.attrs:
            nxprocess.data.attrs['unstructured_axes'] = \
                nxdata.attrs['unstructured_axes']
        for k in nxdata:
            nxprocess.data[k] = NXlink(os.path.join(nxdata.nxpath, k))

        # Find peaks and add the data to the NXprocess object
        detector_peaks = {}
        for detector_id, intensity in intensities.items():
            peak_max = intensity.max()
            peaks = np.asarray(
                np.where(intensity > config.peak_cutoff * peak_max)).T
            nxprocess.data[f'{detector_id}_peaks'] = peaks

        return nxroot


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
