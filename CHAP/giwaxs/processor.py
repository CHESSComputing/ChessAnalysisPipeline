#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Module for Processors unique to the GIWAXS workflow.

Add discription of GIWAXS
"""

# System modules
from json import loads
from typing import Optional

# Third party modules
import numpy as np
from pydantic import (
    Field,
    PrivateAttr,
    conint,
    constr,
)

# Local modules
from CHAP.giwaxs.models import (
    GiwaxsConversionConfig,
    PyfaiIntegrationConfig,
)
from CHAP.processor import Processor


class GiwaxsConversionProcessor(Processor):
    """A processor for converting GIWAXS images from curved to
    rectangular coordinates (wedge correction).

    :ivar config: Initialization parameters for an instance of
        :class:`~CHAP.giwaxs.models.GiwaxsConversionConfig`.
    :vartype config: dict, optional
    :ivar nxmemory: Maximum memory usage when reading NeXus files.
    :vartype nxmemory: int, optional
    :ivar nxpath: Path to a specific location in the NeXus file tree
        to read the intensity data from.
    :vartype nxpath: str, optional
    :ivar save_figures: Save .pngs of plots for checking inputs &
        outputs of this Processor, defaults to `False`.
    :vartype save_figures: bool, optional
    """

    pipeline_fields: dict = Field(
        default = {
            'config': 'giwaxs.models.GiwaxsConversionConfig'}, init_var=True)
    config: GiwaxsConversionConfig
    nxmemory: Optional[conint(gt=0)] = 100000
    nxpath: Optional[constr(strip_whitespace=True, min_length=1)] = None
    save_figures: Optional[bool] = True

    _animation: list = PrivateAttr(default=[])
    _figures: list = PrivateAttr(default=[])

    def process(self, data):
        """Process the GIWAXS input images & configuration and return
        a map of the images in rectangular coordinates as a NeXus style
        `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
        object.

        :param data: Results of
            :class:`~CHAP.common.processor.MapProcessor` containing
            a map with the GIWAXS input images.
        :type data: list[PipelineData]
        :return: Converted GIWAXS images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        import fabio
        from nexusformat.nexus import (
            NXdata,
            NXfield,
            NXprocess,
            nxsetconfig,
        )
        from pyFAI.gui.utils.units import Unit

        # Local modules
        from CHAP.common.map_utils import get_axes
        from CHAP.utils.general import nxcopy

        nxsetconfig(memory=self.nxmemory)

        # Load the detector data
        nxroot = self.get_nxroot(self.get_data(data))

        # Validate the azimuthal integrator configuration and check
        # against the input data (availability and shape)
        nxentry = nxroot[nxroot.default]
        detector_ids = [
            #str(id, 'utf-8') for id in nxentry.detector_ids.nxdata]
            str(id) for id in nxentry.detector_ids.nxdata]
        if len(detector_ids) > 1:
            raise RuntimeError('More than one detector not yet implemented')
        nxdata = nxentry[nxentry.default]
        data = {}
        independent_dims = {}
        skipped_detectors = []
        ais = []
        for ai in self.config.azimuthal_integrators:
            ai_id = ai.get_id()
            if ai_id in nxdata:
                if nxdata[ai_id].ndim != 3:
                    raise RuntimeError('Inconsistent raw data dimension '
                                       f'{nxdata[ai_id].ndim}')
                ais.append(ai)
                if self.nxpath is None:
                    data[ai_id] = nxdata[ai_id].nxdata
                else:
                    data[ai_id] = nxroot[self.nxpath]
            else:
                skipped_detectors.append(ai_id)
        if skipped_detectors:
            self.logger.warning('Skipping detector(s) '
                                f'{skipped_detectors} (no raw data)')
        if not ais:
            raise RuntimeError('No matching raw detector data found')
        ai_id = ais[0].get_id()
        axes = get_axes(nxdata)
        if not axes:
            self.logger.warning('Unable to find axes information')
        independent_dims[ai_id] = [
            nxcopy(nxdata[a]) for a in axes]
        data[ai_id] = nxdata[ai_id]
        if axes[0] == 'theta':
            thetas = nxdata['theta']
            theta_unit = thetas.attrs.get('units')
            if 'deg' in theta_unit:
                thetas = np.radians(thetas)
        else:
            thetas = None

        # Read the mask(s)
        # FIX read at validation, like the poni file
        masks = {}
        for ai in ais:
            self.logger.debug(f'Reading {ai.mask_file}')
            try:
                with fabio.open(ai.mask_file) as f:
                    mask = f.data
                    self.logger.debug(
                        f'mask shape for {ai.get_id()}: {mask.shape}')
                    masks[ai.get_id()] = mask
            except (IOError, OSError, ValueError):
                self.logger.debug(f'No mask file found for {ai.get_id()}')
        if not masks:
            masks = None

        # Perform integration(s)
        ais_pyfai = {ai.get_id(): ai.ai for ai in ais}
        for integration in self.config.integrations:

            # Add a NXprocess object(s) to the NXroot
            nxprocess = NXprocess()
            try:
                nxroot[f'{nxroot.default}_{integration.name}'] = nxprocess
            except ValueError:
                # Copy nxroot if nxroot is read as read-only
                nxroot = nxcopy(nxroot)
                nxroot[f'{nxroot.default}_{integration.name}'] = nxprocess
            nxprocess.integration_config = integration.model_dump_json()
            nxprocess.azimuthal_integrators = [
                ai.model_dump_json() for ai in ais]

            # Integrate the data
            results = integration.integrate(
                ais_pyfai, data, masks=masks, thetas=thetas)

            # Create the NXdata object with the integrated data
            intensities = results['intensities']
            coords = [i for k, v in independent_dims.items()
                      for i in v if k in ais_pyfai]
            q_outofplane = results['outofplane']['coords']
            if results['outofplane']['unit'] == 'qoop_A^-1':
                unit = Unit.INV_ANGSTROM.symbol
            elif results['outofplane']['unit'] == 'qoop_nm^-1':
                unit = 'nm^-1'
            else:
                unit = results['outofplane']['unit']
            if np.asarray(intensities).ndim == 2:
                intensities = np.expand_dims(intensities, axis=0)
            coords.append(
                NXfield(
                    q_outofplane, 'q_outofplane',
                    attrs={'units': unit}))
            q_inplane = results['inplane']['coords']
            if results['inplane']['unit'] == 'qip_A^-1':
                unit = Unit.INV_ANGSTROM.symbol
            elif results['inplane']['unit'] == 'qip_nm^-1':
                unit = 'nm^-1'
            else:
                unit = results['inplane']['unit']
            coords.append(
                NXfield(
                    q_inplane, 'q_inplane',
                    attrs={'units': unit}))
            nxdata = NXdata(NXfield(intensities, ai_id), tuple(coords))
            if len(axes) > 1:
                nxdata.attrs['unstructured_axes'] = nxdata.attrs['axes'][:-1]
                del nxdata.attrs['axes']
            nxprocess.data = nxdata
            nxprocess.default = 'data'

        self.config.azimuthal_integrators = ais

        return nxroot


class PyfaiIntegrationProcessor(Processor):
    """A processor for azimuthally integrating images.

    :ivar config: Initialization parameters for an instance of
        :class:`~CHAP.giwaxs.models.PyfaiIntegrationConfig`.
    :vartype config: dict, optional
    :ivar nxmemory: Maximum memory usage when reading NeXus files.
    :vartype nxmemory: int, optional
    """

    pipeline_fields: dict = Field(
        default = {
            'config': 'giwaxs.models.PyfaiIntegrationConfig'}, init_var=True)
    config: PyfaiIntegrationConfig
    nxmemory: Optional[conint(gt=0)] = 100000

    def process(self, data):
        """Process the input images & configuration and return a map of
        the azimuthally integrated images as a NeXus style
        `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
        object.

        :param data: Results of
            :class:`~CHAP.common.processor.MapProcessor` or other
            suitable preprocessor of the raw detector data containing
            the map of input images.
        :type data: list[PipelineData]
        :return: Integrated images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        import fabio
        from nexusformat.nexus import (
            NXdata,
            NXfield,
            NXprocess,
            nxsetconfig,
        )
        from pyFAI.gui.utils.units import Unit

        # Local imports
        from CHAP.utils.general import nxcopy

        nxsetconfig(memory=self.nxmemory)

        # Load the detector data
        nxroot = self.get_nxroot(self.get_data(data))

        # Validate the azimuthal integrator configuration and check
        # against the input data (availability and shape)
        data = {}
        independent_dims = {}
        try:
            nxprocess_converted = nxroot[f'{nxroot.default}_converted']
            conversion_config = loads(
                str(nxprocess_converted.conversion_config))
            converted_ais = conversion_config['azimuthal_integrators']
            if len(converted_ais) > 1:
                raise RuntimeError(
                    'More than one detector not yet implemented')
            if self.config.azimuthal_integrators is None:
                # Local modules
                from CHAP.common.models.integration import (
                    AzimuthalIntegratorConfig,
                )

                ais = [AzimuthalIntegratorConfig(**converted_ais[0])]
            else:
                converted_ids = [ai['id'] for ai in converted_ais]
                skipped_detectors = []
                ais = []
                for ai in self.config.azimuthal_integrators:
                    if ai.get_id() in converted_ids:
                        ais.append(ai)
                    else:
                        skipped_detectors.append(ai.get_id())
                if skipped_detectors:
                    self.logger.warning(
                        f'Skipping detector(s) {skipped_detectors} '
                        '(no converted data)')
                if not ais:
                    raise RuntimeError(
                        'No matching azimuthal integrators found')
            nxdata = nxprocess_converted.data
            axes = nxdata.attrs['axes']
            if len(nxdata.attrs['axes']) != 3:
                raise RuntimeError('More than one independent dimension '
                                   'not yet implemented')
            axes = axes[0]
            independent_dims[ais[0].get_id()] = nxcopy(nxdata[axes])
            data[ais[0].get_id()] = np.flip(nxdata.nxsignal.nxdata, axis=1)
        except Exception as exc:
            experiment_type = loads(
                str(nxroot[nxroot.default].map_config))['experiment_type']
            if experiment_type == 'GIWAXS':
                self.logger.warning(
                    'No converted data found, use raw data for integration')
            nxentry = nxroot[nxroot.default]
            detector_ids = [
                #str(id, 'utf-8') for id in nxentry.detector_ids.nxdata]
                str(id) for id in nxentry.detector_ids.nxdata]
            if len(detector_ids) > 1:
                raise RuntimeError(
                    'More than one detector not yet implemented')
            if self.config.azimuthal_integrators is None:
                raise ValueError('Missing azimuthal_integrators parameter in '
                                 f'PyfaiIntegrationProcessor.config '
                                 f'({self.config})')
            nxdata = nxentry[nxentry.default]
            skipped_detectors = []
            ais = []
            for ai in self.config.azimuthal_integrators:
                if ai.get_id() in nxdata:
                    if nxdata[ai.get_id()].ndim != 3:
                        raise RuntimeError('Inconsistent raw data dimension '
                                           f'{nxdata[ai.get_id()].ndim}')
                    ais.append(ai)
                    data[ai.get_id()] = nxdata[ai.get_id()].nxdata
                else:
                    skipped_detectors.append(ai.get_id())
            if skipped_detectors:
                self.logger.warning('Skipping detector(s) '
                                    f'{skipped_detectors} (no raw data)')
            if not ais:
                raise RuntimeError('No matching raw detector data found')
            if 'unstructured_axes' in nxdata.attrs:
                axes = nxdata.attrs['unstructured_axes']
                independent_dims[ais[0].get_id()] = [
                    nxcopy(nxdata[a]) for a in axes]
            elif 'axes' in nxdata.attrs:
                axes = nxdata.attrs['axes']
                independent_dims[ais[0].get_id()] = nxcopy(nxdata[axes])
            else:
                self.logger.warning('Unable to find independent_dimensions')
            data[ais[0].get_id()] = nxdata[ais[0].get_id()]

        # Select the images to integrate
        #if False and self.config.scan_step_indices is not None:
        #    #FIX
        #    independent_dims = independent_dims[self.config.scan_step_indices]
        #    data = data[self.config.scan_step_indices]
        self.logger.debug(
            f'data shape(s): {[(k, v.shape) for k, v in data.items()]}')
        if self.config.sum_axes:
            data = {k:np.sum(v.nxdata, axis=0)[None,:,:]
                    for k, v in data.items()}
            self.logger.debug('data shape(s) after summing: '
                              f'{[(k, v.shape) for k, v in data.items()]}')

        # Read the mask(s)
        # FIX read at validation, like the poni file
        masks = {}
        for ai in ais:
            self.logger.debug(f'Reading {ai.mask_file}')
            try:
                with fabio.open(ai.mask_file) as f:
                    mask = f.data
                    self.logger.debug(
                        f'mask shape for {ai.get_id()}: {mask.shape}')
                    masks[ai.get_id()] = mask
            except (IOError, OSError, ValueError):
                self.logger.debug(f'No mask file found for {ai.get_id()}')
        if not masks:
            masks = None

        # Perform integration(s)
        ais_pyfai = {ai.get_id(): ai.ai for ai in ais}
        for integration in self.config.integrations:

            # Add a NXprocess object(s) to the NXroot
            nxprocess = NXprocess()
            try:
                nxroot[f'{nxroot.default}_{integration.name}'] = nxprocess
            except ValueError:
                # Copy nxroot if nxroot is read as read-only
                nxroot = nxcopy(nxroot)
                nxroot[f'{nxroot.default}_{integration.name}'] = nxprocess
            nxprocess.integration_config = integration.model_dump_json()
            nxprocess.azimuthal_integrators = [
                ai.model_dump_json() for ai in ais]

            # Integrate the data
            results = integration.integrate(ais_pyfai, data, masks)

            # Create the NXdata object with the integrated data
            intensities = results['intensities']
            if self.config.sum_axes:
                coords = []
            elif isinstance(axes, str):
                coords = [
                    v for k, v in independent_dims.items() if k in ais_pyfai]
            else:
                coords = [i for k, v in independent_dims.items()
                          for i in v if k in ais_pyfai]
            if ('azimuthal' in results
                    and results['azimuthal']['unit'] == 'chi_deg'):
                chi = results['azimuthal']['coords']
                if integration.right_handed:
                    chi = -np.flip(chi)
                    intensities = np.flip(intensities, (len(coords)))
                coords.append(NXfield(chi, 'chi', attrs={'units': 'deg'}))
            if results['radial']['unit'] == 'q_A^-1':
                unit = Unit.INV_ANGSTROM.symbol
                coords.append(
                    NXfield(
                        results['radial']['coords'], 'q',
                        attrs={'units': unit}))
            else:
                coords.append(
                    NXfield(
                        results['radial']['coords'], 'r'))#,
#                        attrs={'units': '\u212b'}))
                self.logger.warning(
                    f'Unknown radial unit: {results["radial"]["unit"]}')
            nxdata = NXdata(NXfield(intensities, 'integrated'), tuple(coords))
            if not isinstance(axes, str):
                nxdata.attrs['unstructured_axes'] = nxdata.attrs['axes'][:-1]
                del nxdata.attrs['axes']
            nxprocess.data = nxdata
            nxprocess.default = 'data'

        self.config.azimuthal_integrators = ais

        return nxroot


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
