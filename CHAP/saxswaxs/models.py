"""Models to help construct containers for results of
``saxswaxs.*CorrectionProcessor`` tools."""

# System modules
from functools import cached_property
import os
import re
from typing import Literal, Optional, Union

# Third party modules
from pydantic import (
    confloat,
    conlist,
    model_validator,
    Field,
    AliasChoices,
)

# Local modules
from CHAP import version as chap_version
from CHAP.models import CHAPBaseModel
from CHAP.common.models.common import IndexSliceConfig
from CHAP.common.models.map import SpecScans


class Background(SpecScans):
    """Configuration for background scan data associated with a
    correction.

    Extends :class:`~CHAP.common.models.map.SpecScans` with an
    optional index slice so that only a subset of scan steps is used
    when reading background images.

    :ivar idx_slice: Index slice selecting which scan steps of the
        background scan(s) to read and average.  Defaults to all steps.
    :vartype idx_slice: IndexSliceConfig
    """

    idx_slice: IndexSliceConfig = IndexSliceConfig()

    def zarr_arrays(self, integration_shape):
        """Return a dictionary describing the zarr array that will hold
        the averaged integrated background intensity.

        :param integration_shape: Shape of one frame of integration
            results, as returned by
            :attr:`~CHAP.common.models.integration.PyfaiIntegratorConfig.result_shape`.
        :type integration_shape: tuple[int, ...]
        :returns: Dict mapping the array name ``'I_background'`` to a
            zarr array specification (``dtype``, ``shape``, and
            ``attributes`` keys) compatible with
            :func:`~CHAP.saxswaxs.utils.dict_to_zarr`.
        :rtype: dict
        """
        return {
            'I_background': {
                'attributes': {
                    'long_name': 'Intensity (a.u)',
                    'units': 'a.u,'
                },
                'dtype': 'float64',
                'shape': integration_shape,
            }
        }


class CorrectionConfig(CHAPBaseModel):
    """Base configuration for a single SAXS/WAXS correction step.

    Describes one correction (flux, flux+absorption, or
    flux+absorption+background) to be applied to azimuthally integrated
    detector data.  Subclasses pin ``correction_type`` and may require
    additional fields such as ``background``.

    :ivar correction_type: Identifies the correction algorithm.
        One of ``'flux'``, ``'flux_absorption'``, or
        ``'flux_absorption_background'``.
    :vartype correction_type: str
    :ivar name: Human-readable name used as the key for this
        correction's group in the output zarr / NeXus tree.
    :vartype name: str
    :ivar uncorrected_data_name: Name of the
        :class:`~CHAP.common.models.integration.PyfaiIntegratorConfig`
        integration whose output serves as the uncorrected input for
        this correction.  Must match the ``name`` field of one of the
        integrations in the associated
        :class:`~CHAP.common.models.integration.PyfaiIntegrationConfig`.
    :vartype uncorrected_data_name: str
    :ivar presample_intensity_reference_rate: Fixed reference counting
        rate for the pre-sample beam intensity monitor.  When ``None``
        the rate is computed from the scan data as
        ``nanmean(presample_intensity / dwell_time_actual)``.
    :vartype presample_intensity_reference_rate: float, optional
    :ivar background: Background scan configuration.  Required for
        ``'flux_absorption'`` and ``'flux_absorption_background'``
        correction types.
    :vartype background: Background, optional
    """

    correction_type: Literal['flux', 'flux_absorption',
                             'flux_absorption_background']
    name: str = Field(validation_alias=AliasChoices('name', 'title'))
    uncorrected_data_name: str = Field(validation_alias=AliasChoices(
        'uncorrected_data_name', 'uncorrected_data_title'))
    presample_intensity_reference_rate: Optional[float] = None
    background: Optional[Background] = None

    def zarr_tree(self, dataset_shape, dataset_chunks, integration_shape,
                  nxlinks=None):
        """Return a dictionary representing the zarr tree for this
        correction's output container.

        The returned tree is compatible with
        :func:`~CHAP.saxswaxs.utils.dict_to_zarr` and, after conversion
        with :class:`~CHAP.common.processor.ZarrToNexusProcessor`,
        produces an ``NXprocess`` group containing a ``data`` sub-group
        with an ``I_corrected`` dataset and, when a ``background`` is
        configured, an ``I_background`` dataset.

        :param dataset_shape: Shape of the measurement (scan) dimensions
            of the output dataset, excluding the integration dimensions.
        :type dataset_shape: tuple[int, ...]
        :param dataset_chunks: Chunk shape along the scan dimensions, or
            ``'auto'``.
        :type dataset_chunks: list[int] or str
        :param integration_shape: Shape of one frame of integration
            results for the integration named by
            ``uncorrected_data_name``.
        :type integration_shape: tuple[int, ...]
        :param nxlinks: NeXus path(s) to link into the ``data`` group.
            When the zarr tree is written to a ``.zarr`` file and
            converted to ``.nxs`` with
            :class:`~CHAP.common.processor.ZarrToNexusProcessor`, each
            path produces an ``NXlink`` whose name is
            ``os.path.basename(path)``.  Accepts a single path string or
            a list of path strings.  All links must be explicit; none are
            auto-generated.
        :type nxlinks: str or list[str], optional
        :returns: Nested dict representing the zarr group tree for this
            correction.
        :rtype: dict
        """
        if isinstance(nxlinks, str):
            nxlinks = [nxlinks]
        data_attrs = {}
        if nxlinks:
            data_attrs['__nxlinks__'] = {
                os.path.basename(p): p for p in nxlinks
            }
        if self.background is None:
            background_arrays = {}
        else:
            background_arrays = self.background.zarr_arrays(integration_shape)
            data_attrs['background'] = str(self.background.model_dump())
        return {
            # NXprocess
            'attributes': {
                'correction_type': self.correction_type,
                'default': 'data',
            },
            'children': {
                'program': 'CHAP.saxswaxs',
                'version': chap_version,
                'data': {
                    # NXdata
                    'attributes': data_attrs,
                    'children': {
                        'I_corrected': {
                            'attributes': {
                                'long_name': 'Intensity (a.u)',
                                'units': 'a.u,'
                            },
                            'dtype': 'float64',
                            'shape': (*dataset_shape, *integration_shape),
                        },
                        **background_arrays,
                    }
                }
            }
        }

    @cached_property
    def processor_name(self):
        """Name of the processor class that implements this correction.

        Derived from ``correction_type`` by capitalising each
        ``'_'``-separated word and appending ``'CorrectionProcessor'``
        (e.g. ``'flux_absorption'`` →
        ``'FluxAbsorptionCorrectionProcessor'``).

        :type: str
        """
        return ''.join(
            [x.capitalize() for x in self.correction_type.split('_')] +
            ['CorrectionProcessor']
        )

    @cached_property
    def processor_module(self):
        """Module object containing the processor class for this
        correction.

        :type: module
        """
        return __import__('CHAP.saxswaxs.processor',
                          fromlist=[self.processor_name])

    @cached_property
    def processor_class(self):
        """Processor class that implements this correction.

        :type: type
        """
        return getattr(self.processor_module, self.processor_name)

    @property
    def processor(self):
        """A new instance of the processor class for this correction,
        initialised with the current configuration.

        :type: Processor
        """
        return self.processor_class(config=self.model_dump())


class CorrectionsConfig(CHAPBaseModel):
    """Configuration container for an ordered list of SAXS/WAXS
    correction steps to apply to integrated detector data.

    :ivar corrections: Ordered list of correction configurations.
    :vartype corrections: list[CorrectionConfig]
    """

    corrections: conlist(item_type=CorrectionConfig)

    def zarr_tree(self, dataset_shape, dataset_chunks,
                  integration_shapes, nxlinks=None):
        """Return a dictionary representing the zarr tree for all
        corrections in this configuration.

        Each correction gets its own sub-group keyed by
        :attr:`CorrectionConfig.name`.  See
        :meth:`CorrectionConfig.zarr_tree` for the structure of each
        sub-group.

        :param dataset_shape: Shape of the measurement (scan) dimensions,
            excluding integration dimensions.
        :type dataset_shape: tuple[int, ...]
        :param dataset_chunks: Chunk shape along the scan dimensions, or
            ``'auto'``.
        :type dataset_chunks: list[int] or str
        :param integration_shapes: Mapping from integration name to the
            shape of one integration result frame.  Used to look up the
            ``integration_shape`` for each correction via
            :attr:`CorrectionConfig.uncorrected_data_name`.
        :type integration_shapes: dict[str, tuple[int, ...]]
        :param nxlinks: NeXus links to inject into each correction's
            ``data`` group.  May be a single path string or list of path
            strings (forwarded to every correction), or a dict keyed by
            correction name mapping each correction to its own path(s).
            See :meth:`CorrectionConfig.zarr_tree` for details on how
            individual paths are handled.
        :type nxlinks: str or list[str] or dict[str, str or list[str]],
            optional
        :returns: Nested dict representing the zarr group tree for all
            corrections.
        :rtype: dict
        """
        if not isinstance(nxlinks, dict):
            nxlinks = {corr.name: nxlinks for corr in self.corrections}
        return {
            'root': {
                'attributes': {},
            },
            'children': {
                corr.name: corr.zarr_tree(
                    dataset_shape, dataset_chunks,
                    integration_shapes.get(
                        corr.uncorrected_data_name, None
                    ),
                    nxlinks=nxlinks.get(corr.name),
                )
                for corr in self.corrections
            }
        }


class FluxCorrectionConfig(CorrectionConfig):
    """Correction configuration for flux-only correction.

    Applies a flux correction that normalises the measured intensity by
    the pre-sample beam monitor counts, referenced to a fixed counting
    rate.  No background scan is required.
    """

    correction_type: Literal['flux'] = 'flux'


class FluxAbsorptionCorrectionConfig(FluxCorrectionConfig):
    """Correction configuration for combined flux and absorption
    correction.

    Extends :class:`FluxCorrectionConfig` with a required background
    scan used to determine the sample transmission.

    :ivar background: Background scan configuration used to compute
        the sample transmission term.
    :vartype background: Background
    """

    correction_type: Literal['flux_absorption'] = 'flux_absorption'
    background: Background


class FluxAbsorptionBackgroundCorrectionConfig(
        FluxAbsorptionCorrectionConfig):
    """Correction configuration for combined flux, absorption, and
    background-subtraction correction with optional thickness
    normalisation.

    Extends :class:`FluxAbsorptionCorrectionConfig` with an integrated
    background subtraction step and an optional sample thickness or
    linear attenuation coefficient for thickness normalisation.  At
    most one of ``sample_thickness_cm`` and ``sample_mu_inv_cm`` may be
    provided.

    :ivar background: Background scan configuration.
    :vartype background: Background
    :ivar sample_thickness_cm: Sample thickness in centimetres.  When
        provided, corrected intensities are divided by this value.
        Mutually exclusive with ``sample_mu_inv_cm``.
    :vartype sample_thickness_cm: float, optional
    :ivar sample_mu_inv_cm: Sample linear attenuation coefficient in
        inverse centimetres.  When provided, the effective thickness is
        derived from the measured transmission.  Mutually exclusive with
        ``sample_thickness_cm``.
    :vartype sample_mu_inv_cm: float, optional
    """

    correction_type: Literal[
        'flux_absorption_background'] = 'flux_absorption_background'
    background: Background
    sample_thickness_cm: Optional[confloat(gt=0)] = None
    sample_mu_inv_cm: Optional[confloat(gt=0)] = None

    @model_validator(mode='after')
    def validate_thickness(self):
        """Ensure ``sample_thickness_cm`` and ``sample_mu_inv_cm`` are
        not both specified.

        :raises ValueError: If both fields are set.
        :returns: The validated model instance.
        :rtype: FluxAbsorptionBackgroundCorrectionConfig
        """
        if self.sample_thickness_cm and self.sample_mu_inv_cm:
            raise ValueError(
                'Use sample_thickness_cm OR sample_mu_inv_cm, not both.'
            )
        return self
