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

    Accepts either ``idx_slice`` (a :class:`~CHAP.common.models.common.IndexSliceConfig`
    dict) or the convenience field ``scan_step_indices`` (a list of
    integers or a compact string such as ``"0-4, 6"``), but not both.
    When ``scan_step_indices`` is given it is converted to the
    equivalent ``idx_slice``; the indices must form a uniformly-spaced
    sequence expressible as a Python :class:`slice`.

    :ivar idx_slice: Index slice selecting which scan steps of the
        background scan(s) to read and average.  Defaults to all steps.
    :vartype idx_slice: IndexSliceConfig
    :ivar scan_step_indices: Convenience alternative to ``idx_slice``.
        A list of integer step indices (or a compact string such as
        ``"0-4, 6"``) that are converted to an ``idx_slice`` during
        validation.  The indices must be uniformly spaced.  Mutually
        exclusive with ``idx_slice``.
    :vartype scan_step_indices: list[int] or str, optional
    """

    idx_slice: IndexSliceConfig = IndexSliceConfig()
    scan_step_indices: Optional[Union[list[int], str]] = Field(default=None, exclude=True)

    @model_validator(mode='before')
    @classmethod
    def fill_idx_slice(cls, data):
        scan_step_indices = data.get('scan_step_indices')
        idx_slice = data.get('idx_slice')
        if scan_step_indices is not None and idx_slice is not None:
            raise ValueError(
                'Specify idx_slice or scan_step_indices, not both.')
        if scan_step_indices is not None:
            if isinstance(scan_step_indices, str):
                from CHAP.utils.general import string_to_list
                scan_step_indices = string_to_list(scan_step_indices)
            # scan_step_indices is now list[int]; derive a uniform slice
            indices = sorted(scan_step_indices)
            if len(indices) == 1:
                start, step = indices[0], 1
            else:
                step = indices[1] - indices[0]
                if step <= 0:
                    raise ValueError(
                        'scan_step_indices must contain distinct, '
                        'positive-step values.')
                diffs = [indices[i+1] - indices[i]
                         for i in range(len(indices) - 1)]
                if len(set(diffs)) != 1:
                    raise ValueError(
                        'scan_step_indices must be uniformly spaced so '
                        'they can be expressed as a slice.')
                start = indices[0]
            stop = indices[-1] + step
            data['idx_slice'] = {'start': start, 'stop': stop, 'step': step}
        return data

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
    :ivar input_data_name: Name (or list of names) of the data
        source(s) that serve as uncorrected input to this correction.
        Each name may be a detector ID, a
        :class:`~CHAP.common.models.integration.PyfaiIntegratorConfig`
        integration name, or another correction name.  When a list is
        given the correction is applied independently to each named
        source and the results are stored as ``I_corrected_{name}``
        per source; when a single name is given a single
        ``I_corrected`` array is stored.
    :vartype input_data_name: str or list[str]
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
    input_data_name: Union[str, list[str]] = Field(
        validation_alias=AliasChoices(
            'input_data_name', 'uncorrected_data_title'))

    @property
    def input_data_names(self) -> list[str]:
        """Return ``input_data_name`` always as a list."""
        if isinstance(self.input_data_name, list):
            return self.input_data_name
        return [self.input_data_name]
    presample_intensity_reference_rate: Optional[float] = None
    background: Optional[Background] = None

    def zarr_tree(self, dataset_shape, dataset_chunks, input_shape,
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
        :param input_shape: Shape of one frame of the uncorrected input,
            or a mapping from source name to frame shape when
            ``input_data_name`` is a list.  Each source's shape
            is either an integration result shape or a raw detector
            image shape ``(H, W)``.
        :type input_shape: tuple[int, ...] or dict[str, tuple[int, ...]]
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
            # Use the shape of the first source for background allocation
            first_shape = (
                next(iter(input_shape.values()))
                if isinstance(input_shape, dict) else input_shape
            )
            background_arrays = self.background.zarr_arrays(first_shape)
            data_attrs['background'] = str(self.background.model_dump())
        # Build per-source I_corrected arrays.  When input_data_name
        # is a list each source gets its own I_corrected_{src} array; when
        # it is a single name a single I_corrected array is used.
        if isinstance(self.input_data_name, list):
            corrected_arrays = {
                f'I_corrected_{src}': {
                    'attributes': {
                        'long_name': 'Intensity (a.u)',
                        'units': 'a.u,'
                    },
                    'dtype': 'float64',
                    'shape': (*dataset_shape, *input_shape[src]),
                }
                for src in self.input_data_name
            }
        else:
            corrected_arrays = {
                'I_corrected': {
                    'attributes': {
                        'long_name': 'Intensity (a.u)',
                        'units': 'a.u,'
                    },
                    'dtype': 'float64',
                    'shape': (*dataset_shape, *input_shape),
                }
            }
        return {
            # NXprocess
            'attributes': {
                'correction_type': self.correction_type,
                'default': 'data',
                'program': 'CHAP.saxswaxs',
                'version': chap_version,
            },
            'children': {
                'data': {
                    # NXdata
                    'attributes': data_attrs,
                    'children': {
                        **corrected_arrays,
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
                  input_shapes, nxlinks=None):
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
        :param input_shapes: Mapping from correction ``name`` to the
            frame shape(s) of its uncorrected input.  For corrections
            with a single ``input_data_name`` the value is a
            ``tuple``; for corrections with a list of names the value is
            a ``dict`` mapping each source name to its frame shape.
        :type input_shapes: dict[str, tuple[int, ...] or dict[str, tuple[int, ...]]]
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
                    input_shapes.get(corr.name, None),
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
