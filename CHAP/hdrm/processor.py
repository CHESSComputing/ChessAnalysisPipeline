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


class HdrmOrmfinderProcessor(Processor):
    """A processor for solving the orientation matrix for peaks in
    HDRM images."""
    def __init__(self):
        super().__init__()
        self._B = None
        self._phi = None
        self._eta = None
        self._mu = None
        self._chi = None

    def process(self, data, config=None):
        """Process the stacked input images and the peaks from 
        `hdrm.HdrmPeakfinderProcessor` and return the orientation
        matrix.

        :param data: Results of `common.MapProcessor` or other suitable
            preprocessor of the raw detector data containing the map of
            input images as well as the peaks from
            `hdrm.HdrmPeakfinderProcessor`.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            hdrm.models.HdrmOrmfinderConfig.
        :type config: dict
        :return: Orientation matrix.
        :rtype: nexusformat.nexus.NXroot
        """
        # System modules
        from json import loads

        # Third party modules
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXentry,
            NXlink,
            NXprocess,
            NXparameters,
            NXroot,
            nxsetconfig,
        )

        # Local modules
        from CHAP.edd.processor import get_axes
        from CHAP.giwaxs.models import AzimuthalIntegratorConfig

        nxsetconfig(memory=100000)

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
            config = self.get_config(data, 'hdrm.models.HdrmOrmfinderConfig')
        except Exception:
            self.logger.info('No valid conversion config in input pipeline '
                             'data, using config parameter instead.')
            if config is None:
                config = {}
            try:
                # Local modules
                from CHAP.hdrm.models import HdrmOrmfinderConfig

                config = HdrmOrmfinderConfig(**config)
            except Exception as exc:
                raise RuntimeError from exc

        # Add the NXprocess object to the NXroot
        nxprocess = NXprocess()
        try:
            nxroot[f'{nxroot.default}_orm'] = nxprocess
        except Exception:
            # Local imports
            from CHAP.utils.general import nxcopy

            # Copy nxroot if nxroot is read as read-only
            nxroot = nxcopy(nxroot)
            nxroot[f'{nxroot.default}_orm'] = nxprocess
        nxprocess.ormfinder_config = config.model_dump_json()

        # Load NXdata object created by HdrmPeakfinderProcessor
        nxdata = nxroot[f'{nxroot.default}_peaks'].data
        self._phi = nxdata.phi.nxdata
        self._eta = nxdata.eta.nxdata
        self._mu = nxdata.mu.nxdata
        self._chi = nxdata.chi.nxdata

        # Create the NXdata object to store the ORM data
        nxprocess.data = NXdata()
        if 'axes' in nxdata.attrs:
            nxprocess.data.attrs['axes'] = nxdata.attrs['axes']
        elif 'unstructured_axes' in nxdata.attrs:
            nxprocess.data.attrs['unstructured_axes'] = \
                nxdata.attrs['unstructured_axes']
        for k in nxdata:
            nxprocess.data[k] = NXlink(os.path.join(nxdata.nxpath, k))

        # Find ORM and add the data to the NXprocess object
        #FIX serialization only works for a single detector at this point
        nxprocess.results = NXcollection()
        azimuthal_integrators = loads(str(
            nxroot[nxroot.default+'_hdrm_azimuthal'].azimuthal_integrators))
        nxentry = nxroot[nxroot.default]
        detector_ids = [d.decode() for d in nxentry.detector_ids.nxdata]
        for detector_id in detector_ids:
            if detector_id != azimuthal_integrators['id']:
                raise ValueError(
                    f'Unable to match the detector ID {detector_id} to an '
                    'azimuthal integrator')
            nxpar = NXparameters()
            nxprocess.results[detector_id] = nxpar
            ai = AzimuthalIntegratorConfig(**azimuthal_integrators).ai
            self._wavelenth = ai.wavelength*1.e10
            self._pol = ai.twoThetaArray()*np.cos(ai.chiArray()+(np.pi*0.5))
            self._az = -ai.twoThetaArray()*np.sin(ai.chiArray()+(np.pi*0.5))
            peaks = nxdata[f'{detector_id}_peaks']
            num_peak = len(peaks)
            self.logger.info(f'Loaded {num_peak} peak points from file')

            prelimflag = 0
            if num_peak > 75:
                short_peak_list = peaks[30:]
                short_num_peak = len(short_peak_list)
                prelimflag = 1
            if num_peak > 600:
                peaks = peaks[600:]
                num_peak=len(peaklist)

            self._calcB(config.materials[0].lattice_parameters)

            euler, chisq, UB = self._fit_peaks(peaks, T=0.002)
            self.logger.info(f'Euler angles: {euler}, chisq: {chisq}')

            euler, chisq, UB = self._fit_peaks(
                peaks, x0=euler, T=0.005, stepsize=np.pi/30.0, niter=100,
                seed=23)
            self.logger.info(f'Euler angles: {euler}, chisq: {chisq}')
            nxpar.euler = euler
            nxpar.chisq = chisq
            nxpar.orm = UB

        return nxroot

    def _calcB(self, lattice_parameters):
        a1 = lattice_parameters[0]
        a2 = lattice_parameters[1]
        a3 = lattice_parameters[2]
        alpha1 = lattice_parameters[3]
        alpha2 = lattice_parameters[4]
        alpha3 = lattice_parameters[5]
        Vt = 1.0 - np.cos(alpha1)**2 - np.cos(alpha2)**2 - np.cos(alpha3)**2
        Vt = Vt + 2.0 * np.cos(alpha1) * np.cos(alpha2) * np.cos(alpha3)
        V = a1 * a2 * a3 * np.sqrt(Vt)
        b1 = 2.0 * np.pi * a2 * a3 * np.sin(alpha1)/V
        b2 = 2.0 * np.pi * a3 * a1 * np.sin(alpha2)/V
        b3 = 2.0 * np.pi * a1 * a2 * np.sin(alpha3)/V
        betanum = np.cos(alpha2) * np.cos(alpha3) - np.cos(alpha1)
        betaden = np.sin(alpha2) * np.sin(alpha3)
        beta1 = np.arccos(betanum / betaden)
        betanum = np.cos(alpha1) * np.cos(alpha3) - np.cos(alpha2)
        betaden = np.sin(alpha1) * np.sin(alpha3)
        beta2 = np.arccos(betanum / betaden)
        betanum = np.cos(alpha1) * np.cos(alpha2) - np.cos(alpha3)
        betaden = np.sin(alpha1) * np.sin(alpha2)
        beta3 = np.arccos(betanum / betaden)
        self._B = np.asarray([
            [b1, b2 * np.cos(beta3), b3 * np.cos(beta2)],
            [0.0, -b2 * np.sin(beta3), b3 * np.sin(beta2) * np.cos(alpha1)],
            [0.0, 0.0, -b3]])

    def _fit_peaks(
            self, peaks, x0=[0, 0, 0], T=1.0, stepsize=0.5, niter=100,
            seed=None, method='BFGS'):
        # Third party modules
        from scipy.spatial.transform import Rotation as R
        from scipy.optimize import basinhopping

        # Local modules
        from CHAP.hdrm.hkl import Calc_HKL
        def calcUB(eu):
            r = R.from_euler('zxz', eu)
            return np.matmul(r.as_matrix(), self._B)

        def min_func(eu, peaks, wl):
            UB = calcUB(eu).T
            chisq = 0.0
            for peak in peaks:
                pol = self._pol[peak[1], peak[2]]
                az = self._az[peak[1], peak[2]]
                phi = self._phi[peak[0]]
                eta = self._eta[peak[0]]
                mu = self._mu[peak[0]]
                chi = self._chi[peak[0]]
                IN = Calc_HKL(
                    np.asarray([pol]), np.asarray([az]),
                    eta, mu, chi, phi, wl, UB)
                dvec = IN - np.rint(IN)
                chisq += np.linalg.norm(dvec)
            return chisq/len(peaks)

        kwargs = {'method': method, 'args': (peaks, self._wavelenth)}
        result = basinhopping(
            min_func, x0=x0, T=T, stepsize=stepsize, niter=niter, seed=seed,
            minimizer_kwargs=kwargs)

        return result.x, result.fun, calcUB(result.x)


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
            nxsetconfig,
        )

        # Local modules
        from CHAP.edd.processor import get_axes

        nxsetconfig(memory=100000)

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

        # Load the detector ids, images and scalar data
        nxentry = nxroot[nxroot.default]
        nxdata = nxentry[nxentry.default]
        nxscalardata = nxentry.scalar_data

        # Create the NXdata object to store the peaks data
        axes = get_axes(nxdata)
        nxprocess.data = NXdata()
        if 'axes' in nxdata.attrs:
            nxprocess.data.attrs['axes'] = nxdata.attrs['axes']
        elif 'unstructured_axes' in nxdata.attrs:
            nxprocess.data.attrs['unstructured_axes'] = \
                nxdata.attrs['unstructured_axes']
        for k in nxdata:
            nxprocess.data[k] = NXlink(os.path.join(nxdata.nxpath, k))
        for k in nxscalardata:
            nxprocess.data[k] = NXlink(os.path.join(nxscalardata.nxpath, k))

        # Find peaks and add the data to the NXprocess object
        detector_ids = [d.decode() for d in nxentry.detector_ids.nxdata]
        for detector_id in detector_ids:
            intensity = nxdata[detector_id]
            peak_max = intensity.max()
            peaks = np.asarray(
                np.where(intensity > config.peak_cutoff * peak_max)).T
            nxprocess.data[f'{detector_id}_peaks'] = peaks
        nxprocess.detector_ids = detector_ids

        return nxroot


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
