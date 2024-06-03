#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Rolf Verberg
Description: Module for Processors used only by GIWAXS experiments
"""
# System modules
from copy import deepcopy
from json import dumps
import os

# Third party modules
import numpy as np

# Local modules
from CHAP.processor import Processor
from CHAP.common.models.map import MapConfig


class GiwaxsConversionProcessor(Processor):

    def process(self,
                data,
                map_config=None,
                detectors=None,
                scan_step_indices=None,
                save_raw_data=False,
                save_figures=False,
                interactive=False,
                inputdir='.',
                outputdir='.'):

        # Load the validated GIWAXS conversion configuration
        try:
            config = self.get_config(
                data, 'giwaxs.models.GiwaxsConversionConfig',
                inputdir=inputdir)
        except Exception as data_exc:
            self.logger.info('No valid conversion config in input pipeline '
                             'data, using config parameter instead.')
            try:
                # Local modules
                from CHAP.giwaxs.models import GiwaxsConversionConfig

                config = GiwaxsConversionConfig(
                    inputdir=inputdir, map_config=map_config,
                    detectors=detectors, scan_step_indices=scan_step_indices)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        nxroot = self.get_nxroot(
            config, save_raw_data=save_raw_data, save_figures=save_figures,
            interactive=interactive, outputdir=outputdir)

        return nxroot

    def get_nxroot(
            self, config, save_raw_data=False, save_figures=False,
            interactive=False, outputdir='.'):
        """Return NXroot containing the converted GIWAXS images.

        :param config: The conversion configuration.
        :type config: CHAP.giwaxs.models.GiwaxsConversionConfig
        :param save_raw_data: Save the raw data in the NeXus output,
            default to `False`.
        :type save_figures: bool, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :return: NXroot containing the converted GIWAXS images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        if interactive or save_figures:
            import matplotlib.pyplot as plt
        from nexusformat.nexus import (
            NXdata,
            NXfield,
            NXprocess,
            NXroot,
        )

        # Local modules
        from CHAP.common import MapProcessor

        map_config = config.map_config
        if len(config.detectors) > 1:
            raise RuntimeError('More than one detector not yet implemented')
        detector = config.detectors[0]
        if len(map_config.dims) > 1:
            raise RuntimeError(
                'More than one independent dimension not yet implemented')
        if config.scan_step_indices is None:
            thetas = map_config.coords[map_config.dims[0]]
        else:
            thetas = [map_config.coords[map_config.dims[0]][i]
                      for i in config.scan_step_indices]

        # Create the NXroot object
        nxroot = NXroot()
        nxroot[map_config.title] = MapProcessor.get_nxentry(
            map_config)
        nxentry = nxroot[map_config.title]
        nxroot[f'{map_config.title}_conversion'] = NXprocess()
        nxprocess = nxroot[f'{map_config.title}_conversion']
        nxprocess.conversion_config = dumps(config.dict())

        # Collect the raw giwaxs images
        self.logger.debug(f'Reading data ...')
        giwaxs_data = config.giwaxs_data()[0]
        self.logger.debug(f'... done')
        self.logger.debug(f'giwaxs_data.shape: {giwaxs_data.shape}')
        effective_map_shape = giwaxs_data.shape[:-2]
        self.logger.debug(f'effective_map_shape: {effective_map_shape}')
        image_dims = giwaxs_data.shape[1:]
        self.logger.debug(f'image_dims: {image_dims}')

        # Get the components of q parallel and perpendicular to the
        # detector
        q_par, q_perp = self._calc_q_coords(
            giwaxs_data, thetas, detector.poni_file)
 
        # Get the range of the perpendicular component of q and that
        # of the parallel one at near grazing incidence as well as
        # the corresponding rectangular grid with the same dimensions
        # as the detector grid and converted to this grid from the
        # (q_par, q_perp)-grid
        giwaxs_data_rect = []
        q_par_rect = []
        q_perp_rect = []
        for i, theta in enumerate(thetas):
            q_perp_min_index = np.argmin(np.abs(q_perp[i,:,0]))
            q_par_rect.append(np.linspace(
                q_par[i,q_perp_min_index,:].min(),
                q_par[i,q_perp_min_index,:].max(), image_dims[1]))
            q_perp_rect.append(np.linspace(
                q_perp[i].min(), q_perp[i].max(), image_dims[0]))
            giwaxs_data_rect.append(
                GiwaxsConversionProcessor.curved_to_rect(
                    giwaxs_data[i], q_par[i], q_perp[i], q_par_rect[i],
                    q_perp_rect[i]))

            if interactive or save_figures:
                vmax = giwaxs_data[i].max()/10
                fig, ax = plt.subplots(1,2, figsize=(10, 5))
                ax[1].imshow(
                    giwaxs_data_rect[i],
                    vmin=0, vmax=vmax,
                    origin='lower',
                    extent=(q_par[i].min(), q_par[i].max(),
                            q_perp[i].min(), q_perp[i].max()))
                ax[1].set_aspect('equal')
                ax[1].set_title('Transformed Image')
                ax[1].set_xlabel('q$_\parallel$ [\u212b$^{-1}$]')
                ax[1].set_ylabel('q$_\perp$ [\u212b$^{-1}$]')
                im = ax[0].imshow(giwaxs_data[i], vmin=0, vmax=vmax)
                ax[0].set_aspect('equal')
                lhs = ax[0].get_position().extents
                rhs = ax[1].get_position().extents
                ax[0].set_position(
                    (lhs[0], rhs[1], rhs[2] - rhs[0], rhs[3] - rhs[1]))
                ax[0].set_title('Raw Image');
                ax[0].set_xlabel('column index')
                ax[0].set_ylabel('row index')
                fig.subplots_adjust(right=0.85)
                cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                if interactive:
                    plt.show()
                if save_figures:
                    fig.savefig(os.path.join(
                        outputdir,
                        f'converted_{config.scan_step_indices[i]}'))
                plt.close()

        # Create the NXdata object with the converted images
        if len(thetas) == 1:
            nxprocess.data = NXdata(
                NXfield(np.asarray(giwaxs_data_rect[0]), 'converted'),
                (NXfield(
                     q_perp_rect[0], 'q_perp_rect',
                     attrs={'units': '\u212b$^{-1}$'}),
                 NXfield(
                     q_par_rect[0], 'q_par_rect',
                     attrs={'units': '\u212b$^{-1}$'})))
            nxprocess.data.theta = NXfield(
                thetas[0], 'thetas', attrs={'units': 'rad'})
            if save_raw_data:
                nxprocess.data.raw = NXfield(giwaxs_data[0])
        else:
            nxprocess.data = NXdata(
                NXfield(np.asarray(giwaxs_data_rect), 'converted'),
                (NXfield(
                     thetas, 'thetas', attrs={'units': 'rad'}),
                 NXfield(
                     q_perp_rect, 'q_perp_rect',
                     attrs={'units': '\u212b$^{-1}$'}),
                 NXfield(
                     q_par_rect, 'q_par_rect',
                     attrs={'units': '\u212b$^{-1}$'})))
            if save_raw_data:
                nxprocess.data.raw = NXfield(giwaxs_data)
        nxprocess.default = 'data'

        return nxroot

    @staticmethod
    def curved_to_rect(
        data_curved, q_par, q_perp, q_par_rect, q_perp_rect,
        return_maps=False, normalize=True):
        """
        data_rect = curved_to_rect(...):
            distributes counts from a curvilinear grid (data_curved),
            e.g. x-ray data collected in angular space, into a
            rectilinear grid (reciprocal space). 
        
        data_rect, norm, xmap, ymap, xwid, ywid =
                curved_to_rect(..., return_maps=True):
            distributes counts from a curvilinear grid (data_curved),
            e.g. x-ray data collected in angular space, into a
            rectilinear grid (reciprocal space). 
 
        q_par, q_perp, and data_curved are M x N following the normal
            convention where the the first & second index corrspond to
            the vertical (y) and horizontal (x) locations of the
            scattering pattern.
        q_par, q_perp represent the q coordinates of the center of
            pixels whose intensities are stored in data_curved.
            Reiterating the convention above, q_par and q_perp vary
            primarilly along the 2nd and 1st index, respectively.
        rect_qpar and rect_qperp are evenly-spaced, monotonically
            increasing, arrays determining the new grid.
           
        data_rect : the new matrix with intensity from data_curved
                    disctributed into a regular grid defined by
                    rect_qpar, rect_qpar.
        norm : a matrix with the same shape of data_rect representing
               the area of the pixel in the original angular units. 
               It should be used to normalize the resulting array as
               norm_z = data_rect / norm.

        Algorithm:
           Step 1 : Compute xmap, ymap, which containt the values of
                    q_par and q_perp, but represented in pixel units of
                    the target coordinates rect_qpar, rect_qperp.
                    In other words, xmap(i,j) = 3.4 means that
                    q_par(i,j) lands 2/5 of the q_distance between
                    rect_qpar(3) and rect_qpar(4). Intensity in
                    qpar(i,j) should thus be distributed in a 2:3 ratio
                    among neighboring mini-columns of pixels 3 and 4.
           Step 2 : Use the procedure described by Barna et al
                    (RSI v.70, p. 2927, 1999) to distribute intensity
                    from each source pixel i,j into each of 9
                    destination pixels around the xmap(i,j) and
                    ymap(i,j). Keep track of how many source "pixels"
                    are placed into each bin in the variable, "norm".
                    Note also that if xmap(i,j)-floor(xmap(i,j)) > 0.5,
                    the "center" pixel of the 9 destination pixels is
                    floor(xmap+0.5).
           (Outside this function): The normalized intensity in each
               new pixel can be obtained asI = data_rect./norm, but
               with the caveat that zero values of "norm" should be
               changed to ones first, norm(data_rect == 0) = 1.0.

        Example Usage: 
            1. Compute the values of q_par and q_perp for each pixel in
               the image z (according to scattering geometry).
            2. Set or determing a good target grid, e.g.:        
               min_qpar, max_qpar = q_par.mix(), q_par.max()
               min_qperp, max_qperp = q_perp.mix(), q_perp.max()
               q_par_rect, q_par_step = np.linspace(min_qpar ,
                   max_qpar, image_dim[1], retstep=True)
               q_perp_rect, q_perp_step = np.linspace(min_qperp,
                   max_qperp, image_dim[0], retstep=True)
           3. data_rect = curved_to_rect(data_curved, q_par, q_perp,
                  q_par_rect, q_perp_rect)
           4. plt.imshow(data_rect, extent = [
                  q_par_rect[0], q_par_rect[-1],
                  q_perp_rect[-1], q_perp_rect[0]])
              xlabel(['Q_{||} [' char(197) '^{-1}]'])
              ylabel(['Q_{\perp} [' char(197) '^{-1}]'])
        """
        out_width, out_height = q_par_rect.size, q_perp_rect.size

        # Check correct dimensionality
        dims = data_curved.shape
        assert q_par.shape == dims and q_perp.shape == dims

        data_rect = np.zeros((out_height, out_width))
        norm = np.zeros_like(data_rect)

        rect_width  = q_par_rect[1] - q_par_rect[0]
        rect_height = q_perp_rect[1]- q_perp_rect[0]
        rect_qpar_shift = q_par_rect - rect_width/2.0
        rect_qperp_shift = q_perp_rect - rect_height/2.0

        # Precompute source pixels that are outside the target area
        out_of_bounds = (
            (q_par < rect_qpar_shift[0])
            | (q_par > rect_qpar_shift[-1] + rect_width)
            | (q_perp < rect_qperp_shift[0])
            | (q_perp > rect_qperp_shift[-1] + rect_height))

        # Vectorize the search for where q_par[i, j] and q_perp[i,j]
        # fall on the grid formed by q_par_rect and q_perp_rect
        #
        # 1. Expand rect_qpar_shift (a vector) such that
        #    rect_qpar_shift_cube[i. j, :] is identical to
        #    rect_qpar_shift, and is a rising sequence of values of
        #    qpar
        # 2. Expand q_par such that qpar_cube[i, j, :] all correspond
        #    to the value q_par[i, j].
        # - Note that I found tile first and used that in once case but
        #   not the other. I think broadcast_to should likely be used
        #   for both. But in both cases, it seemed to be easiest or
        #   only possible if the extra dimensions were leading dims,
        #   not trailing. That is the reason for the use of transpose
        #   in qpar_cube.
        rect_qpar_shift_cube = np.tile(rect_qpar_shift, q_par.shape + (1,))
        qpar_cube = np.transpose(np.broadcast_to(
            q_par, ((len(rect_qpar_shift),) + q_par.shape)), (1,2,0))
        rect_qperp_shift_cube = np.tile(rect_qperp_shift, q_perp.shape + (1,))
        qperp_cube = np.transpose(np.broadcast_to(
            q_perp, ((len(rect_qperp_shift),) + q_perp.shape)), (1,2,0))

        # We want the index of the highest rect_qpar_shift that is
        # still below qpar, whereas the argmax # operation yields the
        # first rect_qpar_shift that is above qpar, We subtract 1 to
        # take care of this and then correct for any negative indices
        # to 0.
        highpx_x = np.argmax(qpar_cube < rect_qpar_shift_cube, axis=2) - 1
        highpx_y = np.argmax(qperp_cube < rect_qperp_shift_cube, axis=2) - 1
        highpx_x[highpx_x < 0] = 0
        highpx_y[highpx_y < 0] = 0

        # Compute xmap and ymap    
        xmap = np.where(
            out_of_bounds, np.nan,
            highpx_x - 0.5 + (q_par - rect_qpar_shift[highpx_x]) / rect_width)
        ymap = np.where(
            out_of_bounds, np.nan,
            highpx_y - 0.5
                + (q_perp - rect_qperp_shift[highpx_y]) / rect_height)

        # Optionally, print out-of-bounds pixels
        if np.any(out_of_bounds):
            print(f'Warning: Found {out_of_bounds.sum()} source pixels that '
                  'are outside the target bounding box')
            # out_of_bounds_indices = np.transpose(np.where(out_of_bounds))
            # for i, j in out_of_bounds_indices:
            #     print(f'pixel {i}, {j} is out of bounds...skip')

        x1 = np.floor(xmap + 0.5).astype(int)
        y1 = np.floor(ymap + 0.5).astype(int)

        # Compute the effective size of each source pixel (for
        # comparison with target pixels)
        xwid = np.abs(np.diff(xmap, axis=1))
        ywid = np.abs(np.diff(ymap, axis=0))

        # Prepend xwid, ywid with their first column, row
        # (respectively) to match shape with xmap, ymap
        xwid = np.insert(xwid, 0, xwid[:,0], axis=1)
        ywid = np.insert(ywid, 0, ywid[0,:], axis=0)

        # Compute mapping of source pixel to up to 9 closest pixels,
        # after Barna (1999)
        col = np.zeros((3,)+xmap.shape)
        row = np.zeros((3,)+xmap.shape)
        col[0,:,:] = np.where(
            0.5 - (xmap - x1 + 0.5)/xwid > 0.0,
            0.5 - (xmap - x1 + 0.5)/xwid, 0.0)
        col[2,:,:] = np.where(
            0.5 + (xmap - x1 - 0.5)/xwid > 0.0,
            0.5 + (xmap - x1 - 0.5)/xwid, 0.0)
        col[1,:,:] = 1.0 - col[0,:,:] - col[2,:,:]
        row[0,:,:] = np.where(
            0.5 - (ymap - y1 + 0.5)/ywid > 0.0,
            0.5 - (ymap - y1 + 0.5)/ywid , 0.0)
        row[2,:,:] = np.where(
            0.5 + (ymap - y1 - 0.5)/ywid > 0.0,
            0.5 + (ymap - y1 - 0.5)/ywid, 0.0)
        row[1,:,:] = 1.0 - row[0,:,:] - row[2,:,:]

        for k in (-1, 0, 1):
            for m in (-1, 0, 1):
                source_indices = (x1+k > -1) & (x1+k < out_width) & (y1+m > -1) & (y1+m < out_height)
                x1_sub = x1[source_indices]+k
                y1_sub = y1[source_indices]+m

                np.add.at(data_rect, (y1_sub, x1_sub),
                          data_curved[source_indices] * col[k+1, source_indices] * row[m+1, source_indices])
                np.add.at(norm, (y1_sub, x1_sub),
                          col[k+1, source_indices] * row[m+1, source_indices])
                # The following fails because although
                # [y1_sub, x1_sub] can refer to the same location more
                # than once, the "+=" operation acts on the original
                # value and does not have knowledge of incremental
                # changes.
                # data_rect[y1_sub, x1_sub] += (data_curved[source_indices]
                #   * col[k+1, source_indices] * row[m+1, source_indices])
                # norm[y1_sub, x1_sub] += \
                #     col[k+1, source_indices] * row[m+1, source_indices]

        if normalize:
            norm[norm == 0] = 1.0
            data_rect /= norm

        if return_maps:
            return data_rect, norm, xmap, ymap, xwid, ywid
        else:
            return data_rect

    def _calc_q_coords(self, images, thetas, poni_file):
        """Return a 3D arrays representing the perpendicular and
        parallel components of q relative to the detector surface
        for each pixel in an image for each theta.
        """
        # Third party modules
        from pyFAI import load

        # Load the PONI file info:
        # PONI coordinates relative to the left bottom detector corner
        # viewed along the beam with the "1" and "2" directions along
        # the detector rows and columns, respectively
        poni = load(poni_file)
        assert poni.get_shape() == images.shape[1:]
        image_dim = poni.get_shape()
        sample_to_detector = poni.dist*1000     # Sample to detector  (mm)
        pixel_size = round(poni.pixel1*1000, 3) # Pixel size (mm)
        poni1 = poni.poni1*1000 # Point of normal incidence 1 (mm)
        poni2 = poni.poni2*1000 # Point of normal incidence 2 (mm)
        rot1 = poni.rot1        # Rotational angle 1 (rad)
        rot2 = poni.rot2        # Rotational angle 2 (rad)
        xray_wavevector = 2.e-10*np.pi / poni.wavelength

        # Pixel locations relative to where the incident beam
        # intersects the detector in the GIWAXS coordinates frame
        pixel_vert_position = (poni1 + sample_to_detector*np.tan(rot2)
                               - pixel_size*np.arange(image_dim[0]))
        pixel_hor_position = (pixel_size*np.arange(image_dim[1])
                             - poni2 + sample_to_detector*np.tan(rot1))

        # Deflection angles relative to the incident beam at each
        # pixel location
        delta = np.tile(
            np.arctan(pixel_vert_position/sample_to_detector),
            (image_dim[1],1)).T
        nu = np.tile(
            np.arctan(pixel_hor_position/sample_to_detector),
            (image_dim[0],1))
        sign_nu = 2*(nu>=0)-1

        # Calculate q_par, q_perp
        q_par = []
        q_perp = []
        for theta in thetas:
            alpha = np.deg2rad(theta)
            beta = delta - alpha;
            cosnu = np.cos(nu);
            cosb = np.cos(beta);
            cosa = np.cos(alpha);
            sina = np.sin(alpha);
            q_par.append(sign_nu * xray_wavevector * np.sqrt(
                cosa*cosa + cosb*cosb - 2*cosa*cosb*cosnu))
            q_perp.append(xray_wavevector*(sina + np.sin(beta)))

        return np.asarray(q_par), np.asarray(q_perp)

