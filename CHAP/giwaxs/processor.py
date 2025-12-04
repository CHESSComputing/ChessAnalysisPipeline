#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File       : processor.py
Author     : Rolf Verberg
Description: Module for Processors used only by GIWAXS experiments
"""
# System modules
from json import loads
import os

# Third party modules
import numpy as np

# Local modules
from CHAP.processor import Processor


class GiwaxsConversionProcessor(Processor):
    """A processor for converting GIWAXS images from curved to
    rectangular coordinates.
    """
    def process(
            self, data, config, save_figures=False, inputdir='.',
            outputdir='.', interactive=False):
        """Process the GIWAXS input images & configuration and returns
        a map of the images in rectangular coordinates as a
        `nexusformat.nexus.NXroot` object.

        :param data: Results of `common.MapProcessor` containing the
            map of GIWAXS input images.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            giwaxs.models.GiwaxsConversionConfig.
        :type config: dict
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :return: Converted GIWAXS images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )

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

        # Load the validated GIWAXS conversion configuration
        giwaxs_config = self.get_config(
            data=data, config=config, inputdir=inputdir,
            schema='giwaxs.models.GiwaxsConversionConfig')

        return self.convert_q_rect(
            nxroot, giwaxs_config, save_figures=save_figures,
            interactive=interactive, outputdir=outputdir)

    def convert_q_rect(
            self, nxroot, config, save_figures=False, interactive=False,
            outputdir='.'):
        """Return NXroot containing the converted GIWAXS images.

        :param nxroot: GIWAXS map with the raw detector data.
        :type nxroot: nexusformat.nexus.NXroot
        :param config: GIWAXS conversion configuration.
        :type config: CHAP.giwaxs.models.GiwaxsConversionConfig
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :return: Converted GIWAXS images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        if interactive or save_figures:
            import matplotlib.pyplot as plt
        from nexusformat.nexus import (
            NXdata,
            NXfield,
            NXprocess,
        )

        # Add the NXprocess object to the NXroot
        nxprocess = NXprocess()
        try:
            nxroot[f'{nxroot.default}_converted'] = nxprocess
        except Exception:
            # Local imports
            from CHAP.utils.general import nxcopy

            # Copy nxroot if nxroot is read as read-only
            nxroot = nxcopy(nxroot)
            nxroot[f'{nxroot.default}_converted'] = nxprocess
        nxprocess.conversion_config = config.model_dump_json()

        # Validate the azimuthal integrators and independent dimensions
        nxentry = nxroot[nxroot.default]
        nxdata = nxentry[nxentry.default]
        ais = config.azimuthal_integrators
        if len(ais) > 1:
            raise RuntimeError(
                'More than one azimuthal integrator not yet implemented')
        if ais[0].id not in nxdata:
            raise RuntimeError('Unable to find detector data for '
                               f'{ais[0].id} in {nxentry.tree}')
        if not isinstance(nxdata.attrs['axes'], str):
            raise RuntimeError(
                'More than one independent dimension not yet implemented')

        # Collect the raw giwaxs images
        if config.scan_step_indices is None:
            thetas = nxdata[nxdata.attrs['axes']]
            giwaxs_data = nxdata[ais[0].id]
        else:
            thetas = nxdata[nxdata.attrs['axes']][config.scan_step_indices]
            giwaxs_data = nxdata[ais[0].id][config.scan_step_indices]
        self.logger.debug(f'giwaxs_data.shape: {giwaxs_data.shape}')
        effective_map_shape = giwaxs_data.shape[:-2]
        self.logger.debug(f'effective_map_shape: {effective_map_shape}')
        image_dims = giwaxs_data.shape[1:]
        self.logger.debug(f'image_dims: {image_dims}')

        # Get the components of q parallel and perpendicular to the
        # detector
        q_par, q_perp = self._calc_q_coords(giwaxs_data, thetas, ais[0].ai)

        # Get the range of the perpendicular component of q and that
        # of the parallel one at near grazing incidence as well as
        # the corresponding rectangular grid with the same dimensions
        # as the detector grid and converted to this grid from the
        # (q_par, q_perp)-grid
        # RV: For now use the same q-coords for all thetas, based on
        # the range for the first theta
        q_perp_min_index = np.argmin(np.abs(q_perp[:,0]))
        q_par_rect = np.linspace(
            q_par[q_perp_min_index,:].min(),
            q_par[q_perp_min_index,:].max(), image_dims[1])
        q_perp_rect = np.linspace(
            q_perp.min(), q_perp.max(), image_dims[0])
        giwaxs_data_rect = []
#        q_par_rect = []
#        q_perp_rect = []
        for i in range(len(thetas)):
#            q_perp_min_index = np.argmin(np.abs(q_perp[i,:,0]))
#            q_par_rect.append(np.linspace(
#                q_par[i,q_perp_min_index,:].min(),
#                q_par[i,q_perp_min_index,:].max(), image_dims[1]))
#            q_perp_rect.append(np.linspace(
#                q_perp[i].min(), q_perp[i].max(), image_dims[0]))
#            giwaxs_data_rect.append(
#                GiwaxsConversionProcessor.curved_to_rect(
#                    giwaxs_data[i], q_par[i], q_perp[i], q_par_rect[i],
#                    q_perp_rect[i]))
            giwaxs_data_rect.append(
                GiwaxsConversionProcessor.curved_to_rect(
                    giwaxs_data[i], q_par, q_perp, q_par_rect,
                    q_perp_rect))

            if interactive or save_figures:
                vmax = giwaxs_data[i].max()/10
                fig, ax = plt.subplots(1,2, figsize=(10, 5))
                ax[1].imshow(
                    giwaxs_data_rect[i],
                    vmin=0, vmax=vmax,
                    origin='lower',
                    extent=(q_par_rect.min(), q_par_rect.max(),
                            q_perp_rect.min(), q_perp_rect.max()))
                ax[1].set_aspect('equal')
                ax[1].set_title('Transformed Image')
                ax[1].set_xlabel(r'q$_\parallel$'+' [\u212b$^{-1}$]')
                ax[1].set_ylabel(r'q$_\perp$'+' [\u212b$^{-1}$]')
                im = ax[0].imshow(giwaxs_data[i], vmin=0, vmax=vmax)
                ax[0].set_aspect('equal')
                lhs = ax[0].get_position().extents
                rhs = ax[1].get_position().extents
                ax[0].set_position(
                    (lhs[0], rhs[1], rhs[2] - rhs[0], rhs[3] - rhs[1]))
                ax[0].set_title('Raw Image')
                ax[0].set_xlabel('column index')
                ax[0].set_ylabel('row index')
                fig.subplots_adjust(right=0.85)
                cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                if interactive:
                    plt.show()
                if save_figures:
                    if config.scan_step_indices is None:
                        fig.savefig(os.path.join(outputdir, 'converted'))
                    else:
                        fig.savefig(os.path.join(
                            outputdir,
                            f'converted_{config.scan_step_indices[i]}'))
                plt.close()

        # Create the NXdata object with the converted images
        if False: #RV len(thetas) == 1:
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
        q_par_rect and q_perp_rect are evenly-spaced, monotonically
            increasing, arrays determining the new grid.
           
        data_rect : the new matrix with intensity from data_curved
                    disctributed into a regular grid defined by
                    q_par_rect, q_perp_rect.
        norm : a matrix with the same shape of data_rect representing
               the area of the pixel in the original angular units. 
               It should be used to normalize the resulting array as
               norm_z = data_rect / norm.

        Algorithm:
           Step 1 : Compute xmap, ymap, which containt the values of
                    q_par and q_perp, but represented in pixel units of
                    the target coordinates q_par_rect, q_perp_rect.
                    In other words, xmap(i,j) = 3.4 means that
                    q_par(i,j) lands 2/5 of the q_distance between
                    q_par_rect(3) and q_par_rect(4). Intensity in
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
              xlabel(r'q$_\parallel$'' [\u212b$^{-1}$]')
              ylabel(r'q$_\perp$'' [\u212b$^{-1}$]')
        """
        out_width, out_height = q_par_rect.size, q_perp_rect.size

        # Check correct dimensionality
        dims = data_curved.shape
        assert q_par.shape == dims and q_perp.shape == dims

        data_rect = np.zeros((out_height, out_width))
        norm = np.zeros_like(data_rect)

        rect_width  = q_par_rect[1] - q_par_rect[0]
        rect_height = q_perp_rect[1]- q_perp_rect[0]
        q_par_rect_shift = q_par_rect - rect_width/2.0
        q_perp_rect_shift = q_perp_rect - rect_height/2.0

        # Precompute source pixels that are outside the target area
        out_of_bounds = (
            (q_par < q_par_rect_shift[0])
            | (q_par > q_par_rect_shift[-1] + rect_width)
            | (q_perp < q_perp_rect_shift[0])
            | (q_perp > q_perp_rect_shift[-1] + rect_height))

        # Vectorize the search for where q_par[i, j] and q_perp[i,j]
        # fall on the grid formed by q_par_rect and q_perp_rect
        #
        # 1. Expand q_par_rect_shift (a vector) such that
        #    q_par_rect_shift_cube[i. j, :] is identical to
        #    q_par_rect_shift, and is a rising sequence of values of
        #    qpar
        # 2. Expand q_par such that qpar_cube[i, j, :] all correspond
        #    to the value q_par[i, j].
        # - Note that I found tile first and used that in once case but
        #   not the other. I think broadcast_to should likely be used
        #   for both. But in both cases, it seemed to be easiest or
        #   only possible if the extra dimensions were leading dims,
        #   not trailing. That is the reason for the use of transpose
        #   in qpar_cube.
        q_par_rect_shift_cube = np.tile(q_par_rect_shift, q_par.shape + (1,))
        qpar_cube = np.transpose(np.broadcast_to(
            q_par, ((len(q_par_rect_shift),) + q_par.shape)), (1,2,0))
        q_perp_rect_shift_cube = np.tile(
            q_perp_rect_shift, q_perp.shape + (1,))
        qperp_cube = np.transpose(np.broadcast_to(
            q_perp, ((len(q_perp_rect_shift),) + q_perp.shape)), (1,2,0))

        # We want the index of the highest q_par_rect_shift that is
        # still below qpar, whereas the argmax # operation yields the
        # first q_par_rect_shift that is above qpar, We subtract 1 to
        # take care of this and then correct for any negative indices
        # to 0.
        highpx_x = np.argmax(qpar_cube < q_par_rect_shift_cube, axis=2) - 1
        highpx_y = np.argmax(qperp_cube < q_perp_rect_shift_cube, axis=2) - 1
        highpx_x[highpx_x < 0] = 0
        highpx_y[highpx_y < 0] = 0

        # Compute xmap and ymap
        xmap = np.where(
            out_of_bounds, np.nan,
            highpx_x - 0.5 + (q_par - q_par_rect_shift[highpx_x]) / rect_width)
        ymap = np.where(
            out_of_bounds, np.nan,
            highpx_y - 0.5
                + (q_perp - q_perp_rect_shift[highpx_y]) / rect_height)

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
                source_indices = ((x1+k > -1) & (x1+k < out_width) &
                                  (y1+m > -1) & (y1+m < out_height))
                x1_sub = x1[source_indices]+k
                y1_sub = y1[source_indices]+m

                np.add.at(data_rect, (y1_sub, x1_sub),
                          data_curved[source_indices] *
                          col[k+1, source_indices] *
                          row[m+1, source_indices])
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
        return data_rect

    def _calc_q_coords(self, images, thetas, ai):
        """Return a 3D arrays representing the perpendicular and
        parallel components of q relative to the detector surface
        for each pixel in an image for each theta.

        :param images: GIWAXS images.
        :type images: numpy.ndarray
        :param thetas: Image theta values.
        :type thetas: numpy.ndarray
        :param ai: Azimuthal integrator.
        :type ai: pyFAI.azimuthalIntegrator.AzimuthalIntegrator
        :return: Perpendicular and parallel components of q relative
            to the detector surface.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        # Load the PONI file info:
        # PONI coordinates relative to the left bottom detector corner
        # viewed along the beam with the "1" and "2" directions along
        # the detector rows and columns, respectively
        assert ai.get_shape() == images.shape[1:]
        image_dim = ai.get_shape()
        sample_to_detector = ai.dist*1000     # Sample to detector  (mm)
        pixel_size = round(ai.pixel1*1000, 3) # Pixel size (mm)
        poni1 = ai.poni1*1000 # Point of normal incidence 1 (mm)
        poni2 = ai.poni2*1000 # Point of normal incidence 2 (mm)
        rot1 = ai.rot1        # Rotational angle 1 (rad)
        rot2 = ai.rot2        # Rotational angle 2 (rad)
        xray_wavevector = 2.e-10*np.pi / ai.wavelength

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
        # RV: For now use the same q-coords for all thetas, based on
        # the range for the first theta
#        q_par = []
#        q_perp = []
#        for theta in thetas:
#            alpha = np.deg2rad(theta)
#            beta = delta - alpha;
#            cosnu = np.cos(nu);
#            cosb = np.cos(beta);
#            cosa = np.cos(alpha);
#            sina = np.sin(alpha);
#            q_par.append(sign_nu * xray_wavevector * np.sqrt(
#                cosa*cosa + cosb*cosb - 2*cosa*cosb*cosnu))
#            q_perp.append(xray_wavevector*(sina + np.sin(beta)))
        alpha = np.deg2rad(thetas[0])
        beta = delta - alpha
        cosnu = np.cos(nu)
        cosb = np.cos(beta)
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        q_par = sign_nu * xray_wavevector * np.sqrt(
            cosa*cosa + cosb*cosb - 2*cosa*cosb*cosnu)
        q_perp = xray_wavevector*(sina + np.sin(beta))

        return q_par, q_perp


class PyfaiIntegrationProcessor(Processor):
    """A processor for azimuthally integrating images."""
    def process(self, data, config, inputdir='.'):
        """Process the input images & configuration and return a map of
        the azimuthally integrated images.

        :param data: Results of `common.MapProcessor` or other suitable
            preprocessor of the raw detector data containing the map of
            input images.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            giwaxs.models.PyfaiIntegrationConfig.
        :type config: dict
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str, optional
        :return: Integrated images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        import fabio
        from nexusformat.nexus import (
            NXdata,
            NXentry,
            NXfield,
            NXprocess,
            NXroot,
            nxsetconfig,
        )
        from pyFAI.gui.utils.units import Unit

        # Local imports
        from CHAP.utils.general import nxcopy

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

        # Load the validated integration configuration
        config = self.get_config(
            data=data, config=config, inputdir=inputdir,
            schema='giwaxs.models.PyfaiIntegrationConfig')

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
            if config.azimuthal_integrators is None:
                # Local modules
                from CHAP.giwaxs.models import AzimuthalIntegratorConfig

                config.azimuthal_integrators = [AzimuthalIntegratorConfig(
                    **converted_ais[0])]
            else:
                converted_ids = [ai['id'] for ai in converted_ais]
                skipped_detectors = []
                ais = []
                for ai in config.azimuthal_integrators:
                    if ai.id in converted_ids:
                        ais.append(ai)
                    else:
                        skipped_detectors.append(ai.id)
                if skipped_detectors:
                    self.logger.warning(
                        f'Skipping detector(s) {skipped_detectors} '
                        '(no converted data)')
                if not ais:
                    raise RuntimeError(
                        'No matching azimuthal integrators found')
                config.azimuthal_integrators = ais
            nxdata = nxprocess_converted.data
            axes = nxdata.attrs['axes']
            if len(nxdata.attrs['axes']) != 3:
                raise RuntimeError('More than one independent dimension '
                                   'not yet implemented')
            axes = axes[0]
            independent_dims[config.azimuthal_integrators[0].id] = \
                nxcopy(nxdata[axes])
            data[config.azimuthal_integrators[0].id] = np.flip(
                nxdata.converted.nxdata, axis=1)
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
                    'More than one detector not yet implemented') from exc
            if config.azimuthal_integrators is None:
                raise ValueError(
                    'Missing azimuthal_integrators parameter in '
                    f'PyfaiIntegrationProcessor.config ({config})') from exc
            nxdata = nxentry[nxentry.default]
            skipped_detectors = []
            ais = []
            for ai in config.azimuthal_integrators:
                if ai.id in nxdata:
                    if nxdata[ai.id].ndim != 3:
                        raise RuntimeError(
                            'Inconsistent raw data dimension '
                            f'{nxdata[ai.id].ndim}') from exc
                    ais.append(ai)
                else:
                    skipped_detectors.append(ai.id)
            if skipped_detectors:
                self.logger.warning('Skipping detector(s) '
                                    f'{skipped_detectors} (no raw data)')
            if not ais:
                raise RuntimeError(
                    'No matching raw detector data found') from exc
            config.azimuthal_integrators = ais
            if 'unstructured_axes' in nxdata.attrs:
                axes = nxdata.attrs['unstructured_axes']
                independent_dims[ais[0].id] = [nxcopy(nxdata[a]) for a in axes]
            elif 'axes' in nxdata.attrs:
                axes = nxdata.attrs['axes']
                independent_dims[ais[0].id] = nxcopy(nxdata[axes])
            else:
                self.logger.warning('Unable to find independent_dimensions')
            data[ais[0].id] = nxdata[ais[0].id]

        # Select the images to integrate
        if False and config.scan_step_indices is not None:
            #FIX
            independent_dims = independent_dims[config.scan_step_indices]
            data = data[config.scan_step_indices]
        self.logger.debug(
            f'data shape(s): {[(k, v.shape) for k, v in data.items()]}')
        if config.sum_axes:
            data = {k:np.sum(v.nxdata, axis=0)[None,:,:]
                    for k, v in data.items()}
            self.logger.debug('data shape(s) after summing: '
                              f'{[(k, v.shape) for k, v in data.items()]}')

        # Read the mask(s)
        masks = {}
        for ai in config.azimuthal_integrators:
            self.logger.debug(f'Reading {ai.mask_file}')
            try:
                with fabio.open(ai.mask_file) as f:
                    mask = f.data
                    self.logger.debug(f'mask shape for {ai.id}: {mask.shape}')
                    masks[ai.id] = mask
            except Exception:
                self.logger.debug('No mask file found for {ai.id}')
        if not masks:
            masks = None

        # Perform integration(s)
        ais = {ai.id: ai.ai for ai in config.azimuthal_integrators}
        for integration in config.integrations:

            # Add a NXprocess object(s) to the NXroot
            nxprocess = NXprocess()
            try:
                nxroot[f'{nxroot.default}_{integration.name}'] = nxprocess
            except Exception:
                # Copy nxroot if nxroot is read as read-only
                nxroot = nxcopy(nxroot)
                nxroot[f'{nxroot.default}_{integration.name}'] = nxprocess
            nxprocess.integration_config = integration.model_dump_json()
            nxprocess.azimuthal_integrators = [
                ai.model_dump_json() for ai in config.azimuthal_integrators]

            # Integrate the data
            results = integration.integrate(ais, data, masks)

            # Create the NXdata object with the integrated data
            intensities = results['intensities']
            if config.sum_axes:
                coords = []
            elif isinstance(axes, str):
                coords = [v for k, v in independent_dims.items() if k in ais]
            else:
                coords = [i for k, v in independent_dims.items()
                          for i in v if k in ais]
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

        return nxroot


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
