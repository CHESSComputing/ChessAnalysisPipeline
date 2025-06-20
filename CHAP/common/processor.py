#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Processors used in multiple experiment-specific
             workflows.
"""

# System modules
from copy import deepcopy
import os

# Third party modules
import numpy as np

# Local modules
from CHAP import Processor


class AnimationProcessor(Processor):
    """A Processor to show and return an animation.
    """
    def process(
            self, data, num_frames, vmin=None, vmax=None, axis=None,
            interval=1000, blit=True, repeat=True, repeat_delay=1000,
            interactive=False):
        """Show and return an animation of image slices from a dataset
        contained in `data`.

        :param data: Input data.
        :type data: list[PipelineData]
        :param num_frames: Number of frames for the animation.
        :type num_frames: int
        :param vmin: Minimum array value in image slice, defaults to
            `None`, which uses the actual minimum value in the slice.
        :type vmin: float
        :param vmax: Maximum array value in image slice, defaults to
            `None`, which uses the actual maximum value in the slice.
        :type vmax: float
        :param axis: Axis direction or name of the image slices,
            defaults to `0`
        :type axis: Union[int, str], optional
        :param interval: Delay between frames in milliseconds (only
            used when interactive=True), defaults to `1000`
        :type interval: int, optional
        :param blit: Whether blitting is used to optimize drawing,
            default to `True`
        :type blit: bool, optional
        :param repeat: Whether the animation repeats when the sequence
            of frames is completed (only used when interactive=True),
            defaults to `True`
        :type repeat: bool, optional
        :param repeat_delay: Delay in milliseconds between consecutive
            animation runs if repeat is `True` (only used when
            interactive=True), defaults to `1000`
        :type repeat_delay: int, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :return: The matplotlib animation.
        :rtype: matplotlib.animation.ArtistAnimation
        """
        # Third party modules
        from matplotlib import animation
        import matplotlib.pyplot as plt

        # Get the default Nexus NXdata object
        data = self.unwrap_pipelinedata(data)[0]
        try:
            nxdata = data.get_default()
        except Exception as exc:
            raise ValueError(
                f'Unable to find an NXdata object in ({data})') from exc
        if nxdata.nxclass != 'NXdata':
            raise ValueError(
                f'Invalid default pathway to an NXdata object in ({data})')

        # Get the frames
        axes = nxdata.attrs.get('axes', None)
        title = f'{nxdata.nxpath}/{nxdata.signal}'
        if nxdata.nxsignal.ndim == 2:
            exit('AnimationProcessor not tested yet for a 2D dataset')
        elif nxdata.nxsignal.ndim == 3:
            if isinstance(axis, int):
                if not 0 <= axis < nxdata.nxsignal.ndim:
                    raise ValueError(f'axis index out of range ({axis} not in '
                                     f'[0, {nxdata.nxsignal.ndim-1}])')
            elif isinstance(axis, str):
                if axes is None or axis not in list(axes.nxdata):
                    raise ValueError(
                        f'Unable to match axis = {axis} in {nxdata.tree}')
                axes = list(axes.nxdata)
                axis = axes.index(axis)
            else:
                raise ValueError(f'Invalid parameter axis ({axis})')
            delta = int(nxdata.nxsignal.shape[axis]/(num_frames+1))
            indices = np.linspace(
                delta, nxdata.nxsignal.shape[axis]-delta, num_frames)
            if not axis:
                frames = [nxdata[nxdata.signal][int(index),:,:]
                          for index in indices]
            elif axis == 1:
                frames = [nxdata[nxdata.signal][:,int(index),:]
                          for index in indices]
            else:
                frames = [nxdata[nxdata.signal][:,:,int(index)]
                          for index in indices]
            if axes is None:
                axes = [i for i in range(3) if i != axis]
                row_coords = range(frames[0].shape[1])
                row_label = f'axis {axes[1]} index'
                column_coords = range(frames[0].shape[0])
                column_label = f'axis {axes[0]} index'
            else:
                axes.pop(axis)
                row_coords = nxdata[axes[1]].nxdata
                row_label = axes[1]
                if 'units' in nxdata[axes[1]].attrs:
                    row_label += f' ({nxdata[axes[1]].units})'
                column_coords = nxdata[axes[0]].nxdata
                column_label = axes[0]
                if 'units' in nxdata[axes[0]].attrs:
                    column_label += f' ({nxdata[axes[0]].units})'
        else:
            raise ValueError('Invalid data dimension (must be 2D or 3D)')

        # Create the movie
        if vmin is None or vmax is None:
            a_max = frames[0].max()
            for n in range(1, num_frames):
                a_max = min(a_max, frames[n].max())
            a_max = float(a_max)
            if vmin is None:
                vmin = -a_max
            if vmax is None:
                vmax = a_max
        extent = (
            row_coords[0], row_coords[-1], column_coords[-1], column_coords[0])
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_title(title, fontsize='xx-large', pad=20)
        ax.set_xlabel(row_label, fontsize='x-large')
        ax.set_ylabel(column_label, fontsize='x-large')
        fig.tight_layout()
        ims = [[plt.imshow(
                    frames[n], extent=extent, origin='lower',
                    vmin=vmin, vmax=vmax, cmap='gray',
                    animated=True)]
               for n in range(num_frames)]
        plt.colorbar()
        if interactive:
            ani = animation.ArtistAnimation(
                fig, ims, interval=interval, blit=blit, repeat=repeat,
                repeat_delay=repeat_delay)
            plt.show()
        else:
            ani = animation.ArtistAnimation(fig, ims, blit=blit)

        return ani


class AsyncProcessor(Processor):
    """A Processor to process multiple sets of input data via asyncio
    module.

    :ivar mgr: The `Processor` used to process every set of input data.
    :type mgr: Processor
    """
    def __init__(self, mgr):
        super().__init__()
        self.mgr = mgr

    def process(self, data):
        """Asynchronously process the input documents with the
        `self.mgr` `Processor`.

        :param data: Input data documents to process.
        :type data: iterable
        """
        # System modules
        import asyncio

        async def task(mgr, doc):
            """Process given data using provided `Processor`.

            :param mgr: The object that will process given data.
            :type mgr: Processor
            :param doc: The data to process.
            :type doc: object
            :return: The processed data.
            :rtype: object
            """
            return mgr.process(doc)

        async def execute_tasks(mgr, docs):
            """Process given set of documents using provided task
            manager.

            :param mgr: The object that will process all documents.
            :type mgr: Processor
            :param docs: The set of data documents to process.
            :type doc: iterable
            """
            coroutines = [task(mgr, d) for d in docs]
            await asyncio.gather(*coroutines)

        asyncio.run(execute_tasks(self.mgr, data))


class BinarizeProcessor(Processor):
    """A Processor to binarize a dataset."""
    def process(
            self, data, nxpath='', interactive=False, method='CHAP',
            num_bin=256, axis=None, remove_original_data=False):
        """Show and return a binarized dataset from a dataset
        contained in `data`. The dataset must either be of type
        `numpy.ndarray` or a NeXus NXobject object with a default path
        to a NeXus NXfield object. 

        :param data: Input data.
        :type data: list[PipelineData]
        :param nxpath: The relative path to a specific NeXus NXentry or
            NeXus NXdata object in the NeXus file tree to read the
            input data from (ignored for Numpy or NeXus NXfield input
            datasets), defaults to `''`
        :type nxpath: str, optional
        :param interactive: Allows for user interactions (ignored
            for any method other than `'manual'`), defaults to `False`.
        :type interactive: bool, optional
        :param method: Binarization method, defaults to `'CHAP'`
            (CHAP's internal implementation of Otzu's method).
        :type method: Literal['CHAP', 'manual', 'otsu', 'yen',
            'isodata', 'minimum']
        :param num_bin: The number of bins used to calculate the
            histogram in the binarization algorithms (ignored for
            method = `'manual'`), defaults to `256`.
        :type num_bin: int, optional
        :param axis: Axis direction of the image slices (ignored
            for any method other than `'manual'`).
        :type axis: int, optional
        :param remove_original_data: Removes the original data field
            (ignored for Numpy input datasets), defaults to `False`.
        :type force_remove_original_data: bool, optional
        :raises ValueError: Upon invalid input parameters.
        :return: The binarized dataset with a return type equal to
            that of the input dataset.
        :rtype: typing.Union[numpy.ndarray, nexusformat.nexus.NXobject]
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
            NXlink,
            NXprocess,
            nxsetconfig,
        )

        # Local modules
        from CHAP.utils.general import (
            is_int,
            nxcopy,
        )

        if method not in [
                'CHAP', 'manual', 'otsu', 'yen', 'isodata', 'minimum']:
            raise ValueError(f'Invalid parameter method ({method})')
        if not is_int(num_bin, gt=0):
            raise ValueError(f'Invalid parameter num_bin ({num_bin})')
        if not isinstance(remove_original_data, bool):
            raise ValueError('Invalid parameter remove_original_data '
                             f'({remove_original_data})')

        nxsetconfig(memory=100000)

        # Get the dataset and make a copy if it is a NeXus NXgroup
        dataset = self.unwrap_pipelinedata(data)[-1]
        if isinstance(dataset, np.ndarray):
            if method == 'manual':
                if axis is not None and not is_int(axis, gt=0, lt=3):
                    raise ValueError(f'Invalid parameter axis ({axis})')
                axes = ['i', 'j', 'k']
            data = dataset
        elif isinstance(dataset, NXfield):
            if method == 'manual':
                if axis is not None and not is_int(axis, gt=0, lt=3):
                    raise ValueError(f'Invalid parameter axis ({axis})')
                axes = ['i', 'j', 'k']
            if isinstance(dataset, NXfield):
                if nxpath not in ('', '/'):
                    self.logger.warning('Ignoring parameter nxpath')
                data = dataset.nxdata
            else:
                try:
                    data = dataset[nxpath].nxdata
                except Exception as exc:
                    raise ValueError(
                        f'Invalid parameter nxpath ({nxpath})') from exc
        else:
            # Get the default Nexus NXdata object
            try:
                nxdefault = dataset.get_default()
            except:
                nxdefault = None
            if nxdefault is not None and nxdefault.nxclass != 'NXdata':
                raise ValueError('Invalid default pathway NXobject type '
                                 f'({nxdefault.nxclass})')
            # Get the requested NeXus NXdata object to binarize
            if nxpath is None:
                nxclass = dataset.nxclass
            else:
                try:
                    nxclass = dataset[nxpath].nxclass
                except Exception as exc:
                    raise ValueError(
                        f'Invalid parameter nxpath ({nxpath})') from exc
            if nxclass == 'NXdata':
                nxdata = dataset[nxpath]
            else:
                if nxdefault is None:
                    raise ValueError('No default pathway to a NXdata object')
                nxdata = nxdefault
            nxsignal = nxdata.nxsignal
            if method == 'manual':
                if hasattr(nxdata.attrs, 'axes'):
                    axes = nxdata.attrs['axes']
                    if isinstance(axis, str):
                        if axis not in axes:
                            raise ValueError(
                                f'Invalid parameter axis ({axis})')
                        axis = axes.index(axis)
                    elif axis is not None and not is_int(axis, gt=0, lt=3):
                        raise ValueError(f'Invalid parameter axis ({axis})')
                else:
                    axes = ['i', 'j', 'k']
                if nxsignal.ndim != 3:
                    raise ValueError('Invalid data dimension (must be 3D)')
            data = nxsignal.nxdata
            # Create a copy of the input NeXus object, removing the
            # default NeXus NXdata object as well as the original
            # dateset if the remove_original_data parameter is set
            exclude_nxpaths = []
            if nxdefault is not None:
                exclude_nxpaths.append(
                    os.path.join(os.path.relpath(
                        nxdefault.nxpath, dataset.nxpath)))
            if remove_original_data:
                if (nxdefault is None
                        or nxdefault.nxpath != nxdata.nxpath):
                    relpath_nxdata = os.path.relpath(
                        nxdata.nxpath, dataset.nxpath)
                    keys = list(nxdata.keys())
                    keys.remove(nxsignal.nxname)
                    for a in nxdata.axes:
                        keys.remove(a)
                    if keys:
                        raise RuntimeError('Not tested yet')
                        exclude_nxpaths.append(os.path.join(
                            os.path.relpath(nxsignal.nxpath, dataset.nxpath)))
                    elif relpath_nxdata == '.':
                        exclude_nxpaths.append(nxsignal.nxname)
                        if dataset.nxclass != 'NXdata':
                            exclude_nxpaths += nxdata.axes
                    else:
                        exclude_nxpaths.append(relpath_nxdata)
                if not (dataset.nxclass == 'NXdata'
                        or nxdata.nxsignal.nxtarget is None):
                    nxsignal = dataset[nxsignal.nxtarget]
                    nxgroup = nxsignal.nxgroup
                    keys = list(nxgroup.keys())
                    keys.remove(nxsignal.nxname)
                    for a in nxgroup.axes:
                        keys.remove(a)
                    if keys:
                        raise RuntimeError('Not tested yet')
                        exclude_nxpaths.append(os.path.join(
                            os.path.relpath(nxsignal.nxpath, dataset.nxpath)))
                    else:
                        exclude_nxpaths.append(os.path.join(
                            os.path.relpath(nxgroup.nxpath, dataset.nxpath)))
            nxobject = nxcopy(dataset, exclude_nxpaths=exclude_nxpaths)

        # Get a histogram of the data
        if method not in ['manual', 'yen']:
            counts, edges = np.histogram(data, bins=num_bin)
            centers = edges[:-1] + 0.5 * np.diff(edges)

        # Calculate the data cutoff threshold
        if method == 'CHAP':
            weights = np.cumsum(counts)
            means = np.cumsum(counts * centers)
            weights = weights[0:-1]/weights[-1]
            means = means[0:-1]/means[-1]
            variances = (means-weights)**2/(weights*(1.-weights))
            threshold = centers[np.argmax(variances)]
        elif method == 'otsu':
            # Third party modules
            from skimage.filters import threshold_otsu

            threshold = threshold_otsu(hist=(counts, centers))
        elif method == 'yen':
            # Third party modules
            from skimage.filters import threshold_yen

            _min = data.min()
            _max = data.max()
            data = 1+(num_bin-1)*(data-_min)/(_max-_min)
            counts, edges = np.histogram(data, bins=num_bin)
            centers = edges[:-1] + 0.5 * np.diff(edges)

            threshold = threshold_yen(hist=(counts, centers))
        elif method == 'isodata':
            # Third party modules
            from skimage.filters import threshold_isodata

            threshold = threshold_isodata(hist=(counts, centers))
        elif method == 'minimum':
            # Third party modules
            from skimage.filters import threshold_minimum

            threshold = threshold_minimum(hist=(counts, centers))
        else:
            # Third party modules
            import matplotlib.pyplot as plt
            from matplotlib.widgets import RadioButtons, Button

            # Local modules
            from CHAP.utils.general import (
                select_roi_1d,
                select_roi_2d,
            )

            def select_direction(direction):
                """Callback function for the "Select direction" input.
                """
                selected_direction.append(radio_btn.value_selected)
                plt.close()

            def accept(event):
                """Callback function for the "Accept" button."""
                selected_direction.append(radio_btn.value_selected)
                plt.close()

            # Select the direction for data averaging
            if axis is not None:
                mean_data = data.mean(axis=axis)
                subaxes = [i for i in range(3) if i != axis]
            else:
                selected_direction = []

                # Setup figure
                title_pos = (0.5, 0.95)
                title_props = {'fontsize': 'xx-large',
                               'horizontalalignment': 'center',
                               'verticalalignment': 'bottom'}
                fig, axs = plt.subplots(ncols=3, figsize=(17, 8.5))
                mean_data = []
                for i, ax in enumerate(axs):
                    mean_data.append(data.mean(axis=i))
                    subaxes = [a for a in axes if a != axes[i]]
                    ax.imshow(mean_data[i], aspect='auto', cmap='gray')
                    ax.set_title(
                        f'Data averaged in {axes[i]}-direction',
                        fontsize='x-large')
                    ax.set_xlabel(subaxes[1], fontsize='x-large')
                    ax.set_ylabel(subaxes[0], fontsize='x-large')
                fig_title = plt.figtext(
                    *title_pos,
                    'Select a direction or press "Accept" for the default one '
                    f'({axes[0]}) to obtain the binary threshold value',
                    **title_props)
                fig.subplots_adjust(bottom=0.25, top=0.85)

                # Setup RadioButtons
                select_text = plt.figtext(
                    0.225, 0.175, 'Averaging direction', fontsize='x-large',
                    horizontalalignment='center', verticalalignment='center')
                radio_btn = RadioButtons(
                    plt.axes([0.175, 0.05, 0.1, 0.1]), labels=axes, active=0)
                radio_cid = radio_btn.on_clicked(select_direction)

                # Setup "Accept" button
                accept_btn = Button(
                    plt.axes([0.7, 0.05, 0.15, 0.075]), 'Accept')
                accept_cid = accept_btn.on_clicked(accept)

                plt.show()

                axis = axes.index(selected_direction[0])
                mean_data = mean_data[axis]
                subaxes = [a for a in axes if a != axes[axis]]

                plt.close()

            # Select the ROI's orthogonal to the selected averaging direction
            bounds = []
            for i, bound in enumerate(['"0"', '"1"']):
                roi = select_roi_2d(
                    mean_data,
                    title=f'Select the ROI to obtain the {bound} data value',
                    title_a=f'Data averaged in the {axes[axis]}-direction',
                    row_label=subaxes[0], column_label=subaxes[1])

                # Select the index range in the selected averaging direction
                if not axis:
                    mean_roi_data = data[:,roi[2]:roi[3],roi[0]:roi[1]].mean(
                        axis=(1,2))
                elif axis == 1:
                    mean_roi_data = data[roi[2]:roi[3],:,roi[0]:roi[1]].mean(
                        axis=(0,2))
                else:
                    mean_roi_data = data[roi[2]:roi[3],roi[0]:roi[1],:].mean(
                        axis=(0,1))

                _range = select_roi_1d(
                    mean_roi_data, preselected_roi=(0, data.shape[axis]),
                    title=f'Select the {axes[axis]}-direction range to obtain '
                          f'the {bound} data bound',
                    xlabel=axes[axis], ylabel='Average data')

                # Obtain the lower/upper data bound
                if not axis:
                    bounds.append(
                        data[
                            _range[0]:_range[1],roi[2]:roi[3],roi[0]:roi[1]
                        ].mean())
                elif axis == 1:
                    bounds.append(
                        data[
                            roi[2]:roi[3],_range[0]:_range[1],roi[0]:roi[1]
                        ].mean())
                else:
                    bounds.append(
                        data[
                            roi[2]:roi[3],roi[0]:roi[1],_range[0]:_range[1]
                        ].mean())

            # Get the data cutoff threshold
            threshold = np.mean(bounds)

        # Apply the data cutoff threshold and return the output
        data = np.where(data<threshold, 0, 1).astype(np.ubyte)
#        from CHAP.utils.general import quick_imshow
#        quick_imshow(data[int(data.shape[0]/2),:,:], block=True)
#        quick_imshow(data[:,int(data.shape[1]/2),:], block=True)
#        quick_imshow(data[:,:,int(data.shape[2]/2)], block=True)
        if isinstance(dataset, np.ndarray):
            return data
        if isinstance(dataset, NXfield):
            attrs = dataset.attrs
            attrs.pop('target', None)
            return NXfield(
                value=data, name=dataset.nxname, attrs=dataset.attrs)
        name = f'{nxsignal.nxname}_binarized'
        if nxobject.nxclass == 'NXdata':
            nxobject[name] = data
            nxobject.attrs['signal'] = name
            return nxobject
        if nxobject.nxclass == 'NXroot':
            nxentry = nxobject[nxobject.default]
        else:
            nxentry = nxobject
        axes = []
        for a in nxdata.axes:
            attrs = nxdata[a].attrs
            attrs.pop('target', None)
            axes.append(
                NXfield(nxdata[a], name=a, attrs=attrs))
        nxentry[name] = NXprocess(
            NXdata(NXfield(data, name=name), axes),
            attrs={'source': nxsignal.nxpath})
        nxdata = nxentry[name].data
        nxentry.data = NXdata(
            NXlink(nxdata.nxsignal.nxpath),
            [NXlink(os.path.join(nxdata.nxpath, a)) for a in nxdata.axes])
        nxentry.data.set_default()
        return nxobject


class ConstructBaseline(Processor):
    """A Processor to construct a baseline for a dataset."""
    def process(
            self, data, x=None, mask=None, tol=1.e-6, lam=1.e6, max_iter=20,
            save_figures=False, outputdir='.', interactive=False):
        """Construct and return the baseline for a dataset.

        :param data: Input data.
        :type data: list[PipelineData]
        :param x: Independent dimension (only used when interactive is
            `True` of when filename is set).
        :param mask: A mask to apply to the spectrum before baseline
           construction.
        :type mask: array-like, optional
        :param tol: The convergence tolerence, defaults to `1.e-6`.
        :type tol: float, optional
        :param lam: The &lambda (smoothness) parameter (the balance
            between the residual of the data and the baseline and the
            smoothness of the baseline). The suggested range is between
            100 and 10^8, defaults to `10^6`.
        :type lam: float, optional
        :param max_iter: The maximum number of iterations,
            defaults to `20`.
        :type max_iter: int, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :return: The smoothed baseline and the configuration.
        :rtype: numpy.array, dict
        """
        try:
            data = np.asarray(self.unwrap_pipelinedata(data)[0])
        except Exception as exc:
            raise ValueError(
                f'The structure of {data} contains no valid data') from exc

        return self.construct_baseline(
            data, x, mask, tol, lam, max_iter, save_figures, outputdir,
            interactive)

    @staticmethod
    def construct_baseline(
            y, x=None, mask=None, tol=1.e-6, lam=1.e6, max_iter=20, title=None,
            xlabel=None, ylabel=None, interactive=False, filename=None):
        """Construct and return the baseline for a dataset.

        :param y: Input data.
        :type y: numpy.array
        :param x: Independent dimension (only used when interactive is
            `True` of when filename is set).
        :type x: array-like, optional
        :param mask: A mask to apply to the spectrum before baseline
           construction.
        :type mask: array-like, optional
        :param tol: The convergence tolerence, defaults to `1.e-6`.
        :type tol: float, optional
        :param lam: The &lambda (smoothness) parameter (the balance
            between the residual of the data and the baseline and the
            smoothness of the baseline). The suggested range is between
            100 and 10^8, defaults to `10^6`.
        :type lam: float, optional
        :param max_iter: The maximum number of iterations,
            defaults to `20`.
        :type max_iter: int, optional
        :param title: Title for the displayed figure.
        :type title: str, optional
        :param xlabel: Label for the x-axis of the displayed figure.
        :type xlabel: str, optional
        :param ylabel: Label for the y-axis of the displayed figure.
        :type ylabel: str, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :param filename: Save a .png of the plot to filename, defaults
            to `None`, in which case the plot is not saved.
        :type filename: str, optional
        :return: The smoothed baseline and the configuration.
        :rtype: numpy.array, dict
        """
        # Third party modules
        if interactive or filename is not None:
            from matplotlib.widgets import TextBox, Button
            import matplotlib.pyplot as plt

        # Local modules
        from CHAP.utils.general import baseline_arPLS

        def change_fig_subtitle(maxed_out=False, subtitle=None):
            """Change the figure's subtitle."""
            if fig_subtitles:
                fig_subtitles[0].remove()
                fig_subtitles.pop()
            if subtitle is None:
                subtitle = r'$\lambda$ = 'f'{lambdas[-1]:.2e}, '
                if maxed_out:
                    subtitle += f'# iter = {num_iters[-1]} (maxed out) '
                else:
                    subtitle += f'# iter = {num_iters[-1]} '
                subtitle += f'error = {errors[-1]:.2e}'
            fig_subtitles.append(
                plt.figtext(*subtitle_pos, subtitle, **subtitle_props))

        def select_lambda(expression):
            """Callback function for the "Select lambda" TextBox."""
            if not expression:
                return
            try:
                lam = float(expression)
                if lam < 0:
                    raise ValueError
            except ValueError:
                change_fig_subtitle(
                    subtitle='Invalid lambda, enter a positive number')
            else:
                lambdas.pop()
                lambdas.append(10**lam)
                baseline, _, _, num_iter, error = baseline_arPLS(
                    y, mask=mask, tol=tol, lam=lambdas[-1], max_iter=max_iter,
                    full_output=True)
                num_iters.pop()
                num_iters.append(num_iter)
                errors.pop()
                errors.append(error)
                if num_iter < max_iter:
                    change_fig_subtitle()
                else:
                    change_fig_subtitle(maxed_out=True)
                baseline_handle.set_ydata(baseline)
            lambda_box.set_val('')
            plt.draw()

        def continue_iter(event):
            """Callback function for the "Continue" button."""
            baseline, _, w, n_iter, error = baseline_arPLS(
                y, mask=mask, w=weights[-1], tol=tol, lam=lambdas[-1],
                max_iter=max_iter, full_output=True)
            num_iters[-1] += n_iter
            errors.pop()
            errors.append(error)
            if n_iter < max_iter:
                change_fig_subtitle()
            else:
                change_fig_subtitle(maxed_out=True)
            baseline_handle.set_ydata(baseline)
            plt.draw()
            weights.pop()
            weights.append(w)

        def confirm(event):
            """Callback function for the "Confirm" button."""
            plt.close()

        baseline, _, w, num_iter, error = baseline_arPLS(
            y, mask=mask, tol=tol, lam=lam, max_iter=max_iter,
            full_output=True)

        if not interactive and filename is None:
            config = {
                'tol': tol, 'lambda': lam, 'max_iter': max_iter,
                'num_iter': num_iter, 'error': error, 'mask': mask}
            return baseline, config

        lambdas = [lam]
        weights = [w]
        num_iters = [num_iter]
        errors = [error]
        fig_subtitles = []

        # Check inputs
        if x is None:
            x = np.arange(y.size)

        # Setup the Matplotlib figure
        title_pos = (0.5, 0.95)
        title_props = {'fontsize': 'xx-large', 'horizontalalignment': 'center',
                       'verticalalignment': 'bottom'}
        subtitle_pos = (0.5, 0.90)
        subtitle_props = {'fontsize': 'x-large',
                          'horizontalalignment': 'center',
                          'verticalalignment': 'bottom'}
        fig, ax = plt.subplots(figsize=(11, 8.5))
        if mask is None:
            ax.plot(x, y, label='input data')
        else:
            ax.plot(
                x[mask.astype(bool)], y[mask.astype(bool)], label='input data')
        baseline_handle = ax.plot(x, baseline, label='baseline')[0]
#        ax.plot(x, y-baseline, label='baseline corrected data')
        ax.legend()
        ax.set_xlabel(xlabel, fontsize='x-large')
        ax.set_ylabel(ylabel, fontsize='x-large')
        ax.set_xlim(x[0], x[-1])
        if title is None:
            fig_title = plt.figtext(*title_pos, 'Baseline', **title_props)
        else:
            fig_title = plt.figtext(*title_pos, title, **title_props)
        if num_iter < max_iter:
            change_fig_subtitle()
        else:
            change_fig_subtitle(maxed_out=True)
        fig.subplots_adjust(bottom=0.0, top=0.85)

        if interactive:

            fig.subplots_adjust(bottom=0.2)

            # Setup TextBox
            lambda_box = TextBox(
                plt.axes([0.15, 0.05, 0.15, 0.075]), r'log($\lambda$)')
            lambda_cid = lambda_box.on_submit(select_lambda)

            # Setup "Continue" button
            continue_btn = Button(
                plt.axes([0.45, 0.05, 0.15, 0.075]), 'Continue smoothing')
            continue_cid = continue_btn.on_clicked(continue_iter)

            # Setup "Confirm" button
            confirm_btn = Button(
                plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
            confirm_cid = confirm_btn.on_clicked(confirm)

            # Show figure for user interaction
            plt.show()

            # Disconnect all widget callbacks when figure is closed
            lambda_box.disconnect(lambda_cid)
            continue_btn.disconnect(continue_cid)
            confirm_btn.disconnect(confirm_cid)

            # ... and remove the buttons before returning the figure
            lambda_box.ax.remove()
            continue_btn.ax.remove()
            confirm_btn.ax.remove()

        if filename is not None:
            fig_title.set_in_layout(True)
            fig_subtitles[-1].set_in_layout(True)
            fig.tight_layout(rect=(0, 0, 1, 0.90))
            fig.savefig(filename)
        plt.close()

        config = {
            'tol': tol, 'lambda': lambdas[-1], 'max_iter': max_iter,
            'num_iter': num_iters[-1], 'error': errors[-1], 'mask': mask}
        return baseline, config


class ConvertStructuredProcessor(Processor):
    """Processor for converting map data between structured /
    unstructued formats.
    """
    def process(self, data):
        # Local modules
        from CHAP.utils.converters import convert_structured_unstructured

        data = self.unwrap_pipelinedata(data)[0]
        return convert_structured_unstructured(data)


class ImageProcessor(Processor):
    """A Processor to plot an image (slice) from a NeXus object."""
    def process(
            self, data, vmin=None, vmax=None, axis=0, index=None,
            coord=None, interactive=False, save_figure=True, outputdir='.',
            filename='image.png'):
        """Plot and/or save an image (slice) from a NeXus NXobject
        object with a default data path contained in `data` and return
        the NeXus NXdata data object.

        :param data: Input data.
        :type data: list[PipelineData]
        :param vmin: Minimum array value in image slice, default to
            `None`, which uses the actual minimum value in the slice.
        :type vmin: float
        :param vmax: Maximum array value in image slice, default to
            `None`, which uses the actual maximum value in the slice.
        :type vmax: float
        :param axis: Axis direction or name of the image slice,
            defaults to `0`
        :type axis: Union[int, str], optional
        :param index: Array index of the slice of data to plot.
        :type index: int, optional
        :param coord: Coordinate value of the slice of data to plot.
        :type coord: Union[int, float], optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :param save_figure: Save a .png of the image, defaults to
            `True`.
        :type save_figure: bool, optional
        :param outputdir: Directory to which any output figure will
            be saved, defaults to `'.'`
        :type outputdir: str, optional
        :param filename: Image filename, defaults to `"image.png"`.
        :type filename: str, optional
        :return: The input data object.
        :rtype: nexusformat.nexus.NXdata
        """
        # Third party modules
        import matplotlib.pyplot as plt

        # Local modules
        from CHAP.utils.general import index_nearest

        # Validate input parameters
        if not isinstance(interactive, bool):
            raise ValueError(f'Invalid parameter interactive ({interactive})')
        if not isinstance(save_figure, bool):
            raise ValueError(f'Invalid parameter save_figure ({save_figure})')
        if not isinstance(outputdir, str):
            raise ValueError(f'Invalid parameter outputdir ({outputdir})')
        if not isinstance(filename, str):
            raise ValueError(f'Invalid parameter filename ({filename})')
        if not os.path.isabs(filename):
            filename = os.path.join(outputdir, filename)

        # Get the default Nexus NXdata object
        data = self.unwrap_pipelinedata(data)[0]
        try:
            nxdata = data.get_default()
        except Exception as exc:
            raise ValueError(
                f'Unable to find an NXdata object in ({data})') from exc
        if nxdata.nxclass != 'NXdata':
            raise ValueError(
                f'Invalid default pathway to an NXdata object in ({data})')

        # Get the data slice
        axes = nxdata.attrs.get('axes', None)
        if axes is not None:
            axes = list(axes.nxdata)
        coords = None
        title = f'{nxdata.nxpath}/{nxdata.signal}'
        if nxdata.nxsignal.ndim == 2:
            exit('ImageProcessor not tested yet for a 2D dataset')
            if axis is not None:
                axis = None
                self.logger.warning('Ignoring parameter axis')
            if index is not None:
                index = None
                self.logger.warning('Ignoring parameter index')
            if coord is not None:
                coord = None
                self.logger.warning('Ignoring parameter coord')
            a = nxdata.nxsignal
        elif nxdata.nxsignal.ndim == 3:
            if isinstance(axis, int):
                if not 0 <= axis < nxdata.nxsignal.ndim:
                    raise ValueError(f'axis index out of range ({axis} not in '
                                     f'[0, {nxdata.nxsignal.ndim-1}])')
            elif isinstance(axis, str):
                if axes is None or axis not in axes:
                    raise ValueError(
                        f'Unable to match axis = {axis} in {nxdata.tree}')
                axis = axes.index(axis)
            else:
                raise ValueError(f'Invalid parameter axis ({axis})')
            if axes is not None and hasattr(nxdata, axes[axis]):
                coords = nxdata[axes[axis]].nxdata
                axis_name = axes[axis]
            else:
                axis_name = f'axis {axis}'
            if index is None and coord is None:
                index = nxdata.nxsignal.shape[axis] // 2
            else:
                if index is not None:
                    if coord is not None:
                        coord = None
                        self.logger.warning('Ignoring parameter coord')
                    if not isinstance(index, int):
                        raise ValueError(f'Invalid parameter index ({index})')
                    if not 0 <= index < nxdata.nxsignal.shape[axis]:
                        raise ValueError(
                            f'index value out of range ({index} not in '
                            f'[0, {nxdata.nxsignal.shape[axis]-1}])')
                else:
                    if not isinstance(coord, (int, float)):
                        raise ValueError(f'Invalid parameter coord ({coord})')
                    if coords is None:
                        raise ValueError(
                            f'Unable to get coordinates for {axis_name} '
                            f'in {nxdata.tree}')
                    index = index_nearest(nxdata[axis_name], coord)
            if coords is None:
                slice_info = f'slice at {axis_name} and index {index}'
            else:
                coord = coords[index]
                slice_info = f'slice at {axis_name} = '\
                             f'{nxdata[axis_name][index]:.3f}'
                if 'units' in nxdata[axis_name].attrs:
                    slice_info += f' ({nxdata[axis_name].units})'
            if not axis:
                a = nxdata[nxdata.signal][index,:,:]
            elif axis == 1:
                a = nxdata[nxdata.signal][:,index,:]
            else:
                a = nxdata[nxdata.signal][:,:,index]
            if coords is None:
                axes = [i for i in range(3) if i != axis]
                row_coords = range(a.shape[1])
                row_label = f'axis {axes[1]} index'
                column_coords = range(a.shape[0])
                column_label = f'axis {axes[0]} index'
            else:
                axes.pop(axis)
                row_coords = nxdata[axes[1]].nxdata
                row_label = axes[1]
                if 'units' in nxdata[axes[1]].attrs:
                    row_label += f' ({nxdata[axes[1]].units})'
                column_coords = nxdata[axes[0]].nxdata
                column_label = axes[0]
                if 'units' in nxdata[axes[0]].attrs:
                    column_label += f' ({nxdata[axes[0]].units})'
        else:
            raise ValueError('Invalid data dimension (must be 2D or 3D)')

        # Create figure
        a_max = a.max()
        if vmin is None:
            vmin = -a_max
        if vmax is None:
            vmax = a_max
        extent = (
            row_coords[0], row_coords[-1], column_coords[-1], column_coords[0])
        fig, ax = plt.subplots(figsize=(11, 8.5))
        plt.imshow(
            a, extent=extent, origin='lower', vmin=vmin, vmax=vmax,
            cmap='gray')
        fig.suptitle(title, fontsize='xx-large')
        ax.set_title(slice_info, fontsize='xx-large', pad=20)
        ax.set_xlabel(row_label, fontsize='x-large')
        ax.set_ylabel(column_label, fontsize='x-large')
        plt.colorbar()
        fig.tight_layout()
        if interactive:
            plt.show()
        if save_figure:
            fig.savefig(filename)
        plt.close()

        return nxdata


class ExpressionProcessor(Processor):
    """Processor to perform an arbitrary expression on input
    data."""
    def process(self, data, expression, symtable=None, nxprocess=False):
        """Return result of plugging input data into the given
        mathematical expression.

        :param data: Input data.
        :type data: list[PipelineData]
        :param expression: Mathemetical expression. May use the
            built-in function `round` and / or numpy functions with
            `np.<function_name>` or `numpy.<function_name>.`
        :type expression: str
        :param symtable: Values to use for names in `expression` that
            should not be obtained from input data. Defaults to `None`.
        :type symtable: dict[str, (float, int)], optional.
        :param nxprocess: Flag to indicate the results should be
            retunred as an `NXprocess`. Defaults to `False`.
        :type nxprocess: bool, optional
        :returns: Result of evaluating the expression.
        :rtype: object
        """
        return self._process(
            data, expression, symtable=symtable, nxprocess=nxprocess)

    def _process(self, data, expression, symtable=None, nxprocess=False):
        from ast import parse
        from asteval import get_ast_names, Interpreter

        names = get_ast_names(parse(expression))
        if symtable is None:
            symtable = {}
        for name in names:
            if name in symtable:
                continue
            elif name == 'round':
                symtable[name] = round
            elif name in ('np', 'numpy'):
                symtable[name] = np
            else:
                symtable[name] = self.get_data(
                    data, name=name, remove=False, nxobject=False)
        aeval = Interpreter(symtable=symtable)
        new_data = aeval(expression)

        if not nxprocess:
            return new_data

        from nexusformat.nexus import (
            NXdata, NXfield, NXprocess
        )
        return NXprocess(
            name='expression',
            entries={
                'data': NXdata(
                    signal=NXfield(
                        name='result',
                        value=new_data,
                        attrs={'expression': expression}
                    ),
                    **{
                        name: NXfield(
                            value=symtable[name]
                        )
                        for name in names
                    },
                    attrs={'expression': expression}
                ),
            }
        )


class MapProcessor(Processor):
    """A Processor that takes a map configuration and returns a NeXus
    NXentry object representing that map's metadata and any
    scalar-valued raw data requested by the supplied map configuration.
    """
    def process(
            self, data, config=None, detectors=None,
            fill_data=True, placeholder_data=False,
            num_proc=1, comm=None, inputdir=None):
        """Process the output of a `Reader` that contains a map
        configuration and returns a NeXus NXentry object representing
        the map.

        :param data: Result of `Reader.read` where at least one item
            has the value `'common.models.map.MapConfig'` for the
            `'schema'` key.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            common.models.map.MapConfig.
        :type config: dict, optional
        :param detectors: Detectors to include raw data for in the
            returned NeXus NXentry object (overruling the detector
            info in data, if present).
        :type detectors: list[dict], optional
        :param fill_data: Flag to indicate whether or not to fill out
            datasets with real data, optional.
        :type fill_data: bool, defaults to `True`
        :param placeholder_data: For SMB EDD maps only. Value to use
            for missing detector data frames, or `False` if missing
            data should raise an error, defaults to `False`.
        :type placeholder_data: object, optional
        :param num_proc: Number of processors used to read map,
            defaults to `1`.
        :type num_proc: int, optional
        :param comm: MPI communicator.
        :type comm: mpi4py.MPI.Comm, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths.
        :type inputdir: str, optional
        :return: Map data and metadata.
        :rtype: nexusformat.nexus.NXentry
        """
        # System modules
        import logging
        from tempfile import NamedTemporaryFile

        # Third party modules
        import yaml

        # Local modules
        from CHAP.runner import RunConfig
        from CHAP.utils.general import (
            is_str_series,
            string_to_list,
        )

        # Get the validated map configuration
        map_config = self.get_config(
            data=data, config=config, schema='common.models.map.MapConfig',
            inputdir=inputdir)

        # Validate the detectors
        try:
            # Local modules
            from CHAP.common.models.map import DetectorConfig

            detector_config = DetectorConfig(detectors=detectors)
        except:
            try:
                detector_config = self.get_config(
                    data=data, schema='common.models.map.DetectorConfig',
                    inputdir=inputdir)
            except Exception as exc:
                raise RuntimeError from exc

        if fill_data:
            # Validate the number of processors
            if not isinstance(num_proc, int):
                self.logger.warning('Ignoring invalid parameter num_proc '
                                    f'({num_proc}), running serially')
                num_proc = 1
            elif num_proc > 1:
                try:
                    # System modules
                    from os import cpu_count

                    # Third party modules
                    from mpi4py import MPI

                    if num_proc > cpu_count():
                        self.logger.warning(
                            f'The requested number of processors ({num_proc}) '
                            'exceeds the maximum number of processors '
                            f'({cpu_count()}): reset it to {cpu_count()}')
                        num_proc = cpu_count()
                except:
                    self.logger.warning(
                        'Unable to load mpi4py, running serially')
                    num_proc = 1
            self.logger.debug(f'Number of processors: {num_proc}')

            # Create the sub-pipeline configuration for each processor
            # FIX: catered to EDD with one spec scan
            assert len(map_config.spec_scans) == 1
            spec_scans = map_config.spec_scans[0]
            scan_numbers = spec_scans.scan_numbers
            num_scan = len(scan_numbers)
            if num_scan < num_proc:
                self.logger.warning(
                    f'The requested number of processors ({num_proc}) exceeds '
                    f'the number of scans ({num_scan}): '
                    f'reset it to {num_scan}')
                num_proc = num_scan
            if num_proc == 1:
                common_comm = comm
                offsets = [0]
            else:
                scans_per_proc = num_scan//num_proc
                num = scans_per_proc
                if num_scan - scans_per_proc*num_proc > 0:
                    num += 1
                spec_scans.scan_numbers = scan_numbers[:num]
                n_scan = num
                pipeline_config = []
                offsets = [0]
                for n_proc in range(1, num_proc):
                    num = scans_per_proc
                    if n_proc < num_scan - scans_per_proc*num_proc:
                        num += 1
                    config = map_config.model_dump()
                    config['spec_scans'][0]['scan_numbers'] = \
                        scan_numbers[n_scan:n_scan+num]
                    pipeline_config.append(
                        [{'common.MapProcessor': {
                            'config': config,
                            'detectors': [
                                d.model_dump()
                                for d in detector_config.detectors],
                         }}])
                    offsets.append(n_scan)
                    n_scan += num

                # Spawn the workers to run the sub-pipeline
                run_config = RunConfig(
                    log_level=logging.getLevelName(self.logger.level), spawn=1)
                tmp_names = []
                with NamedTemporaryFile(delete=False) as fp:
                    fp_name = fp.name
                    tmp_names.append(fp_name)
                    with open(fp_name, 'w') as f:
                        yaml.dump({'config': {'spawn': 1}}, f, sort_keys=False)
                    for n_proc in range(1, num_proc):
                        f_name = f'{fp_name}_{n_proc}'
                        tmp_names.append(f_name)
                        with open(f_name, 'w') as f:
                            yaml.dump(
                                {'config': run_config.model_dump(),
                                 'pipeline': pipeline_config[n_proc-1]},
                                f, sort_keys=False)
                    sub_comm = MPI.COMM_SELF.Spawn(
                        'CHAP', args=[fp_name], maxprocs=num_proc-1)
                    common_comm = sub_comm.Merge(False)
                    # Align with the barrier in RunConfig() on common_comm
                    # called from the spawned main() in common_comm
                    common_comm.barrier()
                    # Align with the barrier in run() on common_comm
                    # called from the spawned main()
                    common_comm.barrier()

            if common_comm is None:
                num_proc = 1
                rank = 0
            else:
                num_proc = common_comm.Get_size()
                rank = common_comm.Get_rank()
            if num_proc == 1:
                offset = 0
            else:
                num_scan = common_comm.bcast(num_scan, root=0)
                offset = common_comm.scatter(offsets, root=0)

            # Read the raw data
            if map_config.experiment_type == 'EDD':
                data, independent_dimensions, all_scalar_data = \
                    self._read_raw_data_edd(
                        map_config, detector_config, common_comm, num_scan,
                        offset, placeholder_data)
            else:
                data, independent_dimensions, all_scalar_data = \
                    self._read_raw_data(
                        map_config, detector_config, common_comm, num_scan,
                        offset)
            if not rank:
                self.logger.debug(f'Data shape: {data.shape}')
                if independent_dimensions is not None:
                    self.logger.debug('Independent dimensions shape: '
                                      f'{independent_dimensions.shape}')
                if all_scalar_data is not None:
                    self.logger.debug('Scalar data shape: '
                                      f'{all_scalar_data.shape}')

            if rank:
                return None

            if num_proc > 1:
                # Reset the scan_numbers to the original full set
                spec_scans.scan_numbers = scan_numbers
                # Align with the barrier in main() on common_comm
                # when disconnecting the spawned worker
                common_comm.barrier()
                # Disconnect spawned workers and cleanup temporary files
                sub_comm.Disconnect()
                for tmp_name in tmp_names:
                    os.remove(tmp_name)
        else:
            # fill_data is False, just use empty arrays
            map_len = 0
            _independent_dimensions = {
                dim.label: [] for dim in map_config.independent_dimensions}
            det_shapes = False
            for scans in map_config.spec_scans:
                for scan_number in scans.scan_numbers:
                    scanparser = scans.get_scanparser(scan_number)
                    map_len += scanparser.spec_scan_npts
                    for dim in map_config.independent_dimensions:
                        val = dim.get_value(
                            scans, scan_number, -1,
                            map_config.scalar_data)
                        if not isinstance(val, list):
                            val = [val]
                        _independent_dimensions[dim.label].extend(val)
                    if not det_shapes:
                        det_shapes = {}
                        for detector in detector_config.detectors:
                            ddata_init = scanparser.get_detector_data(
                                detector.id, 0)
                            if isinstance(ddata_init, tuple):
                                ddata_init = ddata_init[0].squeeze()
                            det_shapes[detector.id] = ddata_init.shape
            all_scalar_data = np.empty(
                (len(map_config.all_scalar_data), map_len))
            data = np.empty(
                (len(detector_config.detectors),
                 map_len,
                 *det_shapes[detector_config.detectors[0].id]))
            independent_dimensions = np.asarray(
                [_independent_dimensions[dim.label]
                 for dim in map_config.independent_dimensions])

        # Construct and return the NeXus NXroot object
        return self._get_nxroot(
            map_config, detector_config, data, independent_dimensions,
            all_scalar_data, placeholder_data)

    def _get_nxroot(
            self, map_config, detector_config, data, independent_dimensions,
            all_scalar_data, placeholder_data):
        """Use a `MapConfig` to construct a NeXus NXroot object.

        :param map_config: A valid map configuration.
        :type map_config: common.models.map.MapConfig
        :param detector_config: A valid detector configuration.
        :type detector_config: common.models.map.DetectorConfig
        :param data: The map's raw data.
        :type data: numpy.ndarray
        :param independent_dimensions: The map's independent
            coordinates.
        :type independent_dimensions: numpy.ndarray
        :param all_scalar_data: The map's scalar data.
        :type all_scalar_data: numpy.ndarray
        :param placeholder_data: For SMB EDD maps only. Value to use
            for missing detector data frames, or `False` if missing
            data should raise an error.
        :type placeholder_data: object
        :return: The map's data and metadata contained in a NeXus
            structure.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXentry,
            NXfield,
            NXlinkfield,
            NXsample,
            NXroot,
        )

        # Local modules:
        from CHAP.common.models.map import PointByPointScanData

        def linkdims(nxgroup, nxdata_source):
            """Link the dimensions for an NXgroup."""
            source_axes = [k for k in nxdata_source.keys()]
            if isinstance(source_axes, str):
                source_axes = [source_axes]
            axes = []
            for dim in source_axes:
                axes.append(dim)
                if isinstance(nxdata_source[dim], NXlinkfield):
                    nxgroup[dim] = nxdata_source[dim]
                else:
                    nxgroup.makelink(nxdata_source[dim])
                if f'{dim}_indices' in nxdata_source.attrs:
                    nxgroup.attrs[f'{dim}_indices'] = \
                        nxdata_source.attrs[f'{dim}_indices']
            if len(axes) == 1:
                nxgroup.attrs['axes'] = axes
            else:
                nxgroup.attrs['unstructured_axes'] = axes

        # Set up NeXus NXroot/NXentry and add CHESS-specific metadata
        nxroot = NXroot()
        nxentry = NXentry(name=map_config.title)
        nxroot[nxentry.nxname] = nxentry
        nxentry.set_default()
        nxentry.map_config = map_config.model_dump_json()
        nxentry.attrs['station'] = map_config.station
        for k, v in map_config.attrs.items():
            nxentry.attrs[k] = v
        nxentry.spec_scans = NXcollection()
        for scans in map_config.spec_scans:
            nxentry.spec_scans[scans.scanparsers[0].scan_name] = \
                NXfield(value=scans.scan_numbers,
                        dtype='int8',
                        attrs={'spec_file': str(scans.spec_file)})

        # Add sample metadata
        nxentry[map_config.sample.name] = NXsample(
            **map_config.sample.model_dump())

        # Set up independent dimensions NeXus NXdata group
        # (squeeze out constant dimensions)
        constant_dim = []
        for i, dim in enumerate(map_config.independent_dimensions):
            unique = np.unique(independent_dimensions[i])
            if unique.size == 1:
                constant_dim.append(i)
        nxentry.independent_dimensions = NXdata()
        for i, dim in enumerate(map_config.independent_dimensions):
            if i not in constant_dim:
                nxentry.independent_dimensions[dim.label] = NXfield(
                    independent_dimensions[i], dim.label,
                    attrs={'units': dim.units,
                           'long_name': f'{dim.label} ({dim.units})',
                           'data_type': dim.data_type,
                           'local_name': dim.name})

        # Set up scalar data NeXus NXdata group
        # (add the constant independent dimensions)
        if all_scalar_data is not None:
            self.logger.debug(
                f'all_scalar_data.shape = {all_scalar_data.shape}\n\n')
        scalar_signals = []
        scalar_data = []
        for i, dim in enumerate(map_config.all_scalar_data):
            scalar_signals.append(dim.label)
            scalar_data.append(NXfield(
                value=all_scalar_data[i],
                units=dim.units,
                attrs={'long_name': f'{dim.label} ({dim.units})',
                       'data_type': dim.data_type,
                       'local_name': dim.name}))
        if (map_config.experiment_type == 'EDD'
                and not placeholder_data is False):
            scalar_signals.append('placeholder_data_used')
            scalar_data.append(NXfield(
                value=all_scalar_data[-1],
                attrs={'description':
                    'Indicates whether placeholder data may be present for'
                    'the corresponding frames of detector data.'}))
        for i, dim in enumerate(deepcopy(map_config.independent_dimensions)):
            if i in constant_dim:
                scalar_signals.append(dim.label)
                scalar_data.append(NXfield(
                    independent_dimensions[i], dim.label,
                    attrs={'units': dim.units,
                           'long_name': f'{dim.label} ({dim.units})',
                           'data_type': dim.data_type,
                           'local_name': dim.name}))
                map_config.all_scalar_data.append(
                    PointByPointScanData(**dim.model_dump()))
                map_config.independent_dimensions.remove(dim)
        if scalar_signals:
            nxentry.scalar_data = NXdata()
            for k, v in zip(scalar_signals, scalar_data):
                nxentry.scalar_data[k] = v
            if 'SCAN_N' in scalar_signals:
                nxentry.scalar_data.attrs['signal'] = 'SCAN_N'
            else:
                nxentry.scalar_data.attrs['signal'] = scalar_signals[0]
            scalar_signals.remove(nxentry.scalar_data.attrs['signal'])
            nxentry.scalar_data.attrs['auxiliary_signals'] = scalar_signals

        # Add detector data
        nxdata = NXdata()
        nxentry.data = nxdata
        nxentry.data.set_default()
        detector_ids = []
        for k, v in map_config.attrs.items():
            nxdata.attrs[k] = v
        min_ = np.min(data, axis=tuple(range(1, data.ndim)))
        max_ = np.max(data, axis=tuple(range(1, data.ndim)))
        for i, detector in enumerate(detector_config.detectors):
            nxdata[detector.id] = NXfield(
                value=data[i],
                attrs={**detector.attrs, 'min': min_[i], 'max': max_[i]})
            detector_ids.append(detector.id)
        linkdims(nxdata, nxentry.independent_dimensions)
        if len(detector_config.detectors) == 1:
            nxdata.attrs['signal'] = detector_config.detectors[0].id
        nxentry.detector_ids = detector_ids

        return nxroot

    def _read_raw_data_edd(
            self, map_config, detector_config, comm, num_scan, offset,
            placeholder_data):
        """Read the raw EDD data for a given map configuration.

        :param map_config: A valid map configuration.
        :type map_config: common.models.map.MapConfig
        :param detector_config: A valid detector configuration.
        :type detector_config: common.models.map.DetectorConfig
        :param comm: MPI communicator.
        :type comm: mpi4py.MPI.Comm, optional
        :param num_scan: Number of scans in the map.
        :type num_scan: int
        :param offset: Offset scan number of current processor.
        :type offset: int
        :param placeholder_data: Value to use for missing detector
            data frames, or `False` if missing data should raise an
            error.
        :type placeholder_data: object
        :return: The map's raw data, independent dimensions and scalar
            data.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        # Third party modules
        try:
            from mpi4py import MPI
            from mpi4py.util import dtlib
        except:
            pass

        # Local modules
        from CHAP.utils.general import list_to_string

        if comm is None:
            num_proc = 1
            rank = 0
        else:
            num_proc = comm.Get_size()
            rank = comm.Get_rank()
        if not rank:
            self.logger.debug(f'Number of processors: {num_proc}')
            self.logger.debug(f'Number of scans: {num_scan}')

        # Create the shared data buffers
        # FIX: just one spec scan at this point
        assert len(map_config.spec_scans) == 1
        scan = map_config.spec_scans[0]
        scan_numbers = scan.scan_numbers
        scanparser = scan.get_scanparser(scan_numbers[0])
        if detector_config.detectors[0].id == 'mca1':
            if len(detector_config.detectors) != 1:
                raise ValueError(
                    'Multiple detectors not implemented for mca1 detector')
            detector_ids = ['mca1']
        else:
            detector_ids = [int(d.id) for d in detector_config.detectors]
        ddata, placeholder_used = scanparser.get_detector_data(
            detector_ids, placeholder_data=placeholder_data)
        spec_scan_shape = scanparser.spec_scan_shape
        num_dim = np.prod(spec_scan_shape)
        num_id = len(map_config.independent_dimensions)
        num_sd = len(map_config.all_scalar_data)
        if placeholder_data is not False:
            num_sd += 1
        if num_proc == 1:
            assert num_scan == len(scan_numbers)
            data = np.empty((num_scan, *ddata.shape), dtype=ddata.dtype)
            independent_dimensions = np.empty(
                (num_id, num_scan*num_dim), dtype=np.float64)
            all_scalar_data = np.empty(
                (num_sd, num_scan*num_dim), dtype=np.float64)
        else:
            self.logger.debug(f'Scan offset on processor {rank}: {offset}')
            self.logger.debug(f'Scan numbers on processor {rank}: '
                              f'{list_to_string(scan_numbers)}')
            datatype = dtlib.from_numpy_dtype(ddata.dtype)
            itemsize = datatype.Get_size()
            if not rank:
                nbytes = num_scan * np.prod(ddata.shape) * itemsize
            else:
                nbytes = 0
            win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
            buf, itemsize = win.Shared_query(0)
            assert itemsize == datatype.Get_size()
            data = np.ndarray(
                buffer=buf, dtype=ddata.dtype, shape=(num_scan, *ddata.shape))
            datatype = dtlib.from_numpy_dtype(np.float64)
            itemsize = datatype.Get_size()
            if not rank:
                nbytes = num_id * num_scan * num_dim * itemsize
            win_id = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
            buf_id, _ = win_id.Shared_query(0)
            independent_dimensions = np.ndarray(
                buffer=buf_id, dtype=np.float64,
                shape=(num_id, num_scan*num_dim))
            if not rank:
                nbytes = num_sd * num_scan * num_dim * itemsize
            win_sd = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
            buf_sd, _ = win_sd.Shared_query(0)
            all_scalar_data = np.ndarray(
                buffer=buf_sd, dtype=np.float64,
                shape=(num_sd, num_scan*num_dim))

        # Read the raw data
        init = True
        for scan in map_config.spec_scans:
            for scan_number in scan.scan_numbers:
                if init:
                    init = False
                else:
                    scanparser = scan.get_scanparser(scan_number)
                    assert spec_scan_shape == scanparser.spec_scan_shape
                    ddata, placeholder_used = scanparser.get_detector_data(
                        detector_ids, placeholder_data=placeholder_data)
                data[offset] = ddata
                spec_scan_motor_mnes = scanparser.spec_scan_motor_mnes
                start_dim = offset * num_dim
                end_dim = start_dim + num_dim
                for i, dim in enumerate(map_config.independent_dimensions):
                    independent_dimensions[i][start_dim:end_dim] = \
                        dim.get_value(
                            scan, scan_number, scan_step_index=-1,
                            relative=False)
                for i, dim in enumerate(map_config.all_scalar_data):
                    all_scalar_data[i][start_dim:end_dim] = dim.get_value(
                        scan, scan_number, scan_step_index=-1,
                        relative=False)
                    if placeholder_data is not False:
                        all_scalar_data[-1][start_dim:end_dim] = \
                            placeholder_used
                offset += 1

        return (
            np.swapaxes(
                data.reshape((np.prod(data.shape[:2]), *data.shape[2:])),
                0, 1),
            independent_dimensions, all_scalar_data)

    def _read_raw_data(
            self, map_config, detector_config, comm, num_scan, offset):
        """Read the raw data for a given map configuration.

        :param map_config: A valid map configuration.
        :type map_config: common.models.map.MapConfig
        :param detector_config: A valid detector configuration.
        :type detector_config: common.models.map.DetectorConfig
        :param comm: MPI communicator.
        :type comm: mpi4py.MPI.Comm, optional
        :param num_scan: Number of scans in the map.
        :type num_scan: int
        :param offset: Offset scan number of current processor.
        :type offset: int
        :return: The map's raw data, independent dimensions and scalar
            data.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        # Third party modules
        try:
            from mpi4py import MPI
            from mpi4py.util import dtlib
        except:
            pass

        # Local modules
        from CHAP.utils.general import list_to_string

        if comm is None:
            num_proc = 1
            rank = 0
        else:
            num_proc = comm.Get_size()
            rank = comm.Get_rank()
        if not rank:
            self.logger.debug(f'Number of processors: {num_proc}')
            self.logger.debug(f'Number of scans: {num_scan}')

        # Create the shared data buffers
        assert len(map_config.spec_scans) == 1
        scans = map_config.spec_scans[0]
        scan_numbers = scans.scan_numbers
        scanparser = scans.get_scanparser(scan_numbers[0])
        ddata = scanparser.get_detector_data(detector_config.detectors[0].id)
        num_det = len(detector_config.detectors)
        num_dim = ddata.shape[0]
        num_id = len(map_config.independent_dimensions)
        num_sd = len(map_config.all_scalar_data)
        if num_proc == 1:
            assert num_scan == len(scan_numbers)
            data = np.empty(
                (num_det, num_scan, *ddata.shape), dtype=ddata.dtype)
            independent_dimensions = np.empty(
                (num_scan, num_id, num_dim), dtype=np.float64)
            if num_sd:
                all_scalar_data = np.empty(
                    (num_scan, num_sd, num_dim), dtype=np.float64)
        else:
            self.logger.debug(f'Scan offset on processor {rank}: {offset}')
            self.logger.debug(f'Scan numbers on processor {rank}: '
                              f'{list_to_string(scan_numbers)}')
            datatype = dtlib.from_numpy_dtype(ddata.dtype)
            itemsize = datatype.Get_size()
            if not rank:
                nbytes = num_scan * np.prod(ddata.shape) * itemsize
            else:
                nbytes = 0
            win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
            buf, _ = win.Shared_query(0)
            data = np.ndarray(
                buffer=buf, dtype=ddata.dtype,
                shape=(num_det, num_scan, *ddata.shape))
            datatype = dtlib.from_numpy_dtype(np.float64)
            itemsize = datatype.Get_size()
            if not rank:
                nbytes = num_scan * num_id * num_dim * itemsize
            else:
                nbytes = 0
            win_id = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
            buf_id, _ = win_id.Shared_query(0)
            independent_dimensions = np.ndarray(
                buffer=buf_id, dtype=np.float64,
                shape=(num_scan, num_id, num_dim))
            if num_sd:
                if not rank:
                    nbytes = num_scan * num_sd * num_dim * itemsize
                win_sd = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
                buf_sd, _ = win_sd.Shared_query(0)
                all_scalar_data = np.ndarray(
                    buffer=buf_sd, dtype=np.float64,
                    shape=(num_scan, num_sd, num_dim))

        # Read the raw data
        init = True
        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                for i, detector in enumerate(detector_config.detectors):
                    if init:
                        init = False
                    else:
                        scanparser = scans.get_scanparser(scan_number)
                        ddata = scanparser.get_detector_data(
                            detector_config.detectors[i].id)
                    data[i][offset] = ddata
                for i, dim in enumerate(map_config.independent_dimensions):
                    if dim.data_type in ['scan_column',
                                         'detector_log_timestamps']:
                        independent_dimensions[offset,i] = dim.get_value(
                        #v = dim.get_value(
                            scans, scan_number, scan_step_index=-1,
                            relative=False)[:num_dim]
                        #print(f'\ndim: {dim}\nv {np.asarray(v).shape}: {v}')
                        #independent_dimensions[offset,i] = v[:num_dim]
                    elif dim.data_type in ['smb_par', 'spec_motor',
                                           'expression']:
                        independent_dimensions[offset,i] = dim.get_value(
                        #v = dim.get_value(
                            scans, scan_number, scan_step_index=-1,
                            relative=False, scalar_data=map_config.scalar_data)
                        #print(f'\ndim: {dim}\nv {np.asarray(v).shape}: {v}')
                        #independent_dimensions[offset,i] = v
                    else:
                        raise RuntimeError(
                            f'{dim.data_type} in data_type not tested')
                for i, dim in enumerate(map_config.all_scalar_data):
                    all_scalar_data[offset,i] = dim.get_value(
                        scans, scan_number, scan_step_index=-1,
                        relative=False)
                offset += 1

        if num_sd:
            return (
                data.reshape(
                    (data.shape[0], np.prod(data.shape[1:3]),
                     *data.shape[3:])),
                np.stack(tuple([independent_dimensions[:,i].flatten()
                                for i in range(num_id)])),
                np.stack(tuple([all_scalar_data[:,i].flatten()
                                for i in range(num_sd)])))
        return (
            data.reshape(
                (data.shape[0], np.prod(data.shape[1:3]), *data.shape[3:])),
            np.stack(tuple([independent_dimensions[:,i].flatten()
                            for i in range(num_id)])),
            None)


class MPICollectProcessor(Processor):
    """A Processor that collects the distributed worker data from
    MPIMapProcessor on the root node.
    """
    def process(self, data, comm, root_as_worker=True):
        """Collect data on root node.

        :param data: Input data.
        :type data: list[PipelineData]
        :param comm: MPI communicator.
        :type comm: mpi4py.MPI.Comm, optional
        :param root_as_worker: Use the root node as a worker,
            defaults to `True`.
        :type root_as_worker: bool, optional
        :return: Returns a list of the distributed worker data on the
            root node.
        """
        num_proc = comm.Get_size()
        rank = comm.Get_rank()
        if root_as_worker:
            data = self.unwrap_pipelinedata(data)[-1]
            if num_proc > 1:
                data = comm.gather(data, root=0)
        else:
            for n_worker in range(1, num_proc):
                if rank == n_worker:
                    comm.send(self.unwrap_pipelinedata(data)[-1], dest=0)
                    data = None
                elif not rank:
                    if n_worker == 1:
                        data = [comm.recv(source=n_worker)]
                    else:
                        data.append(comm.recv(source=n_worker))
        #FIX RV TODO Merge the list of data items in some generic fashion
        return data


class MPIMapProcessor(Processor):
    """A Processor that applies a parallel generic sub-pipeline to 
    a map configuration.
    """
    def process(self, data, config=None, sub_pipeline=None, inputdir=None,
            outputdir=None, interactive=None, log_level=None):
        """Run a parallel generic sub-pipeline.

        :param data: Input data.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            common.models.map.MapConfig.
        :type config: dict, optional
        :param sub_pipeline: The sub-pipeline.
        :type sub_pipeline: Pipeline, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths.
        :type inputdir: str, optional
        :param outputdir: Directory to which any output figures will
            be saved.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions.
        :type interactive: bool, optional
        :ivar log_level: Logger level (not case sensitive).
        :type log_level: Literal[
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], optional
        :return: The `data` field of the first item in the returned
           list of sub-pipeline items.
        """
        # Third party modules
        from mpi4py import MPI

        # Local modules
        from CHAP.runner import (
            RunConfig,
            run,
        )
        from CHAP.common.models.map import SpecScans

        comm = MPI.COMM_WORLD
        num_proc = comm.Get_size()
        rank = comm.Get_rank()

        # Get the validated map configuration
        map_config = self.get_config(
            data=data, config=config, schema='common.models.map.MapConfig',
            inputdir=inputdir)

        # Create the spec reader configuration for each processor
        # FIX: catered to EDD with one spec scan
        assert len(map_config.spec_scans) == 1
        spec_scans = map_config.spec_scans[0]
        scan_numbers = spec_scans.scan_numbers
        num_scan = len(scan_numbers)
        scans_per_proc = num_scan//num_proc
        n_scan = 0
        for n_proc in range(num_proc):
            num = scans_per_proc
            if n_proc == rank:
                if rank < num_scan - scans_per_proc*num_proc:
                    num += 1
                scan_numbers = scan_numbers[n_scan:n_scan+num]
            n_scan += num
        spec_config = {
            'station': map_config.station,
            'experiment_type': map_config.experiment_type,
            'spec_scans': [SpecScans(
                spec_file=spec_scans.spec_file, scan_numbers=scan_numbers)]}

        # Get the run configuration to use for the sub-pipeline
        if sub_pipeline is None:
            sub_pipeline = {}
        run_config = {'inputdir': inputdir, 'outputdir': outputdir,
            'interactive': interactive, 'log_level': log_level}
        run_config.update(sub_pipeline.get('config'))
        run_config = RunConfig(**run_config, comm=comm)
        pipeline_config = []
        for item in sub_pipeline['pipeline']:
            if isinstance(item, dict):
                for k, v in deepcopy(item).items():
                    if k.endswith('Reader'):
                        v['config'] = spec_config
                        item[k] = v
                    if num_proc > 1 and k.endswith('Writer'):
                        r, e = os.path.splitext(v['filename'])
                        v['filename'] = f'{r}_{rank}{e}'
                        item[k] = v
            pipeline_config.append(item)

        # Run the sub-pipeline on each processor
        return run(run_config, pipeline_config, logger=self.logger, comm=comm)


class MPISpawnMapProcessor(Processor):
    """A Processor that applies a parallel generic sub-pipeline to 
    a map configuration by spawning workers processes.
    """
    def process(
            self, data, num_proc=1, root_as_worker=True, collect_on_root=False,
            sub_pipeline=None, inputdir=None, outputdir=None, interactive=None,
            log_level=None):
        """Spawn workers running a parallel generic sub-pipeline.

        :param data: Input data.
        :type data: list[PipelineData]
        :param num_proc: Number of spawned processors, defaults to `1`.
        :type num_proc: int, optional
        :param root_as_worker: Use the root node as a worker,
            defaults to `True`.
        :type root_as_worker: bool, optional
        :param collect_on_root: Collect the result of the spawned
            workers on the root node, defaults to `False`.
        :type collect_on_root: bool, optional
        :param sub_pipeline: The sub-pipeline.
        :type sub_pipeline: Pipeline, optional
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
        :ivar log_level: Logger level (not case sensitive).
        :type log_level: Literal[
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], optional
        :return: The `data` field of the first item in the returned
           list of sub-pipeline items.
        """
        # System modules
        from tempfile import NamedTemporaryFile

        # Third party modules
        try:
            from mpi4py import MPI
        except Exception as exc:
            raise ImportError('Unable to import mpi4py') from exc
        import yaml

        # Local modules
        from CHAP.runner import (
            RunConfig,
            runner,
        )
        from CHAP.common.models.map import SpecScans

        # Get the map configuration from data
        map_config = self.get_config(
            data=data, schema='common.models.map.MapConfig', inputdir=inputdir)

        # Get the run configuration to use for the sub-pipeline
        if sub_pipeline is None:
            sub_pipeline = {}
        run_config = {'inputdir': inputdir, 'outputdir': outputdir,
            'interactive': interactive, 'log_level': log_level}
        run_config.update(sub_pipeline.get('config'))
        run_config = RunConfig(**run_config, logger=self.logger)

        # Create the sub-pipeline configuration for each processor
        spec_scans = map_config.spec_scans[0]
        scan_numbers = spec_scans.scan_numbers
        num_scan = len(scan_numbers)
        scans_per_proc = num_scan//num_proc
        n_scan = 0
        pipeline_config = []
        for n_proc in range(num_proc):
            num = scans_per_proc
            if n_proc < num_scan - scans_per_proc*num_proc:
                num += 1
            spec_config = {
                'station': map_config.station,
                'experiment_type': map_config.experiment_type,
                'spec_scans': [SpecScans(
                    spec_file=spec_scans.spec_file,
                    scan_numbers=scan_numbers[n_scan:n_scan+num]).__dict__]}
            sub_pipeline_config = []
            for item in deepcopy(sub_pipeline['pipeline']):
                if isinstance(item, dict):
                    for k, v in deepcopy(item).items():
                        if k.endswith('Reader'):
                            v['config'] = spec_config
                            item[k] = v
                        if num_proc > 1 and k.endswith('Writer'):
                            r, e = os.path.splitext(v['filename'])
                            v['filename'] = f'{r}_{n_proc}{e}'
                            item[k] = v
                sub_pipeline_config.append(item)
            if collect_on_root and (not root_as_worker or num_proc > 1):
                sub_pipeline_config += [
                    {'common.MPICollectProcessor': {
                        'root_as_worker': root_as_worker}}]
            pipeline_config.append(sub_pipeline_config)
            n_scan += num

        # Optionally include the root node as a worker node
        if root_as_worker:
            first_proc = 1
            run_config.spawn = 1
        else:
            first_proc = 0
            run_config.spawn = -1

        # Spawn the workers to run the sub-pipeline
        if num_proc > first_proc:
            tmp_names = []
            with NamedTemporaryFile(delete=False) as fp:
                fp_name = fp.name
                tmp_names.append(fp_name)
                with open(fp_name, 'w') as f:
                    yaml.dump(
                        {'config': {'spawn': run_config.spawn}}, f,
                        sort_keys=False)
                for n_proc in range(first_proc, num_proc):
                    f_name = f'{fp_name}_{n_proc}'
                    tmp_names.append(f_name)
                    with open(f_name, 'w') as f:
                        yaml.dump(
                            {'config': run_config.model_dump(),
                             'pipeline': pipeline_config[n_proc]},
                            f, sort_keys=False)
                sub_comm = MPI.COMM_SELF.Spawn(
                    'CHAP', args=[fp_name], maxprocs=num_proc-first_proc)
                common_comm = sub_comm.Merge(False)
                if run_config.spawn > 0:
                    # Align with the barrier in RunConfig() on common_comm
                    # called from the spawned main()
                    common_comm.barrier()
        else:
            common_comm = None

        # Run the sub-pipeline on the root node
        if root_as_worker:
            data = runner(run_config, pipeline_config[0], comm=common_comm)
        elif collect_on_root:
            run_config.spawn = 0
            pipeline_config = [{'common.MPICollectProcessor': {
                 'root_as_worker': root_as_worker}}]
            data = runner(run_config, pipeline_config, common_comm)
        else:
            # Align with the barrier in run() on common_comm
            # called from the spawned main()
            common_comm.barrier()
            data = None

        # Disconnect spawned workers and cleanup temporary files
        if num_proc > first_proc:
            # Align with the barrier in main() on common_comm
            # when disconnecting the spawned worker
            common_comm.barrier()
            # Disconnect spawned workers and cleanup temporary files
            sub_comm.Disconnect()
            for tmp_name in tmp_names:
                os.remove(tmp_name)

        return data


class NexusToNumpyProcessor(Processor):
    """A Processor to convert the default plottable data in a NeXus
    object into a `numpy.ndarray`.
    """
    def process(self, data):
        """Return the default plottable data signal in a NeXus object 
        contained in `data` as an `numpy.ndarray`.

        :param data: Input data.
        :type data: nexusformat.nexus.NXobject
        :raises ValueError: If `data` has no default plottable data
            signal.
        :return: The default plottable data signal.
        :rtype: numpy.ndarray
        """
        # Third party modules
        from nexusformat.nexus import NXdata

        data = self.unwrap_pipelinedata(data)[-1]

        if isinstance(data, NXdata):
            default_data = data
        else:
            default_data = data.plottable_data
            if default_data is None:
                default_data_path = data.attrs.get('default')
                default_data = data.get(default_data_path)
            if default_data is None:
                raise ValueError(
                    f'The structure of {data} contains no default data')

        try:
            default_signal = default_data.attrs['signal']
        except:
            raise ValueError(f'The signal of {default_data} is unknown')

        np_data = default_data[default_signal].nxdata

        return np_data


class NexusToXarrayProcessor(Processor):
    """A Processor to convert the default plottable data in a
    NeXus object into an `xarray.DataArray`.
    """
    def process(self, data):
        """Return the default plottable data signal in a NeXus object
        contained in `data` as an `xarray.DataArray`.

        :param data: Input data.
        :type data: nexusformat.nexus.NXobject
        :raises ValueError: If metadata for `xarray` is absent from
            `data`
        :return: The default plottable data signal.
        :rtype: xarray.DataArray
        """
        # Third party modules
        from nexusformat.nexus import NXdata
        from xarray import DataArray

        data = self.unwrap_pipelinedata(data)[-1]

        if isinstance(data, NXdata):
            default_data = data
        else:
            default_data = data.plottable_data
            if default_data is None:
                default_data_path = data.attrs.get('default')
                default_data = data.get(default_data_path)
            if default_data is None:
                raise ValueError(
                    f'The structure of {data} contains no default data')

        try:
            default_signal = default_data.attrs['signal']
        except:
            raise ValueError(f'The signal of {default_data} is unknown')
        signal_data = default_data[default_signal].nxdata

        axes = default_data.attrs['axes']
        if isinstance(axes, str):
            axes = [axes]
        coords = {}
        for axis_name in axes:
            axis = default_data[axis_name]
            coords[axis_name] = (axis_name, axis.nxdata, axis.attrs)

        dims = tuple(axes)
        name = default_signal
        attrs = default_data[default_signal].attrs

        return DataArray(data=signal_data,
                         coords=coords,
                         dims=dims,
                         name=name,
                         attrs=attrs)


class NexusToZarrProcessor(Processor):
    """Converter for NeXus to Zarr format."""
    def process(self, data, chunks='auto'):
        from nexusformat.nexus import (NXfield, NXgroup)
        import zarr
        from zarr.storage import MemoryStore

        nexus_group = self.unwrap_pipelinedata(data)[0]
        zarr_group = zarr.create_group(store=MemoryStore({}))

        def copy_group(nexus_group, zarr_group):
            self.logger.info(f'Copying {nexus_group.nxpath}')
            # Copy attributes
            for attr_key, attr_value in nexus_group.attrs.items():
                if isinstance(attr_value.nxvalue, np.ndarray):
                    zarr_group.attrs[attr_key] = attr_value.nxvalue.tolist()
                else:
                    zarr_group.attrs[attr_key] = attr_value.nxvalue

            # Copy datasets and sub-groups
            for key, item in nexus_group.items():
                if isinstance(item, NXfield):
                    if isinstance(item.nxdata, np.ndarray):
                        try:
                            # Determine chunks
                            if isinstance(chunks, list):
                                if len(chunks) < len(item.nxdata.shape):
                                    _chunks = (
                                        *chunks,
                                        *item.nxdata.shape[len(chunks):]
                                    )
                                elif len(chunks) > len(item.nxdata.shape):
                                    _chunks = 'auto'
                                else:
                                    _chunks = chunks
                            else:
                                _chunks = chunks
                            # Copy dataset
                            zarr_dset = zarr_group.create_array(
                                name=key,
                                shape=item.nxdata.shape,
                                dtype=item.nxdata.dtype,
                                attributes={k: v.nxvalue
                                            for k, v in item.attrs.items()},
                                chunks=_chunks,
                            )
                            self.logger.info(f'Copying {item.nxpath}')
                            zarr_dset[:] = item.nxdata
                        except Exception as e:
                            self.logger.error(f'{item.nxpath}: {e}')
                    else:
                        self.logger.warning(f'Ignoring {item.nxpath}')
                elif isinstance(item, NXgroup):
                    # Recursively copy subgroup
                    zarr_subgroup = zarr_group.create_group(key)
                    copy_group(item, zarr_subgroup)

        copy_group(nexus_group, zarr_group)
        return zarr_group


class NormalizeNexusProcessor(Processor):
    """Processor for scaling one or more NXfields in the input nexus
    structure by the values of another NXfield in the same
    structure."""
    def process(self, data, normalize_nxfields, normalize_by_nxfield):
        """Return copy of the original input nexus structure with
        additional fields containing the normalized data of each field
        in `normalize_nxfields`.

        :param data: Input nexus structure containing all fields to be
            normalized an the field by which to normalize them.
        :type data: nexusformat.nexus.NXgroup
        :param normalize_nxfields:
        :type normalize_nxfields: list[str]
        :param normalize_by_nxfield: Path in `data` to the `NXfield`
            containing normalization data
        :type normalize_by_nxfield: str
        :returns: Copy of input data with additional normalized fields
        :rtype: nexusformat.nexus.NXgroup
        """
        # Third party modules
        from nexusformat.nexus import (
            NXgroup,
            NXfield,
        )

        # Local modules
        from CHAP.utils.general import nxcopy

        # Check input data
        data = self.unwrap_pipelinedata(data)[0]
        data = nxcopy(data)
        if not isinstance(data, NXgroup):
            raise TypeError(f'Expected NXgroup, got (type{data})')

        # Check normalize_by_nxfield
        if normalize_by_nxfield not in data:
            raise ValueError(
                f'{normalize_by_nxfield} not present in input data')
        if not isinstance(data[normalize_by_nxfield], NXfield):
            raise TypeError(
                f'{normalize_by_nxfield} is {type(data[normalize_by_nxfield])}'
                + ', expected NXfield')
        normalization_data = data[normalize_by_nxfield].nxdata

        # Process normalize_nxfields
        for nxfield in normalize_nxfields:
            if nxfield not in data:
                self.logger.error(f'{nxfield} not present in input data')
            elif not isinstance(data[nxfield], NXfield):
                self.logger.error(
                    f'{nxfield} is {type(data[nxfield])}, expected NXfield')
            else:
                field_shape = data[nxfield].nxdata.shape
                if not normalization_data.shape == \
                    field_shape[:normalization_data.ndim]:
                    self.logger.error(
                        f'Incompatible dataset shapes: {normalize_by_nxfield} '
                        + f'is {normalization_data.shape}, '
                        + f'{nxfield} is {field_shape}'
                    )
                else:
                    self.logger.info(f'Normalizing {nxfield}')
                    # make shapes compatible
                    _normalization_data = normalization_data.reshape(
                        normalization_data.shape + (1,)
                        * (data[nxfield].nxdata.ndim
                           - normalization_data.ndim))
                    data[f'{nxfield}_normalized'] = NXfield(
                        value=data[nxfield].nxdata / _normalization_data,
                        attrs={**data[nxfield].attrs,
                               'normalized_by': normalize_by_nxfield}
                    )
        return data


class NormalizeMapProcessor(Processor):
    """Processor for calling `NormalizeNexusProcessor` for (usually
    all) detector data in an `NXroot` resulting from
    `MapProcessor`"""
    def process(self, data, normalize_by_nxfield, detector_ids=None):
        """Return copy of the original input map `NXroot` with
        additional fields containing normalized detector data.

        :param data: Input nexus structure containing all fields to be
            normalized an the field by which to normalize them.
        :type data: nexusformat.nexus.NXroot
        :param normalize_by_nxfield: Path in `data` to the `NXfield`
            containing normalization data
        :type normalize_by_nxfield: str
        :returns: Copy of input data with additional normalized fields
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXlink,
        )

        # Check input data
        data = self.unwrap_pipelinedata(data)[0]
        map_title = None
        for k, v in data.items():
            if isinstance(v, NXentry):
                map_title = k
                break
        if map_title is None:
            self.logger.error(f'Input data contains no NXentry')
        else:
            self.logger.info(f'Got map_title: {map_title}')

        # Check detector_ids
        normalize_nxfields = []
        if detector_ids is None:
            detector_ids = [k for k in data[map_title].data.keys()
                            if not isinstance(data[map_title].data[k], NXlink)]
            self.logger.info(f'Using detector_ids: {detector_ids}')
        normalize_nxfields = [f'{map_title}/data/{_id}'
                              for _id in detector_ids]

        # Normalize
        normalizer = NormalizeNexusProcessor()
        normalizer.logger = self.logger
        return normalizer.process(
            data, normalize_nxfields, normalize_by_nxfield)


class PandasToXarrayProcessor(Processor):
    """Converter for `pandas.DataFrame` to `xarray.DataArray` or
    `xarray.Dataset`"""
    def process(self, data):
        """Return input dataframe converted to xarray.

        :param data: Input `pandas.DataFrame`
        :type data: list[PipelineData]
        :returns: Input dataframe as xarray.
        :rtype: Union[`xarray.DataArray`, `xarray.Dataset`]
        """
        dataframe = self.get_data(data, nxobject=False)
        return dataframe.to_xarray()


class PrintProcessor(Processor):
    """A Processor to simply print the input data to stdout and return
    the original input data, unchanged in any way.
    """
    def process(self, data):
        """Print and return the input data.

        :param data: Input data.
        :type data: object
        :return: `data`
        :rtype: object
        """
        print(f'{self.__name__} data :')
        if callable(getattr(data, '_str_tree', None)):
            # If data is likely a NeXus NXobject, print its tree
            # representation (since NXobjects' str representations are
            # just their nxname)
            print(data._str_tree(attrs=True, recursive=True))
        else:
            print(str(data))

        return data


class PyfaiAzimuthalIntegrationProcessor(Processor):
    """Processor to azimuthally integrate one or more frames of 2d
    detector data using the
    [pyFAI](https://pyfai.readthedocs.io/en/v2023.1/index.html)
    package.
    """
    def process(
            self, data, poni_file, npt, mask_file=None,
            integrate1d_kwargs=None, inputdir='.'):
        """Azimuthally integrate the detector data provided and return
        the result as a dictionary of numpy arrays containing the
        values of the radial coordinate of the result, the intensities
        along the radial direction, and the poisson errors for each
        intensity spectrum.

        :param data: Detector data to integrate.
        :type data: Union[PipelineData, list[np.ndarray]]
        :param poni_file: Name of the [pyFAI PONI file]
            (https://pyfai.readthedocs.io/en/v2023.1/glossary.html?highlight=poni%20file#poni-file)
        containing the detector properties pyFAI needs to perform
        azimuthal integration.
        :type poni_file: str
        :param npt: Number of points in the output pattern.
        :type npt: int
        :param mask_file: A file to use for masking the input data.
        :type mask_file: str, optional
        :param integrate1d_kwargs: Optional dictionary of keyword
            arguments to use with
            [`pyFAI.azimuthalIntegrator.AzimuthalIntegrator.integrate1d`](https://pyfai.readthedocs.io/en/v2023.1/api/pyFAI.html#pyFAI.azimuthalIntegrator.AzimuthalIntegrator.integrate1d).
        :type integrate1d_kwargs: Optional[dict]
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str, optional
        :returns: Azimuthal integration results as a dictionary of
            numpy arrays.
        """
        # Third party modules
        from pyFAI import load

        if not os.path.isabs(poni_file):
            poni_file = os.path.join(inputdir, poni_file)
        ai = load(poni_file)

        if mask_file is None:
            mask = None
        else:
            # Third party modules
            import fabio

            if not os.path.isabs(mask_file):
                mask_file = os.path.join(inputdir, mask_file)
            mask = fabio.open(mask_file).data

        try:
            det_data = self.unwrap_pipelinedata(data)[0]
        except:
            det_data = data

        if integrate1d_kwargs is None:
            integrate1d_kwargs = {}
        integrate1d_kwargs['mask'] = mask

        return [ai.integrate1d(d, npt, **integrate1d_kwargs) for d in det_data]


class RawDetectorDataMapProcessor(Processor):
    """A Processor to return a map of raw detector data in a
    NeXus NXroot object.
    """
    def process(self, data, detector_name, detector_shape):
        """Process configurations for a map and return the raw
        detector data data collected over the map.

        :param data: Input map configuration.
        :type data: list[PipelineData]
        :param detector_name: The detector prefix.
        :type detector_name: str
        :param detector_shape: The shape of detector data for a single
            scan step.
        :type detector_shape: list
        :return: Map of raw detector data.
        :rtype: nexusformat.nexus.NXroot
        """
        map_config = self.get_config()
        nxroot = self.get_nxroot(map_config, detector_name, detector_shape)

        return nxroot

    def get_config(self, data):
        """Get instances of the map configuration object needed by this
        `Processor`.

        :param data: Result of `Reader.read` where at least one item
            has the value `'common.models.map.MapConfig'` for the
            `'schema'` key.
        :type data: list[PipelineData]
        :raises Exception: If a valid map config object cannot be
            constructed from `data`.
        :return: A valid instance of the map configuration object with
            field values taken from `data`.
        :rtype: common.models.map.MapConfig
        """
        # Local modules
        from CHAP.common.models.map import MapConfig

        map_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    schema = item.get('schema')
                    if schema == 'common.models.map.MapConfig':
                        map_config = item.get('data')

        if not map_config:
            raise ValueError('No map configuration found in input data')

        return MapConfig(**map_config)

    def get_nxroot(self, map_config, detector_name, detector_shape):
        """Get a map of the detector data collected by the scans in
        `map_config`. The data will be returned along with some
        relevant metadata in the form of a NeXus structure.

        :param map_config: The map configuration.
        :type map_config: common.models.map.MapConfig
        :param detector_name: The detector prefix.
        :type detector_name: str
        :param detector_shape: The shape of detector data for a single
            scan step.
        :type detector_shape: list
        :return: A map of the raw detector data.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXdetector,
            NXinstrument,
            NXroot,
        )

        raise RuntimeError('Not updated for the new MapProcessor')
        nxroot = NXroot()

        nxroot[map_config.title] = MapProcessor.get_nxentry(map_config)
        nxentry = nxroot[map_config.title]

        nxentry.instrument = NXinstrument()
        nxentry.instrument.detector = NXdetector()

        nxentry.instrument.detector.data = NXdata()
        nxdata = nxentry.instrument.detector.data
        nxdata.raw = np.empty((*map_config.shape, *detector_shape))
        nxdata.raw.attrs['units'] = 'counts'
        for i, det_axis_size in enumerate(detector_shape):
            nxdata[f'detector_axis_{i}_index'] = np.arange(det_axis_size)

        for map_index in np.ndindex(map_config.shape):
            scans, scan_number, scan_step_index = \
                map_config.get_scan_step_index(map_index)
            scanparser = scans.get_scanparser(scan_number)
            self.logger.debug(
                f'Adding data to nxroot for map point {map_index}')
            nxdata.raw[map_index] = scanparser.get_detector_data(
                detector_name,
                scan_step_index)

        nxentry.data.makelink(
            nxdata.raw,
            name=detector_name)
        for i, det_axis_size in enumerate(detector_shape):
            nxentry.data.makelink(
                nxdata[f'detector_axis_{i}_index'],
                name=f'{detector_name}_axis_{i}_index'
            )
            if isinstance(nxentry.data.attrs['axes'], str):
                nxentry.data.attrs['axes'] = [
                    nxentry.data.attrs['axes'],
                    f'{detector_name}_axis_{i}_index']
            else:
                nxentry.data.attrs['axes'] += [
                    f'{detector_name}_axis_{i}_index']

        nxentry.data.attrs['signal'] = detector_name

        return nxroot


class SetupNXdataProcessor(Processor):
    """Processor to set up and return an "empty" NeXus representation
    of a structured dataset. This representation will be an instance
    of a NeXus NXdata object that has:
    1. A NeXus NXfield entry for every coordinate/signal specified.
    1. `nxaxes` that are the NeXus NXfield entries for the coordinates
       and contain the values provided for each coordinate.
    1. NeXus NXfield entries of appropriate shape, but containing all
       zeros, for every signal.
    1. Attributes that define the axes, plus any additional attributes
       specified by the user.

    This `Processor` is most useful as a "setup" step for
    constucting a representation of / container for a complete dataset
    that will be filled out in pieces later by
    `UpdateNXdataProcessor`.

    Examples of use in a `Pipeline` configuration:
    - With inputs from a previous `PipelineItem` specifically written
      to provide inputs to this `Processor`:
      ```yaml
      config:
        inputdir: /rawdata/samplename
        outputdir: /reduceddata/samplename
      pipeline:
        - edd.SetupNXdataReader:
            filename: SpecInput.txt
            dataset_id: 1
        - common.SetupNXdataProcessor:
            nxname: samplename_dataset_1
        - common.NexusWriter:
            filename: data.nxs
      ```
     - With inputs provided directly though the optional arguments:
       ```yaml
      config:
        outputdir: /reduceddata/samplename
      pipeline:
        - common.SetupNXdataProcessor:
            nxname: your_dataset_name
            coords:
              - name: x
                values: [0.0, 0.5, 1.0]
                attrs:
                  units: mm
                  yourkey: yourvalue
              - name: temperature
                values: [200, 250, 275]
                attrs:
                  units: Celsius
                  yourotherkey: yourothervalue
            signals:
              - name: raw_detector_data
                shape: [407, 487]
                attrs:
                  local_name: PIL11
                  foo: bar
              - name: presample_intensity
                shape: []
                attrs:
                   local_name: a3ic0
                   zebra: fish
            attrs:
              arbitrary: metadata
              from: users
              goes: here
        - common.NexusWriter:
            filename: data.nxs
       ```
    """
    def process(
            self, data, nxname='data', coords=None, signals=None, attrs=None,
            data_points=None, extra_nxfields=None, duplicates='overwrite'):
        """Return a NeXus NXdata object that has the requisite axes
        and NeXus NXfield entries to represent a structured dataset
        with the properties provided. Properties may be provided either
        through the `data` argument (from an appropriate `PipelineItem`
        that immediately preceeds this one in a `Pipeline`), or through
        the `coords`, `signals`, `attrs`, and/or `data_points`
        arguments. If any of the latter are used, their values will
        completely override any values for these parameters found from
        `data.`

        :param data: Data from the previous item in a `Pipeline`.
        :type data: list[PipelineData]
        :param nxname: Name for the returned NeXus NXdata object,
            defaults to `'data'`.
        :type nxname: str, optional
        :param coords: List of dictionaries defining the coordinates
            of the dataset. Each dictionary must have the keys
            `'name'` and `'values'`, whose values are the name of the
            coordinate axis (a string) and all the unique values of
            that coordinate for the structured dataset (a list of
            numbers), respectively. A third item in the dictionary is
            optional, but highly recommended: `'attrs'` may provide a
            dictionary of attributes to attach to the coordinate axis
            that assist in in interpreting the returned NeXus NXdata
            representation of the dataset. It is strongly recommended
            to provide the units of the values along an axis in the
            `attrs` dictionary.
        :type coords: list[dict[str, object]], optional
        :param signals: List of dictionaries defining the signals of
            the dataset. Each dictionary must have the keys `'name'`
            and `'shape'`, whose values are the name of the signal
            field (a string) and the shape of the signal's value at
            each point in the dataset (a list of zero or more
            integers), respectively. A third item in the dictionary is
            optional, but highly recommended: `'attrs'` may provide a
            dictionary of attributes to attach to the signal fieldthat
            assist in in interpreting the returned NeXus NXdata
            representation of the dataset. It is strongly recommended
            to provide the units of the signal's values `attrs`
            dictionary.
        :type signals: list[dict[str, object]], optional
        :param attrs: An arbitrary dictionary of attributes to assign
            to the returned NeXus NXdata object.
        :type attrs: dict[str, object], optional
        :param data_points: A list of data points to partially (or
            even entirely) fil out the "empty" signal NeXus NXfield's
            before returning the NeXus NXdata object.
        :type data_points: list[dict[str, object]], optional
        :param extra_nxfields: List "extra" NeXus NXfields to include
            that can be described neither as a signal of the dataset,
            not a dedicated coordinate. This paramteter is good for
            including "alternate" values for one of the coordinate
            dimensions -- the same coordinate axis expressed in
            different units, for instance. Each item in the list should
            be a dictionary of parameters for the
            `nexusformat.nexus.NXfield` constructor.
        :type extra_nxfields: list[dict[str, object]], optional
        :param duplicates: Behavior to use if any new data points occur
            at the same point in the dataset's coordinate space as an
            existing data point. Allowed values for `duplicates` are:
            `'overwrite'` and `'block'`. Defaults to `'overwrite'`.
        :type duplicates: Literal['overwrite', 'block']
        :returns: A NeXus NXdata object that represents the structured
            dataset as specified.
        :rtype: nexusformat.nexus.NXdata
        """
        self.nxname = nxname

        if coords is None:
            coords = []
        if signals is None:
            signals = []
        if attrs is None:
            attrs = {}
        if extra_nxfields is None:
            extra_nxfields = []
        self.coords = coords
        self.signals = signals
        self.attrs = attrs
        self.data_points = data_points
        try:
            setup_params = self.unwrap_pipelinedata(data)[0]
        except:
            setup_params = None
        if isinstance(setup_params, dict):
            for a in ('coords', 'signals', 'attrs', 'data_points'):
                setup_param = setup_params.get(a)
                if not getattr(self, a) and setup_param is not None:
                    self.logger.info(f'Using input data from pipeline for {a}')
                    setattr(self, a, setup_param)
                else:
                    self.logger.info(
                        f'Ignoring input data from pipeline for {a}')
        else:
            self.logger.warning('Ignoring all input data from pipeline')
        self.shape = tuple(len(c['values']) for c in self.coords)
        self.extra_nxfields = extra_nxfields
        self.duplicates = duplicates
        self.init_nxdata()

        if self.data_points is not None:
            for d in self.data_points:
                self.add_data_point(d)

        return self.nxdata

    def add_data_point(self, data_point):
        """Add a data point to this dataset.
        1. Validate `data_point`.
        2. Append `data_point` to `self.data_points`.
        3. Update signal `NXfield`s in `self.nxdata`.

        :param data_point: Data point defining a point in the
            dataset's coordinate space and the new signal values at
            that point.
        :type data_point: dict[str, object]
        :returns: None
        """
        self.logger.info(
            f'Adding data point no. {data_point["dataset_point_index"]+1} of '
            f'{len(self.data_points)}')
        self.logger.debug(f'New data point: {data_point}')
        valid, msg = self.validate_data_point(data_point)
        if not valid:
            self.logger.error(f'Cannot add data point: {msg}')
        else:
            self.update_nxdata(data_point)

    def validate_data_point(self, data_point):
        """Return `True` if `data_point` occurs at a valid point in
        this structured dataset's coordinate space, `False`
        otherwise. Also validate shapes of signal values and add NaN
        values for any missing signals.

        :param data_point: Data point defining a point in the
            dataset's coordinate space and the new signal values at
            that point.
        :type data_point: dict[str, object]
        :returns: Validity of `data_point`, message
        :rtype: bool, str
        """
        valid = True
        msg = ''
        # Convert all values to numpy types
        data_point = {k: np.asarray(v) for k, v in data_point.items()}
        # Ensure data_point defines a specific point in the dataset's
        # coordinate space
        if not all(c['name'] in data_point for c in self.coords):
            valid = False
            msg = 'Missing coordinate values'
        # Ensure a value is present for all signals
        for s in self.signals:
            name = s['name']
            if name not in data_point:
                data_point[name] = np.full(s['shape'], 0)
            else:
                if not data_point[name].shape == tuple(s['shape']):
                    valid = False
                    msg = f'Shape mismatch for signal {s}'
        return valid, msg

    def init_nxdata(self):
        """Initialize an empty NeXus NXdata representing this dataset
        to `self.nxdata`; values for axes' `NXfield`s are filled out,
        values for signals' `NXfield`s are empty an can be filled out
        later. Save the empty NeXus NXdata object to the NeXus file.
        Initialise `self.nxfile` and `self.nxdata_path` with the
        `NXFile` object and actual nxpath used to save and make updates
        to the Nexus NXdata object.
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        axes = tuple(NXfield(
            value=c['values'],
            name=c['name'],
            attrs=c.get('attrs'),
            dtype=c.get('dtype', 'float64')) for c in self.coords)
        entries = {s['name']: NXfield(
            value=np.full((*self.shape, *s['shape']), 0),
            name=s['name'],
            attrs=s.get('attrs'),
            dtype=s.get('dtype', 'float64')) for s in self.signals}
        extra_nxfields = [NXfield(**params) for params in self.extra_nxfields]
        extra_nxfields = {f.nxname: f for f in extra_nxfields}
        entries.update(extra_nxfields)
        self.nxdata = NXdata(
            name=self.nxname, axes=axes, entries=entries, attrs=self.attrs)

    def update_nxdata(self, data_point):
        """Update `self.nxdata`'s NXfield values.

        :param data_point: Data point defining a point in the
            dataset's coordinate space and the new signal values at
            that point.
        :type data_point: dict[str, object]
        :returns: None
        """
        index = self.get_index(data_point)
        for s in self.signals:
            name = s['name']
            if name in data_point:
                self.nxdata[name][index] = data_point[name]

    def get_index(self, data_point):
        """Return a tuple representing the array index of `data_point`
        in the coordinate space of the dataset.

        :param data_point: Data point defining a point in the
            dataset's coordinate space.
        :type data_point: dict[str, object]
        :returns: Multi-dimensional index of `data_point` in the
            dataset's coordinate space.
        :rtype: tuple
        """
        return tuple(c['values'].index(data_point[c['name']])
                     for c in self.coords)


class UnstructuredToStructuredProcessor(Processor):
    """Processor to reshape data in an NXdata from an "unstructured"
    to a "structured" representation.
    """
    def process(self, data, nxpath=None):
        # Third party modules
        from nexusformat.nexus import NXdata

        try:
            nxobject = self.get_data(data)
        except:
            nxobject = self.unwrap_pipelinedata(data)[0]
        if isinstance(nxobject, NXdata):
            return self.convert_nxdata(nxobject)
        elif nxpath is not None:
            # Local modules
#            from CHAP.utils.general import nxcopy
            try:
                nxobject = nxobject[nxpath]
            except:
                raise ValueError(f'Invalid parameter nxpath ({nxpath})')
        else:
            raise ValueError(f'Invalid input data ({data})')
        return self.convert_nxdata(nxobject)

    def convert_nxdata(self, nxdata):
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        # Local modules
        from CHAP.edd.processor import get_axes

        # Extract axes from the NXdata attributes
        axes = get_axes(nxdata)
        for a in axes:
            if a not in nxdata:
                raise ValueError(f'Missing coordinates for {a}')

        # Check the independent dimensions and axes
        unstructured_axes = []
        unstructured_dim = None
        for a in axes:
            if not isinstance(nxdata[a], NXfield):
                raise ValueError(
                    f'Invalid axis field type ({type(nxdata[a])})')
            if len(nxdata[a].shape) == 1:
                if not unstructured_axes:
                    unstructured_axes.append(a)
                    unstructured_dim = nxdata[a].size
                else:
                    if nxdata[a].size == unstructured_dim:
                        unstructured_axes.append(a)
                    elif 'unstructured_axes' in nxdata.attrs:
                        raise ValueError(f'Inconsistent axes dimensions')
            elif 'unstructured_axes' in nxdata.attrs:
                raise ValueError(
                    f'Invalid unstructered axis shape ({nxdata[a].shape})')
        if not axes and hasattr(nxdata, 'signal'):
            if len(nxdata[nxdata.signal].shape) < 2:
                raise ValueError(
                    f'Invalid signal shape ({nxdata[nxdata.signal].shape})')
            unstructured_dim = nxdata[nxdata.signal].shape[0]
            for k, v in nxdata.items():
                if (isinstance(v, NXfield) and len(v.shape) == 1
                        and v.shape[0] == unstructured_dim):
                    unstructured_axes.append(k)
        if unstructured_dim is None:
            raise ValueError(f'Unable to determine the unstructered axes')
        axes = unstructured_axes

        # Identify unique coordinate points for each axis
        unique_coords = {}
        coords = {}
        axes_attrs = {}
        for a in axes:
            coords[a] = nxdata[a].nxdata
            unique_coords[a] = np.sort(np.unique(nxdata[a].nxdata))
            axes_attrs[a] = deepcopy(nxdata[a].attrs)
            if 'target' in axes_attrs[a]:
                del axes_attrs[a]['target']

        # Calculate the total number of unique coordinate points
        unique_npts = np.prod([len(v) for k, v in unique_coords.items()])
        if unique_npts != unstructured_dim:
            self.logger.warning('The unstructered grid does not fully map to '
                                'a structered one (there are missing points)')

        # Identify the signals and the data point axes
        signals = []
        data_point_axes = []
        data_point_shape = []
        if hasattr(nxdata, 'signal'):
            if (len(nxdata[nxdata.signal].shape) < 2
                    or nxdata[nxdata.signal].shape[0] != unstructured_dim):
                raise ValueError(
                    f'Invalid signal shape ({nxdata[nxdata.signal].shape})')
            signals = [nxdata.signal]
            data_point_shape = [nxdata[nxdata.signal].shape[1:]]
        for k, v in nxdata.items():
            if (isinstance(v, NXfield) and k not in axes and k not in signals
                    and v.shape[0] == unstructured_dim):
                signals.append(k)
                if not data_point_shape:
                    data_point_shape.append(v.shape[1:])
        if len(data_point_shape) == 1:
            data_point_shape = data_point_shape[0]
        else:
            data_point_shape = []
        for dim in data_point_shape:
            for k, v in nxdata.items():
                if (isinstance(v, NXfield) and k not in axes
                        and v.shape == data_point_shape):
                    data_point_axes.append(k)

        # Create the structured NXdata object
        structured_shape = tuple(len(unique_coords[a]) for a in axes)
        attrs = deepcopy(nxdata.attrs)
        if 'unstructured_axes' in attrs:
            attrs.pop('unstructured_axes')
        attrs['axes'] = axes
        nxdata_structured = NXdata(
            name=f'{nxdata.nxname}_structured',
            **{a: NXfield(
                value=unique_coords[a],
                attrs=axes_attrs[a])
               for a in axes},
            **{s: NXfield(
#                value=np.reshape( # FIX not always a sound way to reshape.
#                    nxdata[s], (*structured_shape, *nxdata[s].shape[1:])),
                 dtype=nxdata[s].dtype,
                 shape=(*structured_shape, *nxdata[s].shape[1:]),
                attrs=nxdata[s].attrs)
               for s in signals},
            attrs=attrs)
        if len(data_point_axes) == 1:
            axes = nxdata_structured.attrs['axes']
            if isinstance(axes, str):
                axes = [axes]
            nxdata_structured.attrs['axes'] = axes + data_point_axes
        for a in data_point_axes:
            nxdata_structured[a] = NXfield(
                value=nxdata[a], attrs=nxdata[a].attrs)

        # Populate the structured NXdata object with values
        for i, coord in enumerate(zip(*tuple(nxdata[a].nxdata for a in axes))):
            structured_index = tuple(
                np.asarray(
                    coord[ii] == unique_coords[axes[ii]]).nonzero()[0][0]
                for ii in range(len(axes)))
            for s in signals:
                nxdata_structured[s][structured_index] = nxdata[s][i]

        return nxdata_structured


class UpdateNXvalueProcessor(Processor):
    """Processor to fill in part(s) of a NeXus object representing a
    structured dataset that's already been written to a NeXus file.

    This Processor is most useful as an "update" step for a NeXus
    NXdata object created by `common.SetupNXdataProcessor`, and is
    most easy to use in a `Pipeline` immediately after another
    `PipelineItem` designed specifically to return a value that can
    be used as input to this `Processor`.

    Example of use in a `Pipeline` configuration:
    ```yaml
    config:
      inputdir: /rawdata/samplename
    pipeline:
      - edd.UpdateNXdataReader:
          spec_file: spec.log
          scan_number: 1
      - common.UpdateNXvalueProcessor:
          nxfilename: /reduceddata/samplename/data.nxs
    ```
    """
    def process(self, data, nxfilename, data_points=None, inputdir=None):
        """Write new data values to an existing NeXus object
        representing an unstructured dataset in a NeXus file.
        Return the list of data points used to update the dataset.

        :param data: Data from the previous item in a `Pipeline`. May
            contain a list of data points that will extend the list of
            data points optionally provided with the `data_points`
            argument.
        :type data: list[PipelineData]
        :param nxfilename: Name of the NeXus file containing the
            NeXus object to update.
        :type nxfilename: str
        :param data_points: List of data points, each one a dictionary
            whose keys are the names of the nxpath, the index of the
            data point in the dataset, and the data value.
        :type data_points: Optional[list[dict[str, object]]]
        :param inputdir: Input directory, used only if `nxfilename` is
            not an absolute paths.
        :type inputdir: str, optional
        :returns: Complete list of data points used to update the
            dataset.
        :rtype: list[dict[str, object]]
        """
        # Third party modules
        from nexusformat.nexus import NXFile

        # Local modules
        from CHAP.utils.general import list_to_string

        if data_points is None:
            data_points = []
        self.logger.debug(f'Got {len(data_points)} data points from keyword')
        ddata_points = self.unwrap_pipelinedata(data)[0]
        if isinstance(ddata_points, list):
            self.logger.debug(f'Got {len(ddata_points)} from pipeline data')
            data_points.extend(ddata_points)
        self.logger.info(f'Updating a total of {len(data_points)} data points')

        if inputdir is not None and not os.path.isabs(nxfilename):
            nxfilename = os.path.join(inputdir, nxfilename)
        nxfile = NXFile(nxfilename, 'rw')

        indices = []
        for data_point in data_points:
            try:
                nxfile.writevalue(
                    data_point['nxpath'], np.asarray(data_point['value']),
                    data_point['index'])
                indices.append(data_point['index'])
            except Exception as exc:
                self.logger.error(f'Error updating {data_point["nxpath"]} for '
                                  f'data point {data_point["index"]}: {exc}')
            else:
                self.logger.debug(f'Updated data point {data_point}')

        nxfile.close()

        return data_points


class UpdateNXdataProcessor(Processor):
    """Processor to fill in part(s) of a NeXus NXdata representing a
    structured dataset that's already been written to a NeXus file.

    This Processor is most useful as an "update" step for a NeXus
    NXdata object created by `common.SetupNXdataProcessor`, and is
    most easy to use in a `Pipeline` immediately after another
    `PipelineItem` designed specifically to return a value that can
    be used as input to this `Processor`.

    Example of use in a `Pipeline` configuration:
    ```yaml
    config:
      inputdir: /rawdata/samplename
    pipeline:
      - edd.UpdateNXdataReader:
          spec_file: spec.log
          scan_number: 1
      - common.UpdateNXdataProcessor:
          nxfilename: /reduceddata/samplename/data.nxs
          nxdata_path: /entry/samplename_dataset_1
    ```
    """
    def process(
            self, data, nxfilename, nxdata_path, data_points=None,
            allow_approximate_coordinates=False, inputdir=None):
        """Write new data points to the signal fields of an existing
        NeXus NXdata object representing a structued dataset in a NeXus
        file. Return the list of data points used to update the
        dataset.

        :param data: Data from the previous item in a `Pipeline`. May
            contain a list of data points that will extend the list of
            data points optionally provided with the `data_points`
            argument.
        :type data: list[PipelineData]
        :param nxfilename: Name of the NeXus file containing the
            NeXus NXdata object to update.
        :type nxfilename: str
        :param nxdata_path: The path to the NeXus NXdata object to
            update in the file.
        :type nxdata_path: str
        :param data_points: List of data points, each one a dictionary
            whose keys are the names of the coordinates and axes, and
            whose values are the values of each coordinate / signal at
            a single point in the dataset. Deafults to None.
        :type data_points: Optional[list[dict[str, object]]]
        :param allow_approximate_coordinates: Parameter to allow the
            nearest existing match for the new data points'
            coordinates to be used if an exact match connot be found
            (sometimes this is due simply to differences in rounding
            convetions). Defaults to False.
        :type allow_approximate_coordinates: bool, optional
        :returns: Complete list of data points used to update the dataset.
        :rtype: list[dict[str, object]]
        """
        # Third party modules
        from nexusformat.nexus import NXFile

        if data_points is None:
            data_points = []
        self.logger.debug(f'Got {len(data_points)} data points from keyword')
        _data_points = self.unwrap_pipelinedata(data)[0]
        if isinstance(_data_points, list):
            self.logger.debug(f'Got {len(_data_points)} from pipeline data')
            data_points.extend(_data_points)
        self.logger.info(f'Updating {len(data_points)} data points total')

        if inputdir is not None and not os.path.isabs(nxfilename):
            nxfilename = os.path.join(inputdir, nxfilename)
        nxfile = NXFile(nxfilename, 'rw')
        nxdata = nxfile.readfile()[nxdata_path]
        axes_names = [a.nxname for a in nxdata.nxaxes]

        data_points_used = []
        for i, d in enumerate(data_points):
            # Verify that the data point contains a value for all
            # coordinates in the dataset.
            if not all(a in d for a in axes_names):
                self.logger.error(
                    f'Data point {i} is missing a value for at least one '
                    f'axis. Skipping. Axes are: {", ".join(axes_names)}')
                continue
            self.logger.debug(
                f'Coordinates for data point {i}: ' +
                ', '.join([f'{a}={d[a]}' for a in axes_names]))
            # Get the index of the data point in the dataset based on
            # its values for each coordinate.
            try:
                index = tuple(np.where(a.nxdata == d[a.nxname])[0][0]
                              for a in nxdata.nxaxes)
            except:
                if allow_approximate_coordinates:
                    try:
                        index = tuple(
                            np.argmin(np.abs(a.nxdata - d[a.nxname]))
                            for a in nxdata.nxaxes)
                        self.logger.warning(
                            f'Nearest match for coordinates of data point {i}:'
                            ', '.join(
                                [f'{a.nxname}={a[_i]}'
                                 for _i, a in zip(index, nxdata.nxaxes)]))
                    except:
                        self.logger.error(
                            f'Cannot get the index of data point {i}. '
                            'Skipping.')
                        continue
                else:
                    self.logger.error(
                        f'Cannot get the index of data point {i}. Skipping.')
                    continue
            self.logger.debug(f'Index of data point {i}: {index}')
            # Update the signals contained in this data point at the
            # proper index in the dataset's singal `NXfield`s
            for k, v in d.items():
                if k in axes_names:
                    continue
                try:
                    nxfile.writevalue(
                        os.path.join(nxdata_path, k), np.asarray(v), index)
#                    self.logger.debug(
#                        f'Wrote to {os.path.join(nxdata_path, k)} in '
#                        f'{nxfilename} at index {index} value: {np.asarray(v)}'
#                        f' (type: {type(v)})')
                except Exception as exc:
                    self.logger.error(
                        f'Error updating signal {k} for new data point '
                        f'{i} (dataset index {index}): {exc}')
            data_points_used.append(d)

        nxfile.close()

        return data_points_used


class NXdataToDataPointsProcessor(Processor):
    """Transform a NeXus NXdata object into a list of dictionaries.
    Each dictionary represents a single data point in the coordinate
    space of the dataset. The keys are the names of the signals and
    axes in the dataset, and the values are a single scalar value (in
    the case of axes) or the value of the signal at that point in the
    coordinate space of the dataset (in the case of signals -- this
    means that values for signals may be any shape, depending on the
    shape of the signal itself).

    Example of use in a pipeline configuration:
    ```yaml
    config:
      inputdir: /reduceddata/samplename
    - common.NXdataReader:
        name: data
        axes_names:
          - x
          - y
        signal_name: z
        nxfield_params:
          - filename: data.nxs
            nxpath: entry/data/x
            slice_params:
              - step: 2
          - filename: data.nxs
            nxpath: entry/data/y
            slice_params:
              - step: 2
          - filename: data.nxs
            nxpath: entry/data/z
            slice_params:
              - step: 2
              - step: 2
    - common.NXdataToDataPointsProcessor
    - common.UpdateNXdataProcessor:
        nxfilename: /reduceddata/samplename/sparsedata.nxs
        nxdata_path: /entry/data
    ```
    """
    def process(self, data):
        """Return a list of dictionaries representing the coordinate
        and signal values at every point in the dataset provided.

        :param data: Input pipeline data containing a NeXus NXdata
            object.
        :type data: list[PipelineData]
        :returns: List of all data points in the dataset.
        :rtype: list[dict[str,object]]
        """
        nxdata = self.unwrap_pipelinedata(data)[0]

        data_points = []
        axes_names = [a.nxname for a in nxdata.nxaxes]
        self.logger.info(f'Dataset axes: {axes_names}')
        dataset_shape = tuple([a.size for a in nxdata.nxaxes])
        self.logger.info(f'Dataset shape: {dataset_shape}')
        signal_names = [k for k, v in nxdata.entries.items()
                        if not k in axes_names \
                        and v.shape[:len(dataset_shape)] == dataset_shape]
        self.logger.info(f'Dataset signals: {signal_names}')
        other_fields = [k for k, v in nxdata.entries.items()
                        if not k in axes_names + signal_names]
        if len(other_fields) > 0:
            self.logger.warning(
                'Ignoring the following fields that cannot be interpreted as '
                f'either dataset coordinates or signals: {other_fields}')
        for i in np.ndindex(dataset_shape):
            data_points.append({
                **{a: nxdata[a][_i] for a, _i in zip(axes_names, i)},
                **{s: nxdata[s].nxdata[i] for s in signal_names},
            })
        return data_points


class XarrayToNexusProcessor(Processor):
    """A Processor to convert the data in an `xarray` structure to a
    NeXus NXdata object.
    """
    def process(self, data):
        """Return `data` represented as a NeXus NXdata object.

        :param data: The input `xarray` structure.
        :type data: typing.Union[xarray.DataArray, xarray.Dataset]
        :return: The data and metadata in `data`.
        :rtype: nexusformat.nexus.NXdata
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        data = self.unwrap_pipelinedata(data)[-1]
        signal = NXfield(value=data.data, name=data.name, attrs=data.attrs)
        axes = []
        for name, coord in data.coords.items():
            axes.append(
                NXfield(value=coord.data, name=name, attrs=coord.attrs))
        axes = tuple(axes)

        return NXdata(signal=signal, axes=axes)


class XarrayToNumpyProcessor(Processor):
    """A Processor to convert the data in an `xarray.DataArray`
    structure to an `numpy.ndarray`.
    """
    def process(self, data):
        """Return just the signal values contained in `data`.

        :param data: The input `xarray.DataArray`.
        :type data: xarray.DataArray
        :return: The data in `data`.
        :rtype: numpy.ndarray
        """
        return self.unwrap_pipelinedata(data)[-1].data


class ZarrToNexusProcessor(Processor):
    """Processor for converting .zarr data to .nxs format."""
    def process(self, data, zarr_filename, nexus_filename):
        import zarr
        import h5py

        # Open the Zarr file
        zarr_file = zarr.open(zarr_filename, mode='r')

        # Create the Nexus file
        with h5py.File(nexus_filename, 'w') as nexus_file:
            # Recursively copy all datasets and attributes
            def copy_group(zarr_group, nexus_group):
                self.logger.info(f'Copying {zarr_group.path}')
                # Copy attributes
                for attr_key, attr_value in zarr_group.attrs.items():
                    nexus_group.attrs[attr_key] = attr_value

                # Copy datasets and sub-groups
                for key, item in zarr_group.members():
                    if isinstance(item, zarr.Array):
                        self.logger.info(f'Copying {zarr_group.path}/{key}')
                        # Copy dataset
                        nexus_dset = nexus_group.create_dataset(
                            name=key,
                            data=item.__array__(),
                            # chunks=item.chunks, # FIXME
                            compression='gzip',
                            compression_opts=4  # GZIP compression level
                        )
                        # Copy dataset attributes
                        for attr_key, attr_value in item.attrs.items():
                            nexus_dset.attrs[attr_key] = attr_value
                    elif isinstance(item, zarr.Group):
                        # Recursively copy subgroup
                        nexus_subgroup = nexus_group.create_group(key)
                        copy_group(item, nexus_subgroup)

            # Start copying from the root group
            copy_group(zarr_file, nexus_file)
        return None


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
