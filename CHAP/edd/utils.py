"""Utility functions for EDD workflows"""

# System modules
from copy import deepcopy

# Third party modules
import numpy as np

def get_peak_locations(ds, tth):
    """Return the peak locations for a given set of lattice spacings
    and 2&theta value.

    :param ds: A set of lattice spacings in angstroms.
    :type ds: list[float]
    :param tth: Diffraction angle 2&theta.
    :type tth: float
    :return: The peak locations in keV.
    :rtype: numpy.ndarray
    """
    # Third party modules
    from scipy.constants import physical_constants

    hc = 1e7 * physical_constants['Planck constant in eV/Hz'][0] \
         * physical_constants['speed of light in vacuum'][0]

    return hc / (2. * ds * np.sin(0.5 * np.radians(tth)))

def make_material(name, sgnum, lattice_parameters, dmin=0.6):
    """Return a hexrd.material.Material with the given properties.

    :param name: Material name.
    :type name: str
    :param sgnum: Space group of the material.
    :type sgnum: int
    :param lattice_parameters: The material's lattice parameters
        ([a, b, c, &#945;, &#946;, &#947;], or fewer as the symmetry of
        the space group allows --- for instance, a cubic lattice with
        space group number 225 can just provide [a, ]).
    :type lattice_parameters: list[float]
    :param dmin: Materials's dmin value in angstroms (&#8491;),
        defaults to `0.6`.
    :type dmin: float, optional
    :return: A hexrd material.
    :rtype: heard.material.Material
    """
    # Third party modules
    from hexrd.material import Material    
    from hexrd.valunits import valWUnit

    material = Material()
    material.name = name
    material.sgnum = sgnum
    if isinstance(lattice_parameters, float):
        lattice_parameters = [lattice_parameters]
    material.latticeParameters = lattice_parameters
    material.dmin = valWUnit('lp', 'length',  dmin, 'angstrom')
    nhkls = len(material.planeData.exclusions)
    material.planeData.set_exclusions(np.zeros(nhkls, dtype=bool))

    return material

def get_unique_hkls_ds(materials, tth_tol=None, tth_max=None, round_sig=8):
    """Return the unique HKLs and lattice spacings for the given list
    of materials.

    :param materials: Materials to get HKLs and lattice spacings for.
    :type materials: list[hexrd.material.Material]
    :param tth_tol: Minimum resolvable difference in 2&theta between
        two unique HKL peaks.
    :type tth_tol: float, optional
    :param tth_max: Detector rotation about hutch x axis.
    :type tth_max: float, optional
    :param round_sig: The number of significant figures in the unique
        lattice spacings, defaults to `8`.
    :type round_sig: int, optional
    :return: Unique HKLs, unique lattice spacings.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Local modules
    from CHAP.edd.models import MaterialConfig

    _materials = deepcopy(materials)
    for i, m in enumerate(materials):
        if isinstance(m, MaterialConfig):
            _materials[i] = m._material
    hkls = np.empty((0,3))
    ds = np.empty((0))
    ds_index = np.empty((0))
    for i, material in enumerate(_materials):
        plane_data = material.planeData
        if tth_tol is not None:
            plane_data.tThWidth = np.radians(tth_tol)
        if tth_max is not None:
            plane_data.exclusions = None
            plane_data.tThMax = np.radians(tth_max)
        hkls = np.vstack((hkls, plane_data.hkls.T))
        ds_i = plane_data.getPlaneSpacings()
        ds = np.hstack((ds, ds_i))
        ds_index = np.hstack((ds_index, i*np.ones(len(ds_i))))
    # Sort lattice spacings in reverse order (use -)
    ds_unique, ds_index_unique, _ = np.unique(
        -ds.round(round_sig), return_index=True, return_counts=True)
    ds_unique = np.abs(ds_unique)
    # Limit the list to unique lattice spacings
    hkls_unique = hkls[ds_index_unique,:].astype(int)
    ds_unique = ds[ds_index_unique]

    return hkls_unique, ds_unique

def select_tth_initial_guess(x, y, hkls, ds, tth_initial_guess=5.0,
        interactive=False):
    """Show a matplotlib figure of a reference MCA spectrum on top of
    HKL locations. The figure includes an input field to adjust the
    initial 2&theta guess and responds by updating the HKL locations
    based on the adjusted value of the initial 2&theta guess.

    :param x: MCA channel energies.
    :type x: np.ndarray
    :param y: MCA intensities.
    :type y: np.ndarray
    :param hkls: List of unique HKL indices to fit peaks for in the
        calibration routine.
    :type fit_hkls: Union(numpy.ndarray, list[list[int, int,int]])
    :param ds: Lattice spacings in angstroms associated with the
        unique HKL indices.
    :type ds: Union(numpy.ndarray, list[float])
    :ivar tth_initial_guess: Initial guess for 2&theta,
        defaults to `5.0`.
    :type tth_initial_guess: float, optional
    :param interactive: Allows for user interactions, defaults to
        `False`.
    :type interactive: bool, optional
    :return: A saveable matplotlib figure and the selected initial
        guess for 2&theta.
    :type: matplotlib.figure.Figure, float
    """
    # Third party modules
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    def change_fig_title(title):
        if fig_title:
            fig_title[0].remove()
            fig_title.pop()
        fig_title.append(plt.figtext(*title_pos, title, **title_props))

    def change_error_text(error):
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        error_texts.append(plt.figtext(*error_pos, error, **error_props))

    def new_guess(tth):
        """Callback function for the tth input."""
        try:
            tth_new_guess = float(tth)
        except:
            change_error_text(
                r'Invalid 2$\theta$ 'f'cannot convert {tth} to float, '
                r'enter a valid 2$\theta$')
            return
        for i, (loc, hkl) in enumerate(zip(
                get_peak_locations(ds, tth_new_guess), hkls)):
            if i in hkl_peaks:
                j = hkl_peaks.index(i)
                hkl_lines[j].remove()
                hkl_lbls[j].remove()
                if x[0] <= loc <= x[-1]:
                    hkl_lines[j] = ax.axvline(loc, c='k', ls='--', lw=1)
                    hkl_lbls[j] = ax.text(loc, 1, str(hkls[i])[1:-1],
                                           ha='right', va='top', rotation=90,
                                           transform=ax.get_xaxis_transform())
                else:
                    hkl_peaks.pop(j)
                    hkl_lines.pop(j)
                    hkl_lbls.pop(j)
            elif x[0] <= loc <= x[-1]:
                hkl_peaks.append(i)
                hkl_lines.append(ax.axvline(loc, c='k', ls='--', lw=1))
                hkl_lbls.append(
                    ax.text(
                        loc, 1, str(hkl)[1:-1], ha='right', va='top',
                        rotation=90, transform=ax.get_xaxis_transform()))
        ax.get_figure().canvas.draw()

    def confirm(event):
        """Callback function for the "Confirm" button."""
        change_fig_title(r'Initial guess for 2$\theta$='f'{tth_input.text}')
        plt.close()

    fig_title = []
    error_texts = []

    title_pos = (0.5, 0.95)
    title_props = {'fontsize': 'xx-large', 'ha': 'center', 'va': 'bottom'}
    error_pos = (0.5, 0.90)
    error_props = {'fontsize': 'x-large', 'ha': 'center', 'va': 'bottom'}

    assert np.asarray(hkls).shape[1] == 3
    assert np.asarray(ds).size == np.asarray(hkls).shape[0]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.plot(x, y)
    ax.set_xlabel('MCA channel energy (keV)')
    ax.set_ylabel('MCA intensity (counts)')
    ax.set_xlim(x[0], x[-1])
    peak_locations = get_peak_locations(ds, tth_initial_guess)
    hkl_peaks = [i for i, loc in enumerate(peak_locations)
                   if x[0] <= loc <= x[-1]]
    hkl_lines = [ax.axvline(loc, c='k', ls='--', lw=1) \
                 for loc in peak_locations[hkl_peaks]]
    hkl_lbls = [ax.text(loc, 1, str(hkl)[1:-1],
                        ha='right', va='top', rotation=90,
                        transform=ax.get_xaxis_transform())
                for loc, hkl in zip(peak_locations[hkl_peaks], hkls)]

    if not interactive:

        change_fig_title(r'Initial guess for 2$\theta$='f'{tth_initial_guess}')

    else:

        change_fig_title(r'Adjust initial guess for 2$\theta$')
        fig.subplots_adjust(bottom=0.2)

        # Setup tth input
        tth_input = TextBox(plt.axes([0.125, 0.05, 0.15, 0.075]),
                            '$2\\theta$: ',
                            initial=tth_initial_guess)
        cid_update_tth = tth_input.on_submit(new_guess)

        # Setup "Confirm" button
        confirm_btn = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
        confirm_cid = confirm_btn.on_clicked(confirm)

        # Show figure for user interaction
        plt.show()

        # Disconnect all widget callbacks when figure is closed
        tth_input.disconnect(cid_update_tth)
        confirm_btn.disconnect(confirm_cid)

        # ...and remove the buttons before returning the figure
        tth_input.ax.remove()
        confirm_btn.ax.remove()

    fig_title[0].set_in_layout(True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if not interactive:
        tth_new_guess = tth_initial_guess
    else:
        try:
            tth_new_guess = float(tth_input.text)
        except:
            fig, tth_new_guess = select_tth_initial_guess(
                x, y, hkls, ds, tth_initial_guess, interactive)

    return fig, tth_new_guess

def select_material_params(x, y, tth, materials=[], interactive=False):
    """Interactively select the lattice parameters and space group for
    a list of materials. A matplotlib figure will be shown with a plot
    of the reference data (`x` and `y`). The figure will contain
    widgets to add / remove materials and update selections for space
    group number and lattice parameters for each one. The HKLs for the
    materials defined by the widgets' values will be shown over the
    reference data and updated when the widgets' values are
    updated.

    :param x: MCA channel energies.
    :type x: np.ndarray
    :param y: MCA intensities.
    :type y: np.ndarray
    :param tth: The (calibrated) 2&theta angle.
    :type tth: float
    :param materials: Materials to get HKLs and lattice spacings for.
    :type materials: list[hexrd.material.Material]
    :param interactive: Allows for user interactions, defaults to
        `False`.
    :type interactive: bool, optional
    :return: A saveable matplotlib figure and the selected materials
        for the strain analyses.
    :rtype: matplotlib.figure.Figure,
        list[CHAP.edd.models.MaterialConfig]
    """
    # Third party modules
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    # Local modules
    from CHAP.edd.models import MaterialConfig

    def change_error_text(error):
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        error_texts.append(plt.figtext(*error_pos, error, **error_props))

    def draw_plot():
        """Redraw plot of reference data and HKL locations based on
        the `_materials` list on the Matplotlib axes `ax`.
        """
        ax.clear()
        ax.set_title('Reference Data')
        ax.set_xlabel('MCA channel energy (keV)')
        ax.set_ylabel('MCA intensity (counts)')
        ax.set_xlim(x[0], x[-1])
        ax.plot(x, y)
        for i, material in enumerate(_materials):
            hkls, ds = get_unique_hkls_ds([material])            
            E0s = get_peak_locations(ds, tth)
            for hkl, E0 in zip(hkls, E0s):
                if x[0] <= E0 <= x[-1]:
                    ax.axvline(E0, c=f'C{i}', ls='--', lw=1)
                    ax.text(E0, 1, str(hkl)[1:-1], c=f'C{i}',
                            ha='right', va='top', rotation=90,
                            transform=ax.get_xaxis_transform())
        ax.get_figure().canvas.draw()

    def add_material(*args, material=None, new=True):
        """Callback function for the "Add material" button to add
        a new row of material-property-editing widgets to the figure
        and update the plot with new HKLs.
        """
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        if material is None:
            material = make_material('new_material', 225, 3.0)
            _materials.append(material)
        elif isinstance(material, MaterialConfig):
            material = material._material
        bottom = len(_materials) * 0.075
        plt.subplots_adjust(bottom=bottom + 0.125)
        name_input = TextBox(plt.axes([0.1, bottom, 0.09, 0.05]),
                             'Material: ',
                             initial=material.name)
        sgnum_input = TextBox(plt.axes([0.3, bottom, 0.06, 0.05]),
                              'Space Group: ',
                              initial=material.sgnum)
        a_input = TextBox(plt.axes([0.4, bottom, 0.06, 0.05]),
                          '$a$ ($\\AA$): ',
                          initial=material.latticeParameters[0].value)
        b_input = TextBox(plt.axes([0.5, bottom, 0.06, 0.05]),
                          '$b$ ($\\AA$): ',
                          initial=material.latticeParameters[1].value)
        c_input = TextBox(plt.axes([0.6, bottom, 0.06, 0.05]),
                          '$c$ ($\\AA$): ',
                          initial=material.latticeParameters[2].value)
        alpha_input = TextBox(plt.axes([0.7, bottom, 0.06, 0.05]),
                              '$\\alpha$ ($\\degree$): ',
                              initial=material.latticeParameters[3].value)
        beta_input = TextBox(plt.axes([0.8, bottom, 0.06, 0.05]),
                             '$\\beta$ ($\\degree$): ',
                             initial=material.latticeParameters[4].value)
        gamma_input = TextBox(plt.axes([0.9, bottom, 0.06, 0.05]),
                              '$\\gamma$ ($\\degree$): ',
                              initial=material.latticeParameters[5].value)
        widgets.append(
            (name_input, sgnum_input, a_input, b_input, c_input,
             alpha_input, beta_input, gamma_input))
        widget_callbacks.append(
            [(widget, widget.on_submit(update_materials)) \
             for widget in widgets[-1]])
        draw_plot()

    def update_materials(*args, **kwargs):
        """Callback function for the material-property-editing widgets
         button to validate input material properties from widgets,
         update the `_materials` list, and redraw the plot.
        """
        def set_vals(material_i):
            """Set all widget values from the `_materials` list for a
            particular material.
            """
            material = _materials[material_i]
            # Temporarily disconnect widget callbacks
            callbacks = widget_callbacks[material_i+2]
            for widget, callback in callbacks:
                widget.disconnect(callback)
            # Set widget values
            name_input, sgnum_input, \
                a_input, b_input, c_input, \
                alpha_input, beta_input, gamma_input = widgets[material_i]
            name_input.set_val(material.name)
            sgnum_input.set_val(material.sgnum)
            a_input.set_val(material.latticeParameters[0].value)
            b_input.set_val(material.latticeParameters[1].value)
            c_input.set_val(material.latticeParameters[2].value)
            alpha_input.set_val(material.latticeParameters[3].value)
            beta_input.set_val(material.latticeParameters[4].value)
            gamma_input.set_val(material.latticeParameters[5].value)
            # Reconnect widget callbacks
            for i, (w, cb) in enumerate(widget_callbacks[material_i+2]):
                widget_callbacks[material_i+2][i] = (
                    w, w.on_submit(update_materials))

        # Update the _materials list
        for i, (material,
                (name_input, sgnum_input,
                 a_input, b_input, c_input,
                 alpha_input, beta_input, gamma_input)) \
                in enumerate(zip(_materials, widgets)):
            # Skip if no parameters were changes on this material
            old_material_params = (
                material.name, material.sgnum,
                [material.latticeParameters[i].value for i in range(6)]
            )
            new_material_params = (
                name_input.text, int(sgnum_input.text),
                [float(a_input.text), float(b_input.text), float(c_input.text),
                 float(alpha_input.text), float(beta_input.text),
                 float(gamma_input.text)]
            )
            if old_material_params == new_material_params:
                continue
            try:
                new_material = make_material(*new_material_params)
            except:
                change_error_text(f'Bad input for {material.name}')
            else:
                _materials[i] = new_material
            finally:
                set_vals(i)

        # Redraw reference data plot
        draw_plot()

    def confirm(event):
        """Callback function for the "Confirm" button."""
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        plt.close()

    widgets = []
    widget_callbacks = []
    error_texts = []

    error_pos = (0.5, 0.95)
    error_props = {'fontsize': 'x-large', 'ha': 'center', 'va': 'bottom'}

    _materials = deepcopy(materials)
    for i, m in enumerate(_materials):
        if isinstance(m, MaterialConfig):
            _materials[i] = m._material

    # Set up plot of reference data
    fig, ax = plt.subplots(figsize=(11, 8.5))

    if interactive:

        plt.subplots_adjust(bottom=0.1)

        # Setup "Add material" button
        add_material_btn = Button(
            plt.axes([0.125, 0.015, 0.1, 0.05]), 'Add material')
        add_material_cid = add_material_btn.on_clicked(add_material)
        widget_callbacks.append([(add_material_btn, add_material_cid)])

        # Setup "Confirm" button
        confirm_btn = Button(plt.axes([0.75, 0.015, 0.1, 0.05]), 'Confirm')
        confirm_cid = confirm_btn.on_clicked(confirm)
        widget_callbacks.append([(confirm_btn, confirm_cid)])

        # Setup material-property-editing buttons for each material
        for material in _materials:
            add_material(material=material)

        # Show figure for user interaction
        plt.show()

        # Disconnect all widget callbacks when figure is closed
        # and remove the buttons before returning the figure
        for group in widget_callbacks:
            for widget, callback in group:
                widget.disconnect(callback)
                widget.ax.remove()

    fig.tight_layout()

    new_materials = [
        MaterialConfig(
            material_name=m.name, sgnum=m.sgnum,
            lattice_parameters=[
                m.latticeParameters[i].value for i in range(6)])
        for m in _materials]

    return fig, new_materials

def select_mask_and_hkls(x, y, hkls, ds, tth, preselected_bin_ranges=[],
        preselected_hkl_indices=[], detector_name=None, ref_map=None,
        flux_energy_range=None, calibration_bin_ranges=None,
        interactive=False):
    """Return a matplotlib figure to indicate data ranges and HKLs to
    include for fitting in EDD Ceria calibration and/or strain
    analysis.

    :param x: MCA channel energies.
    :type x: np.ndarray
    :param y: MCA intensities.
    :type y: np.ndarray
    :param hkls: Avaliable Unique HKL values to fit peaks for in the
        calibration routine.
    :type hkls: list[list[int]]
    :param ds: Lattice spacings associated with the unique HKL indices
        in angstroms.
    :type ds: list[float]
    :param tth: The (calibrated) 2&theta angle.
    :type tth: float
    :param preselected_bin_ranges: Preselected MCA channel index ranges
        whose data should be included after applying a mask,
        defaults to `[]`
    :type preselected_bin_ranges: list[list[int]], optional
    :param preselected_hkl_indices: Preselected unique HKL indices to
        fit peaks for in the calibration routine, defaults to `[]`.
    :type preselected_hkl_indices: list[int], optional
    :param detector_name: Name of the MCA detector element.
    :type detector_name: str, optional
    :param ref_map: Reference map of MCA intensities to show underneath
        the interactive plot.
    :type ref_map: np.ndarray, optional
    :param flux_energy_range: Energy range in eV in the flux file
        containing station beam energy in eV versus flux
    :type flux_energy_range: tuple(float, float), optional
    :param calibration_bin_ranges: MCA channel index ranges included
        in the detector calibration.
    :type calibration_bin_ranges: list[[int, int]], optional
    :param interactive: Allows for user interactions, defaults to
        `False`.
    :type interactive: bool, optional
    :return: A saveable matplotlib figure, the list of selected data
        index ranges to include, and the list of HKL indices to
        include
    :rtype: matplotlib.figure.Figure, list[list[int]], list[int]
    """
    # Third party modules
    import matplotlib.lines as mlines
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, SpanSelector

    # Local modules
    from CHAP.utils.general import (
        get_consecutive_int_range,
        index_nearest_down,
        index_nearest_upp,
    )

    def change_fig_title(title):
        if fig_title:
            fig_title[0].remove()
            fig_title.pop()
        fig_title.append(plt.figtext(*title_pos, title, **title_props))

    def change_error_text(error):
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        error_texts.append(plt.figtext(*error_pos, error, **error_props))

    def hkl_locations_in_any_span(hkl_index):
        """Return the index of the span where the location of a specific
        HKL resides. Return(-1 if outside any span."""
        if hkl_index < 0 or hkl_index>= len(hkl_locations):
            return -1
        for i, span in enumerate(spans):
            if (span.extents[0] <= hkl_locations[hkl_index] and
                    span.extents[1] >= hkl_locations[hkl_index]):
                return i
        return -1

    def on_span_select(xmin, xmax):
        """Callback function for the SpanSelector widget."""
        removed_hkls = False
        if not init_flag[0]:
            for hkl_index in deepcopy(selected_hkl_indices):
                if hkl_locations_in_any_span(hkl_index) < 0:
                    hkl_vlines[hkl_index].set(**excluded_hkl_props)
                    selected_hkl_indices.remove(hkl_index)
                    removed_hkls = True
        combined_spans = False
        combined_spans_test = True
        while combined_spans_test:
            combined_spans_test = False
            for i, span1 in enumerate(spans):
                for span2 in reversed(spans[i+1:]):
                    if (span1.extents[1] >= span2.extents[0]
                            and span1.extents[0] <= span2.extents[1]):
                        span1.extents = (
                            min(span1.extents[0], span2.extents[0]),
                            max(span1.extents[1], span2.extents[1]))
                        span2.set_visible(False)
                        spans.remove(span2)
                        combined_spans = True
                        combined_spans_test = True
                        break
                if combined_spans_test:
                    break
        if flux_energy_range is not None:
            for span in spans:
                min_ = max(span.extents[0], min_x)
                max_ = min(span.extents[1], max_x)
                span.extents = (min_, max_)
        added_hkls = False
        for hkl_index in range(len(hkl_locations)):
            if (hkl_index not in selected_hkl_indices
                    and hkl_locations_in_any_span(hkl_index) >= 0):
                hkl_vlines[hkl_index].set(**included_hkl_props)
                selected_hkl_indices.append(hkl_index)
                added_hkls = True
        if combined_spans:
            if added_hkls or removed_hkls:
                change_error_text(
                    'Combined overlapping spans and selected only HKL(s) '
                    'inside the selected energy mask')
            else:
                change_error_text(
                    'Combined overlapping spans in the selected energy mask')
        elif added_hkls and removed_hkls:
            change_error_text(
                'Adjusted the selected HKL(s) to match the selected '
                'energy mask')
        elif added_hkls:
            change_error_text(
                'Added HKL(s) to match the selected energy mask')
        elif removed_hkls:
            change_error_text(
                'Removed HKL(s) outside the selected energy mask')
        plt.draw()

    def add_span(event, xrange_init=None):
        """Callback function for the "Add span" button."""
        spans.append(
            SpanSelector(
                ax, on_span_select, 'horizontal', props=included_data_props,
                useblit=True, interactive=interactive, drag_from_anywhere=True,
                ignore_event_outside=True, grab_range=5))
        if xrange_init is None:
            xmin_init = min_x
            xmax_init = 0.5*(min_x + hkl_locations[0])
        else:
            xmin_init = max(min_x, xrange_init[0])
            xmax_init = min(max_x, xrange_init[1])
        spans[-1]._selection_completed = True
        spans[-1].extents = (xmin_init, xmax_init)
        spans[-1].onselect(xmin_init, xmax_init)

    def pick_hkl(event):
        """The "onpick" callback function."""
        try:
            hkl_index = hkl_vlines.index(event.artist)
        except:
            pass
        else:
            hkl_vline = event.artist
            if hkl_index in deepcopy(selected_hkl_indices):
                hkl_vline.set(**excluded_hkl_props)
                selected_hkl_indices.remove(hkl_index)
                span = spans[hkl_locations_in_any_span(hkl_index)]
                span_next_hkl_index = hkl_locations_in_any_span(hkl_index+1)
                span_prev_hkl_index = hkl_locations_in_any_span(hkl_index-1)
                if span_next_hkl_index < 0 and span_prev_hkl_index < 0:
                    span.set_visible(False)
                    spans.remove(span)
                elif span_next_hkl_index < 0:
                    span.extents = (
                        span.extents[0],
                        0.5*(hkl_locations[hkl_index-1]
                             + hkl_locations[hkl_index]))
                elif span_prev_hkl_index < 0:
                    span.extents = (
                        0.5*(hkl_locations[hkl_index]
                             + hkl_locations[hkl_index+1]),
                        span.extents[1])
                else:
                    xrange_init = [
                        0.5*(hkl_locations[hkl_index]
                             + hkl_locations[hkl_index+1]),
                        span.extents[1]]
                    span.extents = (
                        span.extents[0],
                        0.5*(hkl_locations[hkl_index-1]
                             + hkl_locations[hkl_index]))
                    add_span(None, xrange_init=xrange_init)
                change_error_text(
                    f'Adjusted the selected energy mask to reflect the '
                    'removed HKL')
            else:
                change_error_text(
                    f'Selected HKL is outside any current span, '
                    'extend or add spans before adding this value')
            plt.draw()

    def reset(event):
        """Callback function for the "Confirm" button."""
        for hkl_index in deepcopy(selected_hkl_indices):
            hkl_vlines[hkl_index].set(**excluded_hkl_props)
            selected_hkl_indices.remove(hkl_index)
        for span in reversed(spans):
            span.set_visible(False)
            spans.remove(span)
        plt.draw()

    def confirm(event):
        """Callback function for the "Confirm" button."""
        if not len(spans) or len(selected_hkl_indices) < 2:
            change_error_text('Select at least one span and two HKLs')
            plt.draw()
        else:
            if error_texts:
                error_texts[0].remove()
                error_texts.pop()
            if detector_name is None:
                change_fig_title('Selected data and HKLs used in fitting')
            else:
                change_fig_title(
                    f'Selected data and HKLs used in fitting {detector_name}')
            plt.close()

    selected_hkl_indices = preselected_hkl_indices
    spans = []
    hkl_vlines = []
    fig_title = []
    error_texts = []

    hkl_locations = [loc for loc in get_peak_locations(ds, tth)
                     if x[0] <= loc <= x[-1]]
    hkl_labels = [str(hkl)[1:-1] for hkl, loc in zip(hkls, hkl_locations)]

    title_pos = (0.5, 0.95)
    title_props = {'fontsize': 'xx-large', 'ha': 'center', 'va': 'bottom'}
    error_pos = (0.5, 0.90)
    error_props = {'fontsize': 'x-large', 'ha': 'center', 'va': 'bottom'}
    excluded_hkl_props = {
        'color': 'black', 'linestyle': '--','linewidth': 1,
        'marker': 10, 'markersize': 5, 'fillstyle': 'none'}
    included_hkl_props = {
        'color': 'green', 'linestyle': '-', 'linewidth': 2,
        'marker': 10, 'markersize': 10, 'fillstyle': 'full'}
    excluded_data_props = {
        'facecolor': 'white', 'edgecolor': 'gray', 'linestyle': ':'}
    included_data_props = {
        'alpha': 0.5, 'facecolor': 'tab:blue', 'edgecolor': 'blue'}

    if flux_energy_range is None:
        min_x = x.min()
        max_x = x.max()
    else:
        min_x = x[index_nearest_upp(x, max(x.min(), flux_energy_range[0]))]
        max_x = x[index_nearest_down(x, min(x.max(), flux_energy_range[1]))]

    if ref_map is None:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set(xlabel='Energy (keV)', ylabel='Intensity (counts)')
    else:
        fig, (ax, ax_map) = plt.subplots(
            2, sharex=True, figsize=(11, 8.5), height_ratios=[2, 1])
        ax.set(ylabel='Intensity (counts)')
        ax_map.pcolormesh(x, np.arange(ref_map.shape[0]), ref_map)
        ax_map.set_yticks([])
        ax_map.set_xlabel('Energy (keV)')
        ax_map.set_xlim(x[0], x[-1])
    handles = ax.plot(x, y, color='k', label='Reference Data')
    if calibration_bin_ranges is not None:
        ylow = ax.get_ylim()[0]
        for low, upp in calibration_bin_ranges:
            ax.plot([x[low], x[upp]], [ylow, ylow], color='r', linewidth=2)
        handles.append(mlines.Line2D(
            [], [], label='Energies included in calibration', color='r',
            linewidth=2))
    handles.append(mlines.Line2D(
        [], [], label='Excluded / unselected HKL', **excluded_hkl_props))
    handles.append(mlines.Line2D(
        [], [], label='Included / selected HKL', **included_hkl_props))
    handles.append(Patch(
        label='Excluded / unselected data', **excluded_data_props))
    handles.append(Patch(
        label='Included / selected data', **included_data_props))
    ax.legend(handles=handles)
    ax.set_xlim(x[0], x[-1])

    if selected_hkl_indices and not preselected_bin_ranges:
        index_ranges = get_consecutive_int_range(selected_hkl_indices)
        for index_range in index_ranges:
            i = index_range[0]
            if i:
                min_ = 0.5*(hkl_locations[i-1] + hkl_locations[i])
            else:
                min_ = 0.5*(min_x + hkl_locations[i])
            j = index_range[1]
            if j < len(hkl_locations)-1:
                max_ = 0.5*(hkl_locations[j] + hkl_locations[j+1])
            else:
                max_ = 0.5*(hkl_locations[j] + max_x)
            preselected_bin_ranges.append(
                [index_nearest_upp(x, min_), index_nearest_down(x, max_)])

    for i, (loc, lbl) in enumerate(zip(hkl_locations, hkl_labels)):
        nearest_index = np.searchsorted(x, loc)
        if i in selected_hkl_indices:
            hkl_vline = ax.axvline(loc, **included_hkl_props)
        else:
            hkl_vline = ax.axvline(loc, **excluded_hkl_props)
        ax.text(loc, 1, lbl, ha='right', va='top', rotation=90,
                transform=ax.get_xaxis_transform())
        hkl_vlines.append(hkl_vline)

    init_flag = [True]
    for bin_range in preselected_bin_ranges:
        add_span(None, xrange_init=x[bin_range])
    init_flag = [False]

    if not interactive:

        if detector_name is None:
            change_fig_title('Selected data and HKLs used in fitting')
        else:
            change_fig_title(
                f'Selected data and HKLs used in fitting {detector_name}')

    else:

        if detector_name is None:
            change_fig_title('Select data and HKLs to use in fitting')
        else:
            change_fig_title(
                f'Select data and HKLs to use in fitting {detector_name}')
        fig.subplots_adjust(bottom=0.2)

        # Setup "Add span" button
        add_span_btn = Button(plt.axes([0.125, 0.05, 0.15, 0.075]), 'Add span')
        add_span_cid = add_span_btn.on_clicked(add_span)

        for vline in hkl_vlines:
             vline.set_picker(5)
        pick_hkl_cid = fig.canvas.mpl_connect('pick_event', pick_hkl)

        # Setup "Reset" button
        reset_btn = Button(plt.axes([0.4375, 0.05, 0.15, 0.075]), 'Reset')
        reset_cid = reset_btn.on_clicked(reset)

        # Setup "Confirm" button
        confirm_btn = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
        confirm_cid = confirm_btn.on_clicked(confirm)

        # Show figure for user interaction
        plt.show()

        # Disconnect all widget callbacks when figure is closed
        add_span_btn.disconnect(add_span_cid)
        fig.canvas.mpl_disconnect(pick_hkl_cid)
        reset_btn.disconnect(reset_cid)
        confirm_btn.disconnect(confirm_cid)

        # ...and remove the buttons before returning the figure
        add_span_btn.ax.remove()
        confirm_btn.ax.remove()
        reset_btn.ax.remove()
        plt.subplots_adjust(bottom=0.0)

    selected_bin_ranges = [np.searchsorted(x, span.extents).tolist()
                           for span in spans]
    if not selected_bin_ranges:
        selected_bin_ranges = None
    if selected_hkl_indices:
        selected_hkl_indices = sorted(selected_hkl_indices)
    else:
        selected_hkl_indices = None

    fig_title[0].set_in_layout(True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    return fig, selected_bin_ranges, selected_hkl_indices
