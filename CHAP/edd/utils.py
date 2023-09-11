"""Utility functions for EDD workflows"""

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

    material = Material(name)
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
    # System modules
    from copy import deepcopy

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

def select_tth_initial_guess(x, y, hkls, ds, tth_initial_guess=5.0):
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
    :type fit_hkls: list[int]
    :param ds: Lattice spacings in angstroms associated with the
        unique HKL indices.
    :type ds: list[float]
    :ivar tth_initial_guess: Initial guess for 2&theta,
        defaults to `5.0`.
    :type tth_initial_guess: float, optional
    :return: Selected initial guess for 2&theta
    :type: float
    """
    # Third party modules
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    # Callback for tth input
    def new_guess(tth):
        """Callback function for the tth input."""
        try:
            tth_new_guess = float(tth)
        except:
            print(f'ValueError: Cannot convert {tth} to float')
            return
        for i, (line, loc) in enumerate(zip(
                hkl_lines, get_peak_locations(ds, tth_new_guess))):
            line.remove()
            hkl_lines[i] = ax.axvline(loc, c='k', ls='--', lw=1)
            hkl_lbls[i].remove()
            hkl_lbls[i] = ax.text(loc, 1, str(hkls[i])[1:-1],
                                  ha='right', va='top', rotation=90,
                                  transform=ax.get_xaxis_transform())
        ax.get_figure().canvas.draw()

    def confirm(event):
        """Callback function for the "Confirm" button."""
        plt.close()

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.plot(x, y)
    ax.set_xlabel('MCA channel energy (keV)')
    ax.set_ylabel('MCA intensity (counts)')
    ax.set_title('Adjust initial guess for $2\\theta$')
    hkl_lines = [ax.axvline(loc, c='k', ls='--', lw=1) \
                 for loc in get_peak_locations(ds, tth_initial_guess)]
    hkl_lbls = [ax.text(loc, 1, str(hkl)[1:-1],
                        ha='right', va='top', rotation=90,
                        transform=ax.get_xaxis_transform())
                for loc, hkl
                    in zip(get_peak_locations(ds, tth_initial_guess), hkls)]

    # Setup tth input
    plt.subplots_adjust(bottom=0.25)
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

    try:
        tth_new_guess = float(tth_input.text)
    except:
        tth_new_guess = select_tth_initial_guess(
            x, y, hkls, ds, tth_initial_guess=tth_initial_guess)

    return tth_new_guess

def select_material_params(x, y, tth, materials=[]):
    """Interactively select the lattice parameters and space group for
    a list of materials. A matplotlib figure will be shown with a plot
    of the reference data (`x` and `y`). The figure will contain
    widgets to add / remove materials and update selections for space
    group number and lattice parameters for each one. The HKLs for the
    materials defined by the widgets' values will be shown over the
    reference data and updated when the widgets' values are
    updated. It returns a list of the selected materials when the
    figure is closed.

    :param x: MCA channel energies.
    :type x: np.ndarray
    :param y: MCA intensities.
    :type y: np.ndarray
    :param tth: The (calibrated) 2&theta angle.
    :type tth: float
    :param materials: Materials to get HKLs and lattice spacings for.
    :type materials: list[hexrd.material.Material]
    :return: The selected materials for the strain analyses.
    :rtype: list[CHAP.edd.models.MaterialConfig]
    """
    # System modules
    from copy import deepcopy

    # Third party modules
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    # Local modules
    from CHAP.edd.models import MaterialConfig

    def draw_plot():
        """Redraw plot of reference data and HKL locations based on
        the `_materials` list on the Matplotlib axes `ax`.
        """
        ax.clear()
        ax.set_title('Reference Data')
        ax.set_xlabel('MCA channel energy (keV)')
        ax.set_ylabel('MCA intensity (counts)')
        ax.plot(x, y)
        for i, material in enumerate(_materials):
            hkls, ds = get_unique_hkls_ds([material])            
            E0s = get_peak_locations(ds, tth)
            for hkl, E0 in zip(hkls, E0s):
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
        if material is None:
            material = make_material('new_material', 225, 3.0)
            _materials.append(material)
        elif isinstance(material, MaterialConfig):
            material = material._material
        bottom = len(_materials) * 0.075
        plt.subplots_adjust(bottom=bottom + 0.125)
        name_input = TextBox(plt.axes([0.125, bottom, 0.06, 0.05]),
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
                print(f'Bad input for {material.name}')
            else:
                _materials[i] = new_material
            finally:
                set_vals(i)

        # Redraw reference data plot
        draw_plot()

    def confirm(event):
        """Callback function for the "Confirm" button."""
        plt.close()

    widgets = []
    widget_callbacks = []

    _materials = deepcopy(materials)
    for i, m in enumerate(_materials):
        if isinstance(m, MaterialConfig):
            _materials[i] = m._material

    # Set up plot of reference data
    fig, ax = plt.subplots(figsize=(11, 8.5))
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
    for group in widget_callbacks:
        for widget, callback in group:
            widget.disconnect(callback)

    new_materials = [
        MaterialConfig(
            material_name=m.name, sgnum=m.sgnum,
            lattice_parameters=[
                m.latticeParameters[i].value for i in range(6)])
        for m in _materials]

    return new_materials

def select_mask_and_hkls(x, y, hkls, ds, tth, preselected_bin_ranges=[],
        preselected_hkl_indices=[], detector_name=None, ref_map=None,
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
    :param interactive: Allows for user interactions, defaults to
        `False`.
    :type interactive: bool, optional
    :return: A saveable matplotlib figure, the list of selected data
        index ranges to include, and the list of HKL indices to
        include
    :rtype: matplotlib.figure.Figure, list[list[int]], list[int]
    """
    # System modules
    from copy import deepcopy

    # Third party modules
    import matplotlib.lines as mlines
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, SpanSelector

    def on_span_select(xmin, xmax):
        """Callback function for the SpanSelector widget."""
        removed_hkls = False
        for hkl_index in deepcopy(selected_hkl_indices):
            if not any(True if span.extents[0] <= hkl_locations[hkl_index] and
                    span.extents[1] >= hkl_locations[hkl_index] else False
                    for span in spans):
                hkl_vlines[hkl_index].set(**excluded_hkl_props)
                selected_hkl_indices.remove(hkl_index)
                removed_hkls = True
        if removed_hkls:
            print('Removed HKL(s) outside the currently selected energy mask')
        combined_spans = True
        while combined_spans:
            combined_spans = False
            for i, span1 in enumerate(spans):
                for span2 in reversed(spans[i+1:]):
                    if (span1.extents[1] >= span2.extents[0]
                            and span1.extents[0] <= span2.extents[1]):
                        print('Combined overlapping spans in the currently '
                            'selected energy mask')
                        span1.extents = (
                            min(span1.extents[0], span2.extents[0]),
                            max(span1.extents[1], span2.extents[1]))
                        span2.set_visible(False)
                        spans.remove(span2)
                        combined_spans = True
                        break
                if combined_spans:
                    break
        plt.draw()

    def add_span(event, xrange_init=None):
        """Callback function for the "Add span" button."""
        spans.append(
            SpanSelector(
                ax, on_span_select, 'horizontal', props=included_data_props,
                useblit=True, interactive=interactive, drag_from_anywhere=True,
                ignore_event_outside=True, grab_range=5))
        if xrange_init is None:
            xmin_init, xmax_init = min(x), 0.05*(max(x)-min(x))
        else:
            xmin_init, xmax_init = xrange_init
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
            if hkl_index in selected_hkl_indices:
                hkl_vline.set(**excluded_hkl_props)
                selected_hkl_indices.remove(hkl_index)
            else:
                if any(True if span.extents[0] <= hkl_locations[hkl_index] and
                        span.extents[1] >= hkl_locations[hkl_index] else False
                        for span in spans):
                    hkl_vline.set(**included_hkl_props)
                    selected_hkl_indices.append(hkl_index)
                else:
                    print(f'Selected HKL is outside any current span, '
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
        plt.close()

    selected_hkl_indices = preselected_hkl_indices
    spans = []
    hkl_vlines = []

    hkl_locations = get_peak_locations(ds, tth)
    hkl_labels = [str(hkl)[1:-1] for hkl in hkls]

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

    if ref_map is None:
        fig, ax = plt.subplots(figsize=(11, 8.5))
    else:
        fig, (ax, ax_map) = plt.subplots(
            2, sharex=True, figsize=(11, 8.5), height_ratios=[2, 1])
        ax_map.pcolormesh(x, np.arange(ref_map.shape[0]), ref_map)
        ax_map.set_yticks([])
        ax_map.set_xlabel('Energy (keV)')
    handles = ax.plot(x, y, color='k', label='Reference Data')
    handles.append(mlines.Line2D(
        [], [], label='Excluded / unselected HKL', **excluded_hkl_props))
    handles.append(mlines.Line2D(
        [], [], label='Included / selected HKL', **included_hkl_props))
    handles.append(Patch(
        label='Excluded / unselected data', **excluded_data_props))
    handles.append(Patch(
        label='Included / selected data', **included_data_props))
    ax.legend(handles=handles)
    ax.set(xlabel='Energy (keV)', ylabel='Intensity (counts)')
    if detector_name is None:
        fig.suptitle('Select data and HKLs to use in fitting')
    else:
        fig.suptitle(f'Select data and HKLs to use in fitting {detector_name}')
    fig.subplots_adjust(bottom=0.2)

    for bin_range in preselected_bin_ranges:
        add_span(None, xrange_init=x[bin_range])

    for i, (loc, lbl) in enumerate(zip(hkl_locations, hkl_labels)):
        nearest_index = np.searchsorted(x, loc)
        if i in selected_hkl_indices:
            hkl_vline = ax.axvline(loc, **included_hkl_props)
        else:
            hkl_vline = ax.axvline(loc, **excluded_hkl_props)
        ax.text(loc, 1, lbl, ha='right', va='top', rotation=90,
                transform=ax.get_xaxis_transform())
        hkl_vlines.append(hkl_vline)

    if interactive:

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
    selected_hkl_indices = sorted(selected_hkl_indices)

    return fig, selected_bin_ranges, selected_hkl_indices
