"""Utility functions for EDD workflows"""

from scipy.constants import physical_constants
hc = 1e7 * physical_constants['Planck constant in eV/Hz'][0] \
     * physical_constants['speed of light in vacuum'][0]


def make_material(name, sgnum, lparms, dmin=0.6):
    """Return a hexrd.material.Material with the given properties.
    Taken from
    [CHEDDr](https://gitlab01.classe.cornell.edu/msn-c/cheddr/-/blob/master/notebooks/CHESS_EDD.py#L99).


    :param name: material name
    :type name: str
    :param sgnum: spage group number
    :type sgnum: int
    :param lparms: lattice parameters ([a, b, c, &#945;, &#946;,
        &#947;], or fewer as the symmetry of the space group allows --
        for instance, a cubic lattice with space group number 225 can
        just provide [a, ])
    :type lparms: list[float]
    :param dmin: dmin of the material in Angstrom (&#8491;), defaults to 0.6
    :type dmin: float, optional
    :return: a hexrd material
    :rtype: heard.material.Material
    """
    from hexrd.material import Material    
    from hexrd.valunits import valWUnit
    import numpy as np
    matl = Material()
    matl.name = name
    matl.sgnum = sgnum
    if isinstance(lparms, float):
        lparms = [lparms]
    matl.latticeParameters = lparms
    matl.dmin = valWUnit('lp', 'length',  dmin, 'angstrom')
    nhkls = len(matl.planeData.exclusions)
    matl.planeData.set_exclusions(np.zeros(nhkls, dtype=bool))
    return matl

def get_unique_hkls_ds(materials, tth_tol=None, tth_max=None, round_sig=8):
    """Return the unique HKLs and d-spacings for the given list of
    materials.

    :param materials: list of materials to get HKLs and d-spacings for
    :type materials: list[hexrd.material.Material]
    :return: unique HKLs, unique d-spacings
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    from copy import deepcopy
    import numpy as np
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


def select_hkls(detector, materials, tth, y, x, interactive):
    """Return a plot of `detector.fit_hkls` as a matplotlib
    figure. Optionally modify `detector.fit_hkls` by interacting with
    a matplotlib figure.

    :param detector: the detector to set `fit_hkls` on
    :type detector: MCAElementConfig
    :param material: the material to pick HKLs for
    :type material: MaterialConfig
    :param tth: diffraction angle two-theta
    :type tth: float
    :param y: reference y data to plot
    :type y: np.ndarray
    :param x: reference x data to plot
    :type x: np.ndarray
    :param interactive: show the plot and allow user interactions with
        the matplotlib figure
    :type interactive: bool
    :return: plot showing the user-selected HKLs
    :rtype: matplotlib.figure.Figure
    """
    import numpy as np
    hkls, ds = get_unique_hkls_ds(materials)
    peak_locations = hc / (2. * ds * np.sin(0.5 * np.radians(tth)))
    pre_selected_peak_indices = detector.fit_hkls \
                                if detector.fit_hkls else []
    from CHAP.utils.general import select_peaks
    selected_peaks, figure = select_peaks(
        y, x, peak_locations,
        peak_labels=[str(hkl)[1:-1] for hkl in hkls],
        pre_selected_peak_indices=pre_selected_peak_indices,
        mask=detector.mca_mask(),
        interactive=interactive,
        xlabel='MCA channel energy (keV)',
        ylabel='MCA intensity (counts)',
        title='Mask and HKLs for fitting')

    selected_hkl_indices = [int(np.where(peak_locations == peak)[0][0]) \
                            for peak in selected_peaks]
    detector.fit_hkls = selected_hkl_indices

    return figure


def select_tth_initial_guess(detector, material, y, x):
    """Show a matplotlib figure of a reference MCA spectrum on top of
    HKL locations. The figure includes an input field to adjust the
    initial tth guess and responds by updating the HKL locations based
    on the adjusted value of the initial tth guess.

    :param detector: the detector to set `tth_inital_guess` on
    :type detector: MCAElementConfig
    :param material: the material to show HKLs for
    :type material: MaterialConfig
    :param y: reference y data to plot
    :type y: np.ndarray
    :param x: reference x data to plot
    :type x: np.ndarray
    :return: None
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox
    import numpy as np

    tth_initial_guess = detector.tth_initial_guess \
                        if detector.tth_initial_guess is not None \
                        else 5.0
    hkls, ds = material.unique_ds(
        tth_tol=detector.hkl_tth_tol, tth_max=detector.tth_max)
    def get_peak_locations(tth):
        return hc / (2. * ds * np.sin(0.5 * np.radians(tth)))

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.plot(x, y)
    ax.set_xlabel('MCA channel energy (keV)')
    ax.set_ylabel('MCA intensity (counts)')
    ax.set_title('Adjust initial guess for $2\\theta$')
    hkl_lines = [ax.axvline(loc, c='k', ls='--', lw=1) \
                 for loc in get_peak_locations(tth_initial_guess)]
    hkl_lbls = [ax.text(loc, 1, str(hkl)[1:-1],
                        ha='right', va='top', rotation=90,
                        transform=ax.get_xaxis_transform()) \
                for loc, hkl in zip(get_peak_locations(tth_initial_guess),
                                    hkls)]

    # Callback for tth input
    def new_guess(tth):
        try:
            tth = float(tth)
        except:
            raise ValueError(f'Cannot convert {new_tth} to float')
        for i, (line, loc) in enumerate(zip(hkl_lines,
                                            get_peak_locations(tth))):
            line.remove()
            hkl_lines[i] = ax.axvline(loc, c='k', ls='--', lw=1)
            hkl_lbls[i].remove()
            hkl_lbls[i] = ax.text(loc, 1, str(hkls[i])[1:-1],
                                  ha='right', va='top', rotation=90,
                                  transform=ax.get_xaxis_transform())
        ax.get_figure().canvas.draw()
        detector.tth_initial_guess = tth

    # Setup tth input
    plt.subplots_adjust(bottom=0.25)
    tth_input = TextBox(plt.axes([0.125, 0.05, 0.15, 0.075]),
                        '$2\\theta$: ',
                        initial=tth_initial_guess)
    cid_update_tth = tth_input.on_submit(new_guess)

    # Setup "Confirm" button
    def confirm_selection(event):
        plt.close()
    confirm_b = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
    cid_confirm = confirm_b.on_clicked(confirm_selection)

    # Show figure for user interaction
    plt.show()

    # Disconnect all widget callbacks when figure is closed
    tth_input.disconnect(cid_update_tth)
    confirm_b.disconnect(cid_confirm)



def select_material_params(x, y, tth, materials=[]):
    """Interactively select lattice parameters and space group for a
    list of materials. A matplotlib figure will be shown with a plot
    of the reference data (`x` and `y`). The figure will contain
    widgets to add / remove materials and update selections for space
    group number and lattice parameters for each one. The HKLs for the
    materials defined by the widgets' values will be shown over the
    reference data and updated when the widgets' values are
    updated. Return a list of the selected materials when the figure
    is closed.
    """
    from copy import deepcopy
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox
    import numpy as np
    from CHAP.edd.models import MaterialConfig

    _materials = deepcopy(materials)
    for i, m in enumerate(_materials):
        if isinstance(m, MaterialConfig):
            _materials[i] = m._material

    # Set up plot of reference data
    fig, ax = plt.subplots(figsize=(11, 8.5))

    def draw_plot():
        """Redraw plot of reference data and HKL locations based on
        the `_materials` list on the matplotlib Axes `ax`.
        """
        ax.clear()
        ax.set_title('Reference Data')
        ax.set_xlabel('MCA channel energy (keV)')
        ax.set_ylabel('MCA intensity (counts)')
        ax.plot(x, y)
        for i, material in enumerate(_materials):
            hkls, ds = get_unique_hkls_ds([material])            
            E0s = hc / (2. * ds * np.sin(0.5 * np.radians(tth)))
            for hkl, E0 in zip(hkls, E0s):
                ax.axvline(E0, c=f'C{i}', ls='--', lw=1)
                ax.text(E0, 1, str(hkl)[1:-1], c=f'C{i}',
                        ha='right', va='top', rotation=90,
                        transform=ax.get_xaxis_transform())
        ax.get_figure().canvas.draw()

    # Confirm & close button
    widget_callbacks = []
    plt.subplots_adjust(bottom=0.1)
    def confirm_selection(event):
        plt.close()
    confirm_button = Button(plt.axes([0.75, 0.015, 0.1, 0.05]), 'Confirm')
    cid_confirm = confirm_button.on_clicked(confirm_selection)
    widget_callbacks.append([(confirm_button, cid_confirm)])

    # Widgets to edit materials
    widgets = []
    def update_materials(*args, **kwargs):
        """Validate input material properties from widgets, update the
        `_materials` list, and redraw the plot.
        """
        def set_vals(material_i):
            """Set all widget values from the `_materials` list for a
            particular material
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
                (name_input, sgnum_input, \
                 a_input, b_input, c_input, \
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

    def add_material(*args, material=None, new=True):
        """Add new row of material-property-editing widgets to the
        figure and update the plot with new HKLs
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

    # Button to add materials
    add_material_button = Button(
        plt.axes([0.125, 0.015, 0.1, 0.05]), 'Add material')
    cid_add_material = add_material_button.on_clicked(add_material)
    widget_callbacks.append([(add_material_button, cid_add_material)])

    # Draw data & show plot
    for material in _materials:
        add_material(material=material)
    plt.show()

    # Teardown after figure is closed
    for group in widget_callbacks:
        for widget, callback in group:
            widget.disconnect(callback)

    new_materials = [MaterialConfig(
        material_name=m.name, sgnum=m.sgnum,
        lattice_parameters=[m.latticeParameters[i].value for i in range(6)]) \
                     for m in _materials]
    return new_materials
