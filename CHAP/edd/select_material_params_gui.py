from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox

class MaterialParamSelector:
    def __init__(self, root, x, y, tth, label, preselected_materials,
                 on_complete):

        self.root = root
        self.root.title('Material Parameter Selection')
        self.on_complete = on_complete  # Completion callback
        
        # Reference data
        self.ref_data_x = x
        self.ref_data_y = y
        self.ref_data_label = label
        self.tth = tth
        
        # Materials
        self.materials = []
        self.selected_material = None
        
        # Create plot
        self.figure, self.ax = plt.subplots()
        self.legend_handles = []
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=12)

        # Widgets for material list and parameters
        self.material_listbox = tk.Listbox(root, height=5)
        self.material_listbox.grid(row=0, column=1, rowspan=2)
        self.material_listbox.bind(
            '<<ListboxSelect>>', self.on_material_select)

        self.add_material_button = tk.Button(
            root, text='Add Material', command=self.add_material)
        self.add_material_button.grid(row=0, column=2)
        
        self.remove_material_button = tk.Button(
            root, text='Remove Material', command=self.remove_material)
        self.remove_material_button.grid(row=1, column=2)
        
        # Parameter fields
        self.fields = {}
        for i, field in enumerate(
                ['Material Name', 'Space Group',
                 'a', 'b', 'c', 'alpha', 'beta', 'gamma']):
            if i > 1:
                units = 'Angstroms' if i < 5 else 'degrees'
                text = f'{field} ({units})'
            else:
                text = field
            tk.Label(root, text=text).grid(row=i+2, column=1)
            entry = tk.Entry(root)
            entry.grid(row=i+2, column=2)
            self.fields[field] = entry

        self.update_button = tk.Button(
            root, text='Update Material Properties',
            command=self.update_material)
        self.update_button.grid(row=11, column=1, columnspan=2)
        self.update_button = tk.Button(
            root, text='Update Material Properties',
            command=self.update_material)
        self.update_button.grid(row=11, column=1, columnspan=2)

        self.confirm_button = tk.Button(
            root, text='Confirm\nAll\nSelected\nMaterial\nProperties',
            command=self.on_close)
        self.confirm_button.grid(row=0, column=3, rowspan=12)

        # Initial Material Data
        if not preselected_materials:
            self.materials = [None]
            self.add_material(None)
        else:
            for material in preselected_materials:
                self.add_material(material)
        self.selected_material = 0

        # Overwrite the root window's close action to call `self.on_close`
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def plot_reference_data(self):
        # Plot reference data as a simple sine wave for illustration
        handle = self.ax.plot(
            self.ref_data_x, self.ref_data_y, label=self.ref_data_label)
        self.legend_handles = handle
        self.ax.legend(handles=self.legend_handles)
        self.ax.set_xlabel('Energy (keV)')
        self.ax.set_ylabel('Intensity (a.u)')
        self.canvas.draw()

    def add_material(self, new_material=None):
        from CHAP.edd.models import MaterialConfig
        if new_material is None:
            new_material = MaterialConfig(
                material_name='Ti64',
                sgnum=194,
                lattice_parameters=[2.9217, 4.66027]
            )
        self.materials.append(new_material)
        self.material_listbox.insert(tk.END, new_material.material_name)
        self.material_listbox.select_set(tk.END)
        self.on_material_select(None)
        self.update_plot()
        
    def remove_material(self):
        if self.selected_material is not None:
            self.materials.pop(self.selected_material)
            self.material_listbox.delete(self.selected_material)
            self.selected_material = None
            self.clear_fields()
            self.update_plot()

    def update_material(self):
        from CHAP.edd.utils import make_material

        if self.selected_material is None:
            return
        
        material = self.materials[self.selected_material]
        try:
            # Retrieve values from fields
            name = self.fields['Material Name'].get()
            sgnum = int(self.fields['Space Group'].get())
            lattice_parameters = [
                float(self.fields[param].get())
                for param in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')
            ]
            # Make a hexrd material from those values so we can
            # propagate any other updates required by the material's
            # symmetries            
            _material = make_material(name, sgnum, lattice_parameters)
            material.material_name = name
            material.sgnum = _material.sgnum
            material.lattice_parameters = [
                _material.latticeParameters[i].value for i in range(6)
            ]
            material._material = _material
            # If the updated field forces other field(s) to get new
            # values (because of space group symmetries), propagate
            # those new values to the gui entries too.
            for key, entry in self.fields.items():
                if key == 'Material Name':
                    continue
                entry.delete(0, tk.END)
            self.fields['Space Group'].insert(0, str(material.sgnum))
            for i, key in enumerate(('a', 'b', 'c', 'alpha', 'beta', 'gamma')):
                self.fields[key].insert(
                    0, str(_material.latticeParameters[i].value))
            
            # Update the listbox name display
            self.material_listbox.delete(self.selected_material)
            self.material_listbox.insert(
                self.selected_material, material.material_name)
            self.update_plot()
        except ValueError:
            messagebox.showerror(
                'Invalid input',
                'Please enter valid numbers for lattice parameters.')

    def on_material_select(self, event):
        if len(self.material_listbox.curselection()) == 0:
            # Listbox item deselection event can be ignored
            return
        # Update the selected material index
        self.selected_material = self.material_listbox.curselection()[0]
        material = self.materials[self.selected_material]
        self.clear_fields()
        self.fields['Material Name'].insert(0, material.material_name)
        self.fields['Space Group'].insert(0, str(material.sgnum))
        for i, key in enumerate(('a', 'b', 'c', 'alpha', 'beta', 'gamma')):
            self.fields[key].insert(
                0, str(material._material.latticeParameters[i].value))

    def clear_fields(self):
        for entry in self.fields.values():
            entry.delete(0, tk.END)

    def update_plot(self):
        from CHAP.edd.utils import (
            get_unique_hkls_ds, get_peak_locations
        )
        self.ax.cla()
        self.legend_handles = []
        self.plot_reference_data()  # Re-plot reference data

        # Plot each material's hkl peak locations
        for i, material in enumerate(self.materials):
            hkls, ds = get_unique_hkls_ds([material])
            E0s = get_peak_locations(ds, self.tth)
            for hkl, E0 in zip(hkls, E0s):
                if E0 < min(self.ref_data_x) or E0 > max(self.ref_data_x):
                    continue
                line = self.ax.axvline(
                    E0, c=f'C{i+1}', ls='--', lw=1,
                    label=material.material_name)
                self.ax.text(E0, 1, str(hkl)[1:-1], c=f'C{i+1}',
                             ha='right', va='top', rotation=90,
                             transform=self.ax.get_xaxis_transform())
            self.legend_handles.append(line)
        self.ax.legend(handles=self.legend_handles)
        self.canvas.draw()

    def on_close(self):
        """Handle closing the GUI and triggering the on_complete
        callback."""
        if self.on_complete:
            self.on_complete(self.materials, self.figure)
        self.root.destroy()  # Close the tkinter root window


def run_material_selector(
        x, y, tth, preselected_materials=None, label='Reference Data',
        on_complete=None, interactive=False):
    """Run the MaterialParamSelector tkinter application.

    :param x: MCA channel energies.
    :type x: np.ndarray
    :param y: MCA intensities.
    :type y: np.ndarray
    :param tth: The (calibrated) 2&theta; angle.
    :type tth: float
    :param preselected_materials: Materials to get HKLs and lattice
        spacings for.
    :type preselected_materials: list[hexrd.material.Material], optional
    :param label: Legend label for the 1D plot of reference MCA data
        from the parameters `x`, `y`, defaults to `"Reference Data"`.
    :type label: str, optional
    :param on_complete: Callback function to handle completion of the
        material selection, defaults to `None`.
    :type on_complete: Callable, optional
    :param interactive: Show the plot and allow user interactions with
        the matplotlib figure, defaults to `False`.
    :type interactive: bool, optional
    :return: The selected materials for the strain analyses.
    :rtype: list[CHAP.edd.models.MaterialConfig]
    """
    import tkinter as tk

    # Initialize the main application window
    root = tk.Tk()
    
    # Create the material parameter selection GUI within the main
    # window
    # This GUI allows the user to adjust and visualize lattice
    # parameters and space group
    app = MaterialParamSelector(
        root, x, y, tth, preselected_materials, label, on_complete)
    
    if interactive:
        # If interactive mode is enabled, start the GUI event loop to
        # allow user interaction
        root.mainloop()
    else:
        # If not in interactive mode, immediately close the
        # application
        app.on_close()


def select_material_params(
        x, y, tth, preselected_materials=None, label='Reference Data',
        interactive=False, filename=None):
    """Interactively adjust the lattice parameters and space group for
    a list of materials. It is possible to add / remove materials from
    the list.

    :param x: MCA channel energies.
    :type x: np.ndarray
    :param y: MCA intensities.
    :type y: np.ndarray
    :param tth: The (calibrated) 2&theta angle.
    :type tth: float
    :param preselected_materials: Materials to get HKLs and
        lattice spacings for.
    :type preselected_materials: list[hexrd.material.Material],
        optional
    :param label: Legend label for the 1D plot of reference MCA data
        from the parameters `x`, `y`, defaults to `"Reference Data"`.
    :type label: str, optional
    :param interactive: Show the plot and allow user interactions with
        the matplotlib figure, defaults to `False`.
    :type interactive: bool, optional
    :param filename: Save a .png of the plot to filename, defaults to
        `None`, in which case the plot is not saved.
    :type filename: str, optional
    :return: The selected materials for the strain analyses.
    :rtype: list[CHAP.edd.models.MaterialConfig]
    """
    # Local modules
    from CHAP.edd.select_material_params_gui import run_material_selector

    # Run the MaterialParamSelector with the callback function to
    # handle the materials data and, if requested, the output figure
    materials = None
    figure = None
    def on_complete(_materials, _figure):
        nonlocal materials, figure
        materials = _materials
        figure = _figure

    run_material_selector(x, y, tth, label, preselected_materials,
                          on_complete, interactive)

    if filename is not None:
        figure.savefig(filename)

    return materials


if __name__ == '__main__':
    import numpy as np
    from CHAP.edd.models import MaterialConfig

    x = np.linspace(40, 100, 100)
    y = np.sin(x)
    tth = 5

    preselected_materials = [
        MaterialConfig(
            material_name='Ti64_orig',
            sgnum=194,
            lattice_parameters=[2.9217, 4.66027]
        )
    ]
    materials = select_material_params(
        x, y, tth, preselected_materials=preselected_materials,
        interactive=True,
        filename=None,
    )
    print(f'Returned materials: {materials}')
