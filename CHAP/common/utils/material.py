#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:21:25 2022

@author: rv43
"""

import logging

import os
import numpy as np
try:
    import xrayutilities as xu
    have_xu = True
except:
    have_xu = False
    pass
try:
    from hexrd import material
    have_hexrd = True
except:
    have_hexrd = False
    pass
if have_hexrd:
    try:
        from hexrd.valunits import valWUnit
    except:
        have_hexrd = False
        pass

powder_intensity_cutoff = 1.e-8

class Material:
    """Base class for materials in an sin2psi or EDD analysis.
       Right now it assumes a single material
           extend its ability to do differently when test data is available
    """

    def __init__(self, material_name=None, material_file=None, sgnum=None,
            lattice_parameters_angstroms=None, atoms=None, pos=None, enrgy=None):
        self._enrgy = enrgy
        self._materials = []
        self._ds_min = []
        self._ds_unique = None
        self._hkls_unique = None
        if material_name is not None:
            self.add_material(material_name, material_file, sgnum, lattice_parameters_angstroms,
                    atoms, pos)

    @property
    #FIX passing arguments to a property isn't correct?
    def lattice_parameters(self, index=0):
        """Convert from internal nm units to angstrom
        """
        matl = self._materials[index]
        if isinstance(matl, xu.materials.material.Crystal):
            return [matl.a, matl.b, matl.c]
        elif isinstance(matl, material.Material):
            return [l.getVal("angstrom") for l in self._materials[index].latticeParameters[0:3]]
        else:
            raise ValueError('Illegal material class type')
            return None

    @property
    def ds_unique(self, tth_tol=None, tth_max=None, round_sig=8):
        if self._ds_unique is None:
            self.get_unique_ds(tth_tol, tth_max, round_sig)
        return self._ds_unique

    @property
    def hkls_unique(self, tth_tol=None, tth_max=None, round_sig=8):
        if self._hkls_unique is None:
            self.get_unique_ds(tth_tol, tth_max, round_sig)
        return self._hkls_unique

    def add_material(self, material_name, material_file=None, sgnum=None,
            lattice_parameters_angstroms=None, atoms=None, pos=None, dmin_angstroms=0.6):
        # At this point only for a single material
        # Unique energies works for more, but fitting with different materials is not implemented
        if len(self._materials) == 1:
            exit('Multiple materials not implemented yet')
        self._ds_min.append(dmin_angstroms)
        self._materials.append(Material.make_material(material_name, material_file, sgnum,
                lattice_parameters_angstroms, atoms, pos, dmin_angstroms))

    def get_unique_ds(self, tth_tol=None, tth_max=None, round_sig=8):
        """Get the list of unique lattice spacings from material HKLs

        Parameters
        ----------
        tth_tol     : two theta tolerance (in degrees)
        tth_max     : maximum two theta value (in degrees)
        round_sig   : significant digits, passed to round() function

        Returns
        -------
        hkls     : list of hkl's corresponding to the unique lattice spacings
        ds       : list of the unique lattice spacings
        """
        hkls = np.empty((0,3))
        ds = np.empty((0))
        ids = np.empty((0))
        for i,m in enumerate(self._materials):
            material_class_valid = False
            if have_xu:
                if isinstance(m, xu.materials.material.Crystal):
                    powder = xu.simpack.PowderDiffraction(m, en=self._enrgy)
                    hklsi = [hkl for hkl in powder.data if powder.data[hkl]['active']]
                    dsi = [m.planeDistance(hkl) for hkl in powder.data if powder.data[hkl]['active']]
                    mask = [True if d > self._ds_min[i] else False for d in dsi]
                    hkls = np.vstack((hkls, np.array(hklsi)[mask,:]))
                    dsi = np.array(dsi)[mask]
                    material_class_valid = True
            if have_hexrd:
                if isinstance(m, material.Material):
                    pd = m.planeData
                    if tth_tol is not None:
                        pd.tThWidth = np.radians(tth_tol)
                    if tth_max is not None:
                        pd.exclusions = None
                        pd.tThMax = np.radians(tth_max)
                    hkls = np.vstack((hkls, pd.hkls.T))
                    dsi = pd.getPlaneSpacings()
                    material_class_valid = True
            if not material_class_valid:
                raise ValueError('Illegal material class type')
            ds = np.hstack((ds, dsi))
            ids = np.hstack((ids, i*np.ones(len(dsi))))

        # Sort lattice spacings in reverse order (use -)
        ds_unique, ids_unique, _ = np.unique(-ds.round(round_sig), return_index=True,
                return_counts=True)
        ds_unique = np.abs(ds_unique)

        # Limit the list to unique lattice spacings
        self._hkls_unique = hkls[ids_unique,:].astype(int)
        self._ds_unique = ds[ids_unique]
        hkl_list = np.vstack((np.arange(self._ds_unique.shape[0]), ids[ids_unique],
                self._hkls_unique.T, self._ds_unique)).T
        logging.info("Unique d's:")
        for hkl in hkl_list:
            logging.info(f'{hkl[0]:4.0f} {hkl[1]:.0f} {hkl[2]:.0f} {hkl[3]:.0f} {hkl[4]:.0f} '+
                    f'{hkl[5]:.6f}')

        return self._hkls_unique, self._ds_unique

    @staticmethod
    def make_material(material_name, material_file=None, sgnum=None,
            lattice_parameters_angstroms=None, atoms=None, pos=None, dmin_angstroms=0.6):
        """ Use HeXRD to get material properties when a materials file is provided
            Use xrayutilities otherwise
        """
        if not isinstance(material_name, str):
            raise ValueError(f'Illegal material_name: {material_name} {type(material_name)}')
        if lattice_parameters_angstroms is not None:
            if material_file is not None:
                logging.warning('Overwrite lattice_parameters of material_file with input values '+
                        f'({lattice_parameters_angstroms})')
            if isinstance(lattice_parameters_angstroms, (int, float)):
                lattice_parameters = [lattice_parameters_angstroms]
            elif isinstance(lattice_parameters_angstroms, (tuple, list, np.ndarray)):
                lattice_parameters = list(lattice_parameters_angstroms)
            else:
                raise ValueError(f'Illegal lattice_parameters_angstroms: '+
                        f'{lattice_parameters_angstroms} {type(lattice_parameters_angstroms)}')
        if material_file is None:
            if not isinstance(sgnum, int):
                raise ValueError(f'Illegal sgnum: {sgnum} {type(sgnum)}')
            if sgnum is None or lattice_parameters_angstroms is None or pos is None:
                raise ValueError('Valid inputs for sgnum and lattice_parameters_angstroms and '+
                        'pos are required if materials file is not specified')
            if isinstance(pos, str):
                pos = [pos]
            use_xu = True
            if (np.array(pos).ndim == 1 and isinstance(pos[0], (int, float)) and
                    np.array(pos).size == 3):
                if have_hexrd:
                    pos = np.array([pos])
                    use_xu = False
            elif (np.array(pos).ndim == 2 and np.array(pos).shape[0] > 0 and
                    np.array(pos).shape[1] == 3):
                if have_hexrd:
                    pos = np.array(pos)
                    use_xu = False
            elif not (np.array(pos).ndim == 1 and isinstance(pos[0], str) and
                    np.array(pos).size > 0 and have_xu):
                raise ValueError(f'Illegal pos (have_xu = {have_xu}): {pos} {type(pos)}')
            if use_xu:
                if atoms is None:
                    atoms = [material_name]
                matl = xu.materials.Crystal(material_name, xu.materials.SGLattice(sgnum,
                        *lattice_parameters, atoms=atoms, pos=[p for p in np.array(pos)]))
            else:
                matl = material.Material(material_name)
                matl.sgnum = sgnum
                matl.atominfo = np.vstack((pos.T, np.ones(pos.shape[0]))).T
                matl.latticeParameters = lattice_parameters
                matl.dmin = valWUnit('lp', 'length',  dmin_angstroms, 'angstrom')
                exclusions = matl.planeData.get_exclusions()
                powder_intensity = matl.planeData.powder_intensity
                exclusions = [True if exclusion or i >= len(powder_intensity) or
                        powder_intensity[i] < powder_intensity_cutoff else False
                        for i, exclusion in enumerate(exclusions)]
                matl.planeData.set_exclusions(exclusions)
                logging.debug(f'powder_intensity = {matl.planeData.powder_intensity}')
                logging.debug(f'exclusions = {matl.planeData.exclusions}')
        else:
            if not have_hexrd:
                raise ValueError('Illegal inputs: must provide detailed material info when '+
                        'hexrd package is unavailable')
            if sgnum is not None:
                logging.warning('Ignore sgnum input when material_file is specified')
            if not (os.path.splitext(material_file)[1] in ('.h5', '.hdf5', '.xtal', '.cif')):
                raise ValueError(f'Illegal material file {material_file}')
            matl = material.Material(material_name, material_file,
                    dmin=valWUnit('lp', 'length',  dmin_angstroms, 'angstrom'))
            if lattice_parameters_angstroms is not None:
                matl.latticeParameters = lattice_parameters
            exclusions = matl.planeData.get_exclusions()
            powder_intensity = matl.planeData.powder_intensity
            exclusions = [True if exclusion or i >= len(powder_intensity) or
                    powder_intensity[i] < powder_intensity_cutoff else False
                    for i, exclusion in enumerate(exclusions)]
            matl.planeData.set_exclusions(exclusions)
            logging.debug(f'powder_intensity = {matl.planeData.powder_intensity}')
            logging.debug(f'exclusions = {matl.planeData.exclusions}')

        return matl

