#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Python utility class for handling molecule topologies.

The molecule class here follows the tag specification of the QCSchema.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================


# =============================================================================
# MOLECULE CLASS
# =============================================================================

class Molecule:
    """
    A utility class storing the topology of a molecular system.

    This class is based on "molecule" tag specification of the QCSchema.

    Parameters
    ----------
    geometry : pint.Quantity
        ``geometry[atom_idx][xyz_idx]`` is the ``xyz_idx`` component of the
         position of the atom ``atom_idx``.
    symbols : List[str]
        ``symbol[atom_idx]`` is the atomic element of atom ``atom_idx``.
    molecular_multiplicity : int, optional
        The overall multiplicity of the molecule.
    molecular_charge : int, optional
        The overall charge of the molecule.

    """
    def __init__(
            self,
            geometry,
            symbols,
            molecular_multiplicity=None,
            molecular_charge=None
    ):
        self.geometry = geometry
        self.symbols = symbols
        self.molecular_multiplicity = molecular_multiplicity
        self.molecular_charge = molecular_charge

    @property
    def n_atoms(self):
        """The number of atoms in the topology."""
        return len(self.symbols)

    @property
    def unique_symbols(self):
        """
        Set[str]: The unique symbols in the molecules.

        The order of the symbols in the list is deterministic.
        """
        found = set()
        unique = [x for x in self.symbols if x not in found and (found.add(x) or True)]
        return unique

    def get_n_symbols(self, symbol):
        """
        Return the number of atoms of the given element.

        Parameters
        ----------
        symbol : str
            The atomic element.

        Returns
        -------
        n_symbols : int
            The number of atoms of the given element.

        """
        return self.symbols.count(symbol)

    def get_symbol_indices(self, symbol):
        """
        Return the atom indices for all the atoms of a given element.

        Parameters
        ----------
        symbol : str
            The atomic element.

        Returns
        -------
        symbol_indices : List[int]
            The atom indexed by ``symbol_indices[i]`` has position
            ``molecule.geometry[symbol_indices[i]]``.

        """
        symbol_indices = [atom_idx for atom_idx in range(self.n_atoms)
                          if self.symbols[atom_idx] == symbol]
        return symbol_indices

    def get_symbol_positions(self, symbol):
        """
        Return the position of all the atoms of the given element.

        Parameters
        ----------
        symbol : str
            The atomic element.

        Returns
        -------
        symbol_positions : pint.Quantity
            ``symbol_positions[i][xyz_idx]`` is the ``xyz_idx`` component
            of the position of the atom ``get_symbol_indices(symbol)[i]``.

        """
        symbol_indices = self.get_symbol_indices(symbol)
        return self.geometry[symbol_indices]

    def to_psi4(self, reorient=False, translate=False):
        """
        Convert this molecule into a Psi4 Molecule object.

        The returned molecule is not activated in psi4.

        Parameters
        ----------
        reorient : bool, optional
            Psi4 automatically reorient the molecules to remove the
            symmetric rotational degrees of freedom. Set this to
            ``True`` to allow it. Default is ``False``.
        translate : bool, optional
            Psi4 automatically translate the molecules so that the center
            of mass is in the origin to remove the symmetric translational
            degrees of freedom. Set this to ``True`` to allow it. Default
            is ``False``.

        Returns
        -------
        psi4_molecule : psi4.core.Molecule
            A Psi4 Molecule object.

        """
        import psi4

        return psi4.core.Molecule.from_arrays(
            geom=self.geometry.magnitude,
            elem=self.symbols,
            units=str(self.geometry.units),
            fix_com=not translate,
            fix_orientation=not reorient,
            molecular_charge=self.molecular_charge,
            molecular_multiplicity=self.molecular_multiplicity
        )
