# Scripts to reproduce the experiments

The two scripts in this folder (and the libraries in `modules/`) can be used to run the analysis for the double-well potential and the SN2 reaction in the paper.

Before running the analysis of the SN2 reaction, you must first perform the SQM simulations in repeats (see `../sn2_vacuum/`).

To load the same conda environment used during the analysis published in the paper, use
```
conda env create -f conda_environment.yml
```

Finally, to run the analysis code execute
```
python sn2_reaction.py -d "amber/repeat-0" -l "kl" --lr=0.001 --clrmin=0.0001 -b 256 -y 12 -w 0.001 -a "expensive2" -t 70000 --weightnorm -o "../sn2_vacuum/trp/kl-lr0001-clrmin00001-b256-y8-w0001-expensive2-wnorm-t70000" -p -e 280 --aepochs=1,280
```

## Manifest

- `double_well.py`: Perform the standard and targeted FEP analysis of the double-well potential example..
- `sn2_reaction.py`: Perform the standard and targeted FEP analysis of the chloromethane -> fluoromethane reaction in vacuum.
- `conda_environment.yml`: The conda environment used during the analysis published in the paper.
- `modules/`: Various Python packages and modules necessary for the analysis:
  - `modules/functions/`: Several functions for PyTorch to manipulate molecular geometries, restraint potentials, and interface to Psi4 for QM calculations.
  - `modules/nets/`: Implementation of normalizing flows in PyTorch.
  - `modules/plumedwrapper/`: Python interface for PLUMED executables and to analyze PLUMED output files. This include an auxiliary data reader for MDAnalysis.
  - `modules/tests/`: Test suite for the modules below.
  - `modules/data.py`: Utility classes to create to create PyTorch `Dataset`s from MDAnalysis `Trajectory` objects.
  - `modules/molecule.py`: Implementation of a `Molecule` class representing topology and geometry across packages.
  - `modules/loss.py`: Various loss functions based on the KL divergence between reference and target distributions.
  - `modules/reweighting.py`: Main utility functions and classes to perform standard and targeted FEP analysis from a trajectory dataset.

