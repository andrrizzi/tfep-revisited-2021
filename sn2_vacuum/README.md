# SN2 reaction in vacuum

These folder contains the input files used to run the semi-empirical simulation of the chloromethane to fluoromethane
reaction in vacuum with the sander program in AMBER.

Optionally, you can recreate the conda environment used for the simulations in the paper by `cd`ing into `amber/` and executing
```
conda env create -f conda_environment.yml
```
In the original paper, we ran 4 independent repeats. Each repeat can be run by executing (after substituting the `X` with a number).
```
sander -O -i sn2_vacuum.in -o repeat-X/sn2_vacuum.out -r repeat-X/sn2_vacuum.rst -x repeat-X/sn2_vacuum.crd -frc repeat-X/sn2_vacuum.frc -inf repeat-X/sn2_vacuum.info -p sn2_vacuum.prmtop -c sn2_vacuum.inpcrd
```

Running the analysis in `../scripts` generates a `trp/` folder here.


## Manifest

- `amber/`
  - `sn2_vacuum.in`: Input script for the sander program in AMBER to perform the semi-empirical simulation.
  - `plumed.dat`: Input script for PLUMED with the metadynamics settings.
  - `conda_environment.yml`: The conda environment used for the simulations into a conda-readable format.
  - `sn2_vacuum.prmtop/inpcrd/pdb`: Initial coordinates and parameters for the molecular system.
