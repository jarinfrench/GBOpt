# GBOpt
GBOpt is a Python package for creating, manipulating, and optimizing bicrystal grain boundary structures through configurable global optimization workflows. It uses a modular architecture, with separate modules for structure creation, manipulation, and optimization. In the initial release:
- GB creation is facilitated through a single parameterization of the macroscopic degrees of freedom (misorientation and inclination).
- GBs can be manipulated using grain translation, atom insertion, atom removal, and displacement along soft phonon modes.
- GB optimization is performed using either a Monte Carlo or an evolutionary algorithm optimizer, with energy evaluations currently using LAMMPS as the atomistic calculator.

The modular nature of the software is intended to allow easy extensibility to additional grain boundary parameterizations, structural manipulations, optimization engines, and external calculators.

To install, create a new conda environment
```
conda create --name GBOpt python
conda activate GBOpt
```

Make sure you are in the GBOpt directory:
```
cd ~/projects/GBOpt
```


Then install the dependencies
```
pip install .
```

## Logging

GBOpt does not configure logging automatically when imported as a library, and
existing warnings continue to use Python's `warnings` module.

To enable GBOpt console logs in a script or notebook:
```python
from GBOpt.Utils.logging_utils import configure_logging

configure_logging(level="INFO")
```

Use `level="DEBUG"` to include detailed Monte Carlo mutation events.
