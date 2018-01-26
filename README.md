Cubic_MP2

This is an implementation of the cubic scaling MP2 from https://arxiv.org/abs/1710.01004. It is very slow and just a pilot code. For the optimized direct MP2 version, see the Cubic_MP2J repository.

Requirements: PySCF, NumPy, Cython, ASE, etc. (TODO)

./comp.sh
python cubic_mp2_JK.py 10

10 is the mesh in a single direction (i.e., 10x10x10 mesh in 3D)
