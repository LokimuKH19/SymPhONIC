# Hard Constraints Matters for PINNs

In this filefolder, several experiments using various structures of PINN solving Lid-driven cavity flow problem at `Re=100` is conducted. From the final result it's obvious that applying hard constraints in the forward functions of PINN to encode boundary conditions as many as we can would appear to significantly improve the network performance especially in the cases without data supervision.

## Contents
- `PINN_Coupled_uvp.py`: PINN which takes `u,v,p` as output, trained by original Navier-Stokes equations;
- `PINN_VelocityOnly.py`: PINN which limits the output dimension with only `u, v`, with minimal energy functional training objective (weak solution), additionaly integrating the strong formation of vorticity, continuity, pressure Possion equations.
- `simple_traditional.py`: Very old-school SIMPLE algorithm, but a vectorized, paralleled one. Jacobian iteration is introduced to solve pressure modification equation instead of the conventional direct solution of a linear dynamic system which requires an $O(n^2)$ process to obtain the inversed, large-scaled matrix.
- `NeuroOperators.py`: Toolbox for operator learning methods `FNO,CNO,CFNO`. See [`../WhyWeakFNO`](../WhyWeakFNO) for details.
- `CFNO_Coupled_uvp.py` and `CFNO_VelocityOnly.py`: The opeator learning version of the related PINN programs. (Quite wonder why the operator learning is not satisfying here)

## Update on Apr. 20th
- `simple_operator.py`: Uses the CFNO operator learning as the solver of pressure correlation equation to avoid complicated Krylov subspace based methods (BiCG, GMRES, etc.). Basic targets are generally satisfied. However, the issue of trivial solution has not been completely fixed yet.
