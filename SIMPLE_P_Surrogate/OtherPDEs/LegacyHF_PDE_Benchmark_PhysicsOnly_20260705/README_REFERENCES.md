# References

This benchmark uses manufactured exact solutions while keeping PDE forms close to common operator-learning and PINN benchmarks.
Boundary and initial conditions are enforced by hard output projection. No boundary-condition loss term is used.

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
  Used for: Operator-learning framing and source-to-solution Poisson/Burgers-style benchmarks.
- [PDEBench: An Extensive Benchmark for Scientific Machine Learning](https://arxiv.org/abs/2210.07182)
  Used for: PDE family selection: Burgers, wave, diffusion-reaction, Allen-Cahn-style transient PDEs.
- [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://www.nature.com/articles/s42256-021-00302-5)
  Used for: Manufactured/analytic operator-learning examples with supervised exact fields.
- [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
  Used for: Canonical Burgers, Allen-Cahn and KdV PDE forms; this script uses hard constraints and no BC loss.
- [The Zakharov-Kuznetsov equation and multidimensional KdV-type waves](https://doi.org/10.1016/0167-2789(74)90026-5)
  Used for: The steady 2D KdV case is treated as a forced ZK/KdV-type equation.
