## Case Interpretation

The Case is the 2D steady Burger's flow problem, with the governing equation:
```math
	u\cdot\frac{\partial u}{\partial x} + v\cdot\frac{\partial u}{\partial y} = \nu \nabla^2 u;    	u\cdot\frac{\partial v}{\partial x} + v\cdot\frac{\partial v}{\partial y} = \nu \nabla^2 v
```



where $(x,y) \in [0,1]^2$

It is the first non-linear problem applied in this repository. The Raynolds number:

```math
	Re = \frac{U*L}{\nu}
```

where $U$ is the inlet flow velocity which could be adjusted to control the nonlinearity of the problem, $L=1$ is the characteristic length, and $\nu=0.01$ is the viscosity 


