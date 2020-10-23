# claudius: Analytic computations for scattering

Python toolbox to compute scattering and total field when we have analytical computation so typically when scatters are invariant by rotation and the incident field is a plane wave.

The word _claudius_ is an acronym for _CaLculs AnalytiqUes pour la DIffUSion_ the French translation of analytic computations for scattering.

## 2D Helmholtz

Given a wavenumber $k > 0$ and an incident field $u^{\mathsf{in}} : (x,y) \mapsto \mathsf{e}^{\mathsf{i} k y}$. Find the scattering field $u^{\mathsf{sc}} \in \mathrm{H}^1(\mathbb{R})$ such that the total field $u = u^{\mathsf{in}} + u^{\mathsf{sc}}$ satisfy $-\mu^{-1}\mathrm{div}(\varepsilon^{-1}\, \nabla u) - k^2 u = 0$, $[u]_\Gamma = 0$, $[\varepsilon^{-1}\, \partial_n u]_\Gamma = 0$, and $u$ is $k$-outgoing.

## 3D Helmholtz

## 3D Maxwell

## To Do

- 2D Helmholtz

  - Disk Dirichlet
  - Disk Neumann
  - Disk (constant)
  - Annulus (constant, flat well)

- 3D Helmholtz

  - Sphere Dirichlet
  - Sphere Neumann
  - Sphere (constant)
  - spherical shell (constant, flat well)

- 3D Maxwell

  - Sphere Dirichlet
  - Sphere Neumann
  - Sphere (constant)
  - spherical shell (constant, flat well)

## Acknowledgment

I would like to thank [Camille Carvalho](https://github.com/carvalhocamille) and Friedelinde for helping with the name.
