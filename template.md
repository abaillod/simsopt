---
title: 'Simsopt: A flexible framework for stellarator optimization'
tags:
  - Python
  - plasma
  - plasma physics
  - magnetohydrodynamics
  - optimization
  - stellarator
  - fusion energy
authors:
  - name: Adrian M. Price-Whelan^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: 1 June 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Blah blah.

[//]: # (Comments can be included like this.)

# Statement of need

Blah blah.

We optimize using the VMEC code [@VMEC1983; @VMEC1986] and SPEC code [@SPEC].



# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Examples

~~~python
import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.boozer import Boozer, Quasisymmetry
from simsopt.mhd.spec import Spec, Residue
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve

# Create objects for the Vmec and Spec equilibrium
mpi = MpiPartition()
vmec = Vmec("input.nfp2_QA_iota0.4", mpi=mpi)
surf = vmec.boundary
spec = Spec("nfp2_QA_iota0.4.sp", mpi=mpi)
spec.boundary = surf  # Identify the Vmec and Spec boundaries

# Configure quasisymmetry objective:
boozer = Boozer(vmec)
qs = Quasisymmetry(boozer,
                   0.5, # Radius s to target
                   1, 0) # (M, N) you want in |B|
# iota = p / q
p = -2
q = 5
residue1 = Residue(spec, p, q)
residue2 = Residue(spec, p, q, theta=np.pi)

# Define objective function                                                                                                                      
prob = LeastSquaresProblem([(vmec.aspect, 6, 1.0),
                            (vmec.iota_axis, 0.39, 1),
                            (vmec.iota_edge, 0.42, 1),
                            (qs, 0, 2),
                            (residue1, 0, 2),
                            (residue2, 0, 2)])

for step in range(3):
    max_mode = step + 3

    vmec.indata.mpol = 4 + step
    vmec.indata.ntor = vmec.indata.mpol

    boozer.mpol = 24 + step * 8
    boozer.ntor = boozer.mpol

    # Define parameter space:                                                                                                                    
    surf.all_fixed()
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.set_fixed("rc(0,0)") # Major radius                                                                                                     

    least_squares_mpi_solve(prob, mpi, grad=True)
~~~

# Acknowledgements

This work was supported by a grant from the Simons Foundation (560651, ML).

# References