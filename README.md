VXQR1 BlackBox Solver
=====================

VXQR1 performs an approximate unconstrained minimization of a not 
necessarily smooth function of many continuous arguments. 
No gradients are needed.
A limited amount of noise is tolerated. 

The primary goal is to get in high dimensions quick improvements 
with few function evaluations, not necessarily to find the true 
global minimum.
 
A bounded initial search region must be specified but the actual 
search is not confined to this region; thus the best point found 
may be outside the search region.

The program was originally written by Arnold Neumaier (University of Vienna). 
Originally published MATLAB code at
<http://www.mat.univie.ac.at/~neum/software/vxqr1/>
and ported to Python by [Harald Schilly](http://harald.schil.ly).
 
Please inform the author at
[Arnold.Neumaier@univie.ac.at](mailto:Arnold.Neumaier@univie.ac.at)
if you make serious use of this code. 

License
-------

Apache 2.0


Whitepaper
----------

[Derivative-free unconstrained optimization based on QR factorizations by A. Neumaier, H. Fendl, H. Schilly, and T. Leitner.](docs/vxqr1.pdf)
