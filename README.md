<p align="center">
<img src="docs/images/logo.png" width="400">
</p>


# Getting started

Circuitlib is simple package for simulating electrical circuits. Circuits can be constructed <img align="right" src="https://img.shields.io/github/last-commit/calamont/circuitlib"><br/>
in functions or using traditional SPICE style netlists. Circuitlib leverages the power<img align="right" src="https://img.shields.io/github/license/calamont/circuitlib"><br/>
of NumPy and Matplotlib for circuit analysis. And that's it. Look how easy it is...

```python

import numpy as np
import circuitlib as clb

freq = np.logspace(-2,5,100)

@clb.NodalAnalysis(freq)
def hp_filter(V1, C=100e-12, R=1000):
    return V + C + R
    
fra = clb.FrequencyAnalysis(hp_filter)
ax = fra.bode()
ax = fra.bode(C=400e-12, ax=ax)
```
