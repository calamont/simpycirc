.. image:: docs/images/logo.png
    :width: 400
    :align: center

circuitlib is a lightweight package for quickly and conveniently simulating linear electrical circuits. It is written exclusively in Python and only requires in SciPy, Numpy and matplotlib.

Circuits can be invoked using SPICE style netlists or by simply decorating a circuit described in a function.

.. code:: python

    import numpy as np
    from circuitlib import Circuit, Netlist, populate

    freq = np.logspace(-2,5,100)

    @populate(freq)
    def hp_filter(V1=1, R_load=1e12):
        return V1 + C1 + (R1 | R_load)

    lp_filter = Netlist(freq)
    lp_filter.V1 = (0, 1), 1
    lp_filter.R1 = (1, 2), 1e4
    lp_filter.C1 = (2, 0), 100e-9
    lp_filter.R_load = (2, 0), 1e12

    highpass = Circuit(hp_filter)
    lowpass = Circuit(lp_filter)

    ax = highpass.bode(ax='return')
    ax = lowpass.bode(ax=ax)

Installation
------------
circuitlib can be installed with pip:

.. code:: bash

    $ pip install circuitlib


Or you can grab the source code from GitHub:

.. code:: bash

    $ git clone git://github.com/circuitlib/circuitlib.git
    $ python setup.py installation

Next steps
----------
Check out the user guide to get a flavour of the main features of circuitlib!

.. toctree::
   :maxdepth: 2

   installation
   circuits
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
