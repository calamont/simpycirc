.. image:: docs/images/logo.png
    :width: 400
    :align: center

Circuitlib is a for quickly simulating linear electrical circuits. It is written exclusively in Python and only requires NumPy and Matplotlib. Circuits can be invoked using SPICE style netlists or by simply decorating a circuit described in a function. Look how easy it is...

.. code:: python

    import numpy as np
    import circuitlib as clb

    freq = np.logspace(-2,5,100)

    @clb.NodalAnalysis(freq)
    def highpass_filter(V=1, C=100e-12, R=1000):
        return V + C + R

    fra = clb.FrequencyAnalysis(highpass_filter)
    ax = fra.bode()
    ax = fra.bode(C=400e-12, ax=ax)


.. image:: docs/images/highpass_filter.png
    :width: 400
    :align: center

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
