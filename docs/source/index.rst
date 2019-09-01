simpycirc
=====================================

Simpycirc is a lightweight package for quickly and conveniently building linear electrical circuits.

Installation is as simple as::

    pip install simpycirc

Circuits can be invoked using SPICE style netlists or by simply encapsulating a written circuit in a decorated function.

.. code:: python

    from simpycirc import simulator

    @simulator
    def my_circuit():
        return R_1 + (R_2 + C_1)

    my_circuit(R_1=100, R_2=200, C_1=100e-9)

.. toctree::
   :maxdepth: 2

   circuits



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
