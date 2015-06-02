.. image:: https://raw.githubusercontent.com/trichter/misc/master/logos/logo_qopen.png
   :width: 90 %
   :alt: Qopen
   :align: center

Separation of intrinsic and scattering **Q** by envel\ **ope** inversio\ **n**

| **Author**: Tom Eulenfeld
| **License**: MIT
| **Project page**: https://github.com/trichter/qopen
| **Pypi page**: https://pypi.python.org/pypi/qopen
| **Test status**: |buildstatus|

.. |buildstatus| image:: https://api.travis-ci.org/trichter/qopen.png?
    branch=master
   :target: https://travis-ci.org/trichter/qopen

Qopen is a script in seismology that determines shear wave scattering and intrinsic attenuation parameters by inversion of seismogram envelopes.

How it works
------------

The method will be described in an upcoming publication. Please read our `DGG 2015 poster`__ in the meantime.

.. __: http://www.eulenf.de/publications/richter2015_DGG_attenuation_at_geothermal_sites.pdf


How to use it
-------------

Installation
............

Dependencies of Qopen are:

* ObsPy>=0.10
* NumPy and SciPy>=0.11 (itself dependencies of ObsPy)
* statsmodels
* joblib (optional for multi-core support)

It's probably best to install ObsPy first and then let pip take care of the rest. Install Qopen and its dependencies with ::

    pip install qopen

Qopen provides the two scripts `qopen` and `qopen-runtests`.
The installation can be tested with the second script::

    qopen-runtests

Tutorial
........

The code is run by the installed command line script `qopen`. A tutorial can be created with the appropriate flag::

    qopen --tutorial

This command copies an example configuration file in JSON format and the corresponding data files into the current directory. The configuration file is heavily commented and should be rather self-explanatory. Now you can perform the inversion by simply running ::

    qopen

which will calculate the results and create different plots.

Use your own data
.................

To use the script with your own data you need 1. an inventory (StationXML or other ObsPy readable format) of your stations, 2. the earthquake catalog (QuakeML or other ObsPy readable format) preferable with P and S picks and 3. the waveforms. Waveforms may exist in data files of various formats or can be fetched from a webservice. A custom solution for waveform retrieval is also possible (e.g. mixing of data files and web requests). An example configuration file can be created with ::

    qopen --create-config

This file has to be adopted to your needs (time window selection, etc.). The inversion is started by simply running `qopen` again.
