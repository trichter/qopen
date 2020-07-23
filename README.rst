.. image:: https://raw.githubusercontent.com/trichter/misc/master/logos/logo_qopen.png
   :width: 90 %
   :alt: Qopen
   :align: center

Separation of intrinsic and scattering **Q** by envel\ **ope** inversio\ **n**

|buildstatus| |coverage| |version| |pyversions| |zenodo|

.. |buildstatus| image:: https://api.travis-ci.org/trichter/qopen.svg?
    branch=master
   :target: https://travis-ci.org/trichter/qopen

.. |coverage| image:: https://codecov.io/gh/trichter/qopen/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/trichter/qopen

.. |version| image:: https://img.shields.io/pypi/v/qopen.svg
   :target: https://pypi.python.org/pypi/qopen

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/qopen.svg
   :target: https://python.org

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3953654.svg
   :target: https://doi.org/10.5281/zenodo.3953654


Qopen is a script in seismology that estimates shear wave scattering and intrinsic attenuation parameters by inversion of seismogram envelopes.

How it works
------------

The method is described in the following publication:

Tom Eulenfeld and Ulrich Wegler (2016), Measurement of intrinsic and scattering attenuation of shear waves in two sedimentary basins and comparison to crystalline sites in Germany, *Geophysical Journal International*, 205(2), 744-757, doi:`10.1093/gji/ggw035`__. `[pdf]`_

Alternatively, have a look at our poster_ presented at the 2015 Annual Meeting of the DGG.

.. __: https://dx.doi.org/10.1093/gji/ggw035

.. _`[pdf]`: https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00038348/Eulenfeld_Wegler_2016_Intrinsic_and_scattering_attenuation_a.pdf

.. _poster: https://dx.doi.org/10.6084/m9.figshare.2074693

How to use it
-------------

Installation
............

Since version 2, Qopen runs on Python3 only. If you want to use Qopen with Python2
use version 1.x.

Dependencies of Qopen are:

* ObsPy>=1.0
* NumPy, SciPy and matplotlib (itself dependencies of ObsPy)
* statsmodels

Installation with conda -
it's probably best to install ObsPy and other dependencies first and then let pip take care of the rest::

    conda --add channels conda-forge
    conda create -n qenv matplotlib numpy scipy obspy statsmodels
    conda activate qenv
    pip install qopen

Qopen provides the scripts `qopen`, `qopen-rt` and `qopen-runtests`.
The installation can be tested with::

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

This file has to be adapted to your needs (time window selection, etc.). The inversion is started by simply running `qopen` again.

Use Qopen in Python scripts
...........................

To call Qopen from Python do e.g. ::

    from qopen import run
    run(conf='conf.json')

All configuration options in `conf.json` can be overwritten by keyword
arguments passed to `run()`.

Use Qopen to determine coda Q
.............................

Qopen can be "abused" to determine a mean coda Q with the diffusion approximation with the following settings in conf.json::

    "optimize": null,
    "bulk_window": null,
    "G_plugin": "qopen.rt : G_diffapprox3d",
    "seismic_moment_method": null,

The scattering coefficient and event spectra are meaningless with these settings. Qi corresponds to Qc in this case. For the single scattering approximation use a user-defined Green's function.

Use Qopen with coda normalization
.................................

For comparison, Qopen can be used with coda normalization with the following settings in conf.json::

    "coda_normalization": [180, 200],
    "seismic_moment_method": null,

Of course, site amplifications and event spectra are useless in this case.

Get help and discuss
--------------------

Please consult the `API documentation`_.

The Obspy forum can be used to contact other users and developers. Please post the topic in the `ObsPy Related Projects category <https://discourse.obspy.org/c/obspy-related-projects>`_.

A somewhat advanced example using the Qopen package: `USAttenuation <https://github.com/trichter/usattenuation>`_.

These studies make use of Qopen: `Google Scholar Link <https://scholar.google.com/scholar?cites=2976023441381045818&scipsc=1&q=Qopen>`_.

.. _`API documentation`: https://qopen.readthedocs.io