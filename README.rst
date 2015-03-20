.. image:: https://raw.githubusercontent.com/trichter/misc/master/logos/logo_qopen.png
   :width: 90 %
   :alt: Qopen
   :align: center

Separation of intrinsic and scattering **Q** by envel\ **ope** inversio\ **n**

| **Author**: Tom Richter
| **License**: MIT
| **Project page**: https://github.com/trichter/qopen
| **Test status**: |buildstatus|

.. |buildstatus| image:: https://api.travis-ci.org/trichter/qopen.png?
    branch=master
   :target: https://travis-ci.org/trichter/qopen

Qopen is a script in seismology that determines scattering and intrinsic attenuation parameters by inversion of seismogram envelopes.

Installation
------------

Dependencies of qopen are:

* ObsPy>=0.10
* NumPy and SciPy>=0.11 (itself dependencies of ObsPy)
* statsmodels
* joblib (optional for multi-core support)

It's probably best to install ObsPy first and then let pip take care of the rest. Qopen is not yet released. Install the latest development version and its dependencies with ::

    pip install https://github.com/trichter/qopen/archive/master.zip

Test the installation with the script ::

    qopen-runtests

Usage
-----

The code is run by the installed command line script `qopen`. For a start, try to run the tutorial. The tutorial files are created in the current directory with ::

    qopen --tutorial

This command creates an example configuration file in JSON format and the corresponding data files. Have look at the configuration file, its options and comments. Now you can perform the inversion by simply running ::

    qopen

which will calculate the results and create different plots. For a new project just create a new configuration file and change it to your requirements.

How it works
------------

The method will be described in an upcoming publication. In the meantime you can read our `DGG 2015 poster`__.

.. __: http://www.eulenf.de/publications/richter2015_DGG_attenuation_at_geothermal_sites.pdf
