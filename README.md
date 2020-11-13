![Qopen](https://raw.githubusercontent.com/trichter/misc/master/logos/logo_qopen.png)

Separation of intrinsic and scattering **Q** by envel**ope** inversio**n**

[![buildstatus](https://api.travis-ci.org/trichter/qopen.svg?%0A%20branch=master)](https://travis-ci.org/trichter/qopen)
[![coverage](https://codecov.io/gh/trichter/qopen/branch/master/graph/badge.svg)](https://codecov.io/gh/trichter/qopen)
[![version](https://img.shields.io/pypi/v/qopen.svg)](https://pypi.python.org/pypi/qopen)
[![pyversions](https://img.shields.io/pypi/pyversions/qopen.svg)](https://python.org)
[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3953654.svg)](https://doi.org/10.5281/zenodo.3953654)

Qopen is a script in seismology that estimates shear wave scattering and
intrinsic attenuation parameters by inversion of seismogram envelopes.

## How it works

The method is described in the following publication:

Tom Eulenfeld and Ulrich Wegler (2016), Measurement of intrinsic and scattering attenuation of shear waves in two
sedimentary basins and comparison to crystalline sites in Germany, *Geophysical Journal International*, 205(2), 744-757,
doi:[10.1093/gji/ggw035](https://dx.doi.org/10.1093/gji/ggw035)
[[pdf](https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00038348/Eulenfeld_Wegler_2016_Intrinsic_and_scattering_attenuation_a.pdf)]

Alternatively, have a look at our [poster](https://dx.doi.org/10.6084/m9.figshare.2074693) presented at the
2015 Annual Meeting of the DGG.

## How to use it

#### Installation

Dependencies of Qopen are:

- ObsPy\>=1.0
- NumPy, SciPy and matplotlib (itself dependencies of ObsPy)
- statsmodels

Installation with conda - it's probably best to install ObsPy and other
dependencies first and then let pip take care of the rest:

    conda --add channels conda-forge
    conda create -n qenv matplotlib numpy scipy obspy statsmodels
    conda activate qenv
    pip install qopen

Qopen provides the scripts qopen and qopen-runtests. The installation can be tested with:

    qopen-runtests

#### Tutorial

The code is run by the installed command line script qopen. A tutorial can be created with:

    qopen create --tutorial

This command copies an example configuration file in JSON format and the
corresponding data files into the current directory. The configuration
file is heavily commented and should be rather self-explanatory. Now you
can perform the inversion by simply running :

    qopen go

which will calculate the results and create different plots.

#### Use your own data

To use the script with your own data you need 1. an inventory
(StationXML or other ObsPy readable format) of your stations, 2. the
earthquake catalog (QuakeML or other ObsPy readable format) preferable
with P and S picks and 3. the waveforms. Waveforms may exist in data
files of various formats or can be fetched from a webservice. A custom
solution for waveform retrieval is also possible (e.g. mixing of data
files and web requests). An example configuration file can be created
with:

    qopen create

This file has to be adapted to your needs (time window selection, etc.).
The inversion is started by simply running qopen again.

#### Available Qopen commands

Available Qopen commands can be displayed with `qopen -h`:

    create              Create example configuration in specified file
                        (default: conf.json)
    go                  Estimate intrinsic attenuation and scattering
                        strength, site responses, event spectra (including
                        source parameters) by inversion of envelopes
    fixed               Estimate site responses and event spectra (including
                        source parameters) with fixed attenuation parameters
                        (g0, b) by inversion of envelopes
    source              Estimate event spectra and derive source parameters,
                        e.g. moment magnitude, with fixed attenuation
                        parameters (g0, b) and fixed site responses by
                        inversion of envelopes
    recalc_source       Derive source parameters from source spectra without
                        new inversion (possibly changed configuration, e.g.
                        seismic_moment_options)
    plot                Replot results. Can be used together with -e to plot
                        event results
    rt                  Calculate or plot spectral energy densitiy Green's
                        functions, used in the above inversions, mainly based
                        on radiative transfer

#### Use Qopen in Python scripts

To call Qopen from Python do e.g.:

    from qopen import run
    run('go', conf='conf.json')

All configuration options in conf.json can be overwritten by keyword
arguments passed to run().

#### Use Qopen to determine coda Q

Qopen can be "abused" to determine a mean coda Q with the diffusion
approximation with the following settings in conf.json:

    "optimize": null,
    "bulk_window": null,
    "G_plugin": "qopen.rt : G_diffapprox3d",
    "seismic_moment_method": null,

The scattering coefficient and event spectra are meaningless with these
settings. Qi corresponds to Qc in this case. For the single scattering
approximation use a user-defined Green's function.

#### Use Qopen with coda normalization

For comparison, Qopen can be used with coda normalization with the
following settings in conf.json:

    "coda_normalization": [180, 200],
    "seismic_moment_method": null,

Of course, site amplifications and event spectra are useless in this
case.

## Get help and discuss

Please consult the [API documentation](https://qopen.readthedocs.io).

The Obspy forum can be used to contact other users and developers.
Please post the topic in the
[ObsPy Related Projects category](https://discourse.obspy.org/c/obspy-related-projects).

A somewhat advanced example using the Qopen package:
[USAttenuation](https://github.com/trichter/usattenuation).

These studies make use of Qopen:
[Google Scholar Link](https://scholar.google.com/scholar?cites=2976023441381045818&scipsc=1&q=Qopen).

## References

Initial method:
Sens-Schönfelder C and Wegler U (2006),
Radiative transfer theory for estimation of the seismic moment,
*Geophysical Journal International*, 167(3), 1363–1372.
doi:[10.1111/j.1365-246X.2006.03139.x](https://dx.doi.org/10.1111/j.1365-246X.2006.03139.x)

Enhanced Qopen method and implementation:
Eulenfeld T and Wegler U (2016),
Measurement of intrinsic and scattering attenuation of shear waves in two sedimentary basins and comparison to crystalline sites in Germany,
*Geophysical Journal International*, 205(2), 744-757,
doi:[10.1093/gji/ggw035](https://dx.doi.org/10.1093/gji/ggw035)
[[pdf](https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00038348/Eulenfeld_Wegler_2016_Intrinsic_and_scattering_attenuation_a.pdf)]

Advanced example making use of alignment of site responses:
Eulenfeld T and Wegler U (2017),
Crustal intrinsic and scattering attenuation of high-frequency shear waves in the contiguous United States,
*Journal of Geophysical Research: Solid Earth*, 122, doi:[10.1002/2017JB014038](https://dx.doi.org/10.1002/2017JB014038).
[[pdf](https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00040716/Eulenfeld_Wegler_2017_US_intrinsic_and_scattering_attenuation.pdf)]

Comparison to inversion with the help of Mote-Carlo simulations based on elastic radiative transfer theory, relating g0 to g*:
Gaebler PJ, Eulenfeld T and Wegler U (2015),
Seismic scattering and absorption parameters in the W-Bohemia/Vogtland region from elastic and acoustic radiative transfer theory,
*Geophysical Journal International*, 203 (3), 1471-1481,
doi:[10.1093/gji/ggv393](https://dx.doi.org/10.1093/gji/ggv393)
[[pdf](https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00051750/Gaebler_Eulenfeld_Wegler_Elastic_versus_acoustic_radiative_transfer_theory.pdf)]
