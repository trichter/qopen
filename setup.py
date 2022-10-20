# Copyright 2015-2017 Tom Eulenfeld, MIT license
import os
import re

from setuptools import find_packages, setup


def find_version(*paths):
    fname = os.path.join(os.path.dirname(__file__), *paths)
    with open(fname) as fp:
        code = fp.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", code, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

version = find_version('qopen', '__init__.py')

DESCRIPTION = 'Separation of intrinsic and scattering Q by envelope inversion'
LONG_DESCRIPTION = 'Please look at the project site for more information.'

ENTRY_POINTS = {
    'console_scripts': ['qopen-runtests = qopen.tests:run',
                        'qopen = qopen.core:run_cmdline']}

DEPS = ['matplotlib>=1.3', 'numpy>=1.8', 'scipy>=0.14',
        'setuptools', 'obspy>=1.0', 'statsmodels']

CLASSIFIERS = [
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering :: Physics'
    ]

setup(name='qopen',
      version=version,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url='https://github.com/trichter/qopen',
      author='Tom Eulenfeld',
      author_email='tom.eulenfeld@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=DEPS,
      entry_points=ENTRY_POINTS,
      include_package_data=True,
      zip_safe=False,
      classifiers=CLASSIFIERS
      )
