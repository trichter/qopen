# Author: Tom Richter

from setuptools import find_packages, setup

VERSION = '0.1.0-dev'
with open('README.rst') as f:
    README = f.read()
if not 'dev' in VERSION:  # get image for correct version from travis-ci
    README = README.replace('branch=master', 'branch=v%s' % VERSION)
DESCRIPTION = 'Separation of intrinsic and scattering Q by envelope inversion'

ENTRY_POINTS = {
    'console_scripts': ['qopen-runtests = qopen.tests:run',
                        'qopen = qopen.core:run_cmdline',
                        'qopen-rt = qopen.rt:main']}

setup(name='qopen',
      version=VERSION,
      description=DESCRIPTION,
      long_description=README,
      url='https://github.com/trichter/qopen',
      author='Tom Richter',
      author_email='tom.richter@bgr.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[#'obspy >= 0.10',
                        'statsmodels', 'joblib'],
      entry_points=ENTRY_POINTS,
      include_package_data=True,
      zip_safe=False
      )