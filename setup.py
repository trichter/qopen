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

with open('README.rst') as f:
    README = f.read()
if 'dev' not in version:  # get image for correct version from travis-ci
    README = README.replace('branch=master', 'branch=v%s' % version)
DESCRIPTION = 'Separation of intrinsic and scattering Q by envelope inversion'
LONG_DESCRIPTION = '\n'.join(README.split('\n')[7:13])

ENTRY_POINTS = {
    'console_scripts': ['qopen-runtests = qopen.tests:run',
                        'qopen = qopen.core:run_cmdline',
                        'qopen-rt = qopen.rt:main']}

DEPS = ['future', 'matplotlib>=1.3', 'numpy>=1.7', 'scipy>=0.11',
        'setuptools', 'obspy>=1.0',
        'joblib', 'statsmodels']

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
      zip_safe=False
      )
