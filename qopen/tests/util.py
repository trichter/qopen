# Copyright 2015-2017 Tom Eulenfeld, MIT license

import contextlib
import os
import shutil
import sys
import tempfile


class _Devnull(object):

    def write(self, _):
        pass


@contextlib.contextmanager
def quiet(verbose=False):
    if not verbose:
        stdout_save = sys.stdout
        sys.stdout = _Devnull()
    try:
        yield
    finally:
        if not verbose:
            sys.stdout = stdout_save


def _replace_in_file(fname_src, fname_dest, str_src, str_dest):
    with open(fname_src) as f:
        text = f.read()
    text = text.replace(str_src, str_dest)
    with open(fname_dest, 'w') as f:
        f.write(text)


@contextlib.contextmanager
def tempdir(tempdirname=None, delete=False):
    if tempdirname is None:
        tempdir = tempfile.mkdtemp(prefix='qopen_test_')
    else:
        tempdir = os.path.join(tempfile.gettempdir(), tempdirname)
        if os.path.exists(tempdir) and delete:
            shutil.rmtree(tempdir)
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)
    cwd = os.getcwd()
    os.chdir(tempdir)
    # for coverage put .coveragerc config file into tempdir
    # and append correct data_file parameter to config file
    covfn = os.path.join(cwd, '.coverage')
    if not os.path.exists('.coveragerc') and os.path.exists(covfn + 'rc'):
        _replace_in_file(covfn + 'rc', '.coveragerc', '[run]',
                         '[run]\ndata_file = ' + covfn)
    try:
        yield tempdir
    finally:
        os.chdir(cwd)
        if tempdirname is None and os.path.exists(tempdir):
            try:
                shutil.rmtree(tempdir)
            except PermissionError as ex:
                print('Cannot remove temporary directory: %s' % ex)
