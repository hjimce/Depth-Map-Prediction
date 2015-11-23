'''
logutil.py

utilities for logging, tracking experiment runs
'''
'''
Copyright (C) 2014 New York University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import os
import time
import logging
import subprocess
import shutil
import numpy as np
from PIL import Image

from __builtin__ import open as _open

try:
    from matplotlib import pyplot
    _have_plot = True
except ImportError:
    _have_plot = False

try:
    import IPython
    _ipython_app = IPython.Application.instance()
    _ipython_logger = _ipython_app.shell.logger
except (ImportError, AttributeError):
    _ipython_app = None
    _ipython_logger = None

class _Config(object):
    log_file = True
    log_console = True
    output_dir = None
    ipython_logfname = None

_config = _Config()

_log = logging.getLogger()
_log.setLevel(logging.INFO)

def _setup_logs():
    # setup python logger
    handlers = list(_log.handlers)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
    for h in handlers:
        _log.removeHandler(h)
    if _config.log_console:
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        _log.addHandler(h)
    if _config.log_file and _config.output_dir:
        h = logging.FileHandler(filename('log'))
        h.setFormatter(fmt)
        _log.addHandler(h)

    # setup ipython session history
    iplogger = _ipython_logger
    if iplogger:
        if iplogger.log_active and \
           iplogger.logfname != _config.ipython_logfname:
            # user turned on logging to their own file
            _config.ipython_logfname = None
        else:
            if iplogger.log_active:
                iplogger.logstop()
            if _config.output_dir:
                _config.ipython_logfname = filename('ipython_log.py')
                iplogger.logstart(_config.ipython_logfname,
                                  log_output=True,
                                  timestamp=True)

_setup_logs()

class Subdir(object):
    '''
    Atomically swappable/recoverable subdirectory
    '''
    def __init__(self, name):
        self.name = name
        self.current = name
        self.next = self.current + '.next'
        self.recover()

    def create_next(self):
        try:
            os.mkdir(filename(self.next))
        except OSError, ex:
            if ex.errno != os.errno.EEXIST:
                raise

    def swap(self):
        curr = filename(self.current)
        next = curr + '.next'
        prev = curr + '.prev'

        if os.path.exists(prev):
            shutil.rmtree(prev)
        if os.path.exists(curr):
            os.rename(curr, prev)

        os.rename(next, curr)

        try:
            if os.path.exists(prev):
                shutil.rmtree(prev)
        except (OSError, IOError):
            _log.warn('Error removing prev state dir')
            _log.exception()

    def recover(self):
        curr = filename(self.current)
        prev = curr + '.prev'
        if not os.path.exists(curr) and os.path.exists(prev):
            _log.info('Recovering state from %s' % prev)
            os.rename(prev, curr)

class consistent_dir(object):
    '''
    Checks a directory remains the same (not swapped) while used and
    between uses.  For use in with statement.
    '''

    _dir_inums = {}

    def __init__(self, dirname):
        self.dirname = dirname

    def __enter__(self):
        name = os.path.abspath(self.dirname)
        if name not in self._dir_inums:
            inum = os.stat(name).st_ino
            self._dir_inums[name] = inum

    def __exit__(self, *args):
        name = os.path.abspath(self.dirname)
        inum = os.stat(name).st_ino
        if self._dir_inums[name] != inum:
            raise IOError('Directory changed while reading files: %s'
                          % self.dirname)

def set_output_dir(dirname):
    '''
    Set the current directory for logging and output.
    '''
    assert os.path.exists(dirname)
    _config.output_dir = dirname
    _setup_logs()

def filename(fn):
    '''
    Returns a path for the given filename in the current output directory.
    '''
    if _config.output_dir:
        return os.path.join(_config.output_dir, fn)
    else:
        return fn

def output_dir():
    return _config.output_dir if _config.output_dir else '.'

def getLogger():
    return _log

def open(fn, *args, **kwargs):
    '''
    Open a file in the current output directory
    args same as for open()
    '''
    return _open(filename(fn), *args, **kwargs)

def copy(src, dst=None):
    '''
    Copy a file to the output directory.

    If dst is None, uses basename(src).  Otherwise, dst is the name of the
    file within the current output directory.
    '''
    if dst is None:
        dst = os.path.basename(src)
    dst = filename(dst)
    if os.path.realpath(src) != os.path.realpath(dst):
        shutil.copy(src, dst)

def save_image(fn, img, **kwargs):
    '''
    Save an image img to filename fn in the current output dir.
    kwargs the same as for PIL Image.save()
    '''
    (h, w, c) = img.shape
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if c == 1:
        img = np.concatenate((img,)*3, axis=2)
    if img.dtype.kind == 'f':
        img = (img * 255).astype('uint8')
    elif img.dtype.kind == 'f':
        img = img.astype('uint8')
    else:
        raise ValueError('bad dtype: %s' % img.dtype)
    i = Image.fromarray(img)
    with open(fn, 'w') as f:
        i.save(f, **kwargs)

def save_fig(fn, *args, **kwargs):
    '''
    Save a matplotlib figure to fn in the current output dir.
    args same as for pyplot.savefig().
    '''
    with open(fn, 'w') as f:
        pyplot.savefig(f, *args, **kwargs)

