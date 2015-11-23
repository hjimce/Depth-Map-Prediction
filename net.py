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
import sys
import time
import numpy as np
import ipdb
import cPickle

from collections import OrderedDict

import theano, theano.tensor as T
from theano.tensor.nnet import conv as theano_conv
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

from common import imgutil, logutil, configuration

#import matplotlib.pyplot as plt

import pooling
import thutil

from thutil import test_shape, theano_function, maximum

_log = logutil.getLogger()

floatX = theano.config.floatX

theano.config.compute_test_value = 'raise'
theano.config.store_test_value_maxsize = 32
theano.config.on_unused_input = 'ignore'

# to enable feature not yet in theano main for logicals as float32 on gpu
# theano.config.scalar.logical_op_type = 'same_as_input'

theano_rng = theano.tensor.shared_randomstreams.RandomStreams()

xx = np.newaxis

### Math and nnet util functions ###

def relu(x):
    return maximum(0, x)

def softmax(x, axis=None):
    '''
    Applies softmax to x over the given axis (i.e. exp/sum(exp)).
    '''
    if isinstance(axis, int):
        m = T.max(x, axis=axis, keepdims=True)
    else:
        m = T.max(x)
    exp_x = T.exp(x - m)
    Z = T.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / Z

def logsoftmax(x, axis=None):
    '''
    Applies logsoftmax to x over the given axis (i.e. exp/sum(exp)).
    '''
    if isinstance(axis, int):
        m = T.max(x, axis=axis, keepdims=True)
    else:
        m = T.max(x)
    exp_x = T.exp(x - m)
    Z = T.sum(exp_x, axis=axis, keepdims=True)
    return x - m - T.log(Z)

_mm_enable_compatibility_padding = True

def conv_theano_mm(x, k, border_mode, transpose=False, stride=1):
    '''
    Convolves images x with filters k.
    x has shape (bsize, xchan, h, w)
    k has shape (nfilt, xchan, filt_h, filt_w)
    '''
    (xh, xw) = test_shape(x)[-2:]
    (kh, kw) = test_shape(k)[-2:]

    if border_mode == 'valid':
        pad = (0,0)
    elif border_mode == 'same':
        pad = (kh // 2, kw // 2)
    elif border_mode == 'full':
        pad = (kh - 1, kw - 1)
    else:
        raise ValueError(border_mode)

    if stride != 1 and not transpose and _mm_enable_compatibility_padding:
        # semi-compatibility with cudaconv
        # cudaconv strided convs go one filter tile past the end at the
        # bottom/right.  Get the same size with some extra padding if needed.
        # The padding is centered, so this results in up to a half-stride image
        # shift to the right, not exactly the same as before.
        if border_mode != 'valid':
            raise NotImplementedError()
        old_h = np.ceil((xh - kh) / float(stride)) * stride + kh
        old_w = np.ceil((xw - kw) / float(stride)) * stride + kw
        pad = (int(np.ceil((old_h - xh) / 2.0)),
               int(np.ceil((old_w - xw) / 2.0)))

    if transpose:
        (ph, pw) = pad
        bottom_shape = T.constant(np.array((stride * (xh - 1) - 2*ph + kh,
                                            stride * (xw - 1) - 2*pw + kw)))
        res = theano.sandbox.cuda.blas.GpuCorrMM_gradInputs(
                        pad=pad,
                        subsample=(stride, stride)) \
                    (k, x, shape=bottom_shape)
    else:
        res = theano.sandbox.cuda.blas.GpuCorrMM(
                        pad=pad,
                        subsample=(stride, stride)) \
                    (x, k)
    return res

conv = conv_theano_mm

def upsample_bilinear(x, scale):
    '''
    Bilinearly upsamples x:
    (nimgs, nfeat, h, w) -> (nimgs, nfeat, h*scale, w*scale)
    '''
    kx = np.linspace(0, 1, scale + 1)[1:-1]
    kx = np.concatenate((kx, [1], kx[::-1]))
    ker = kx[xx,:] * kx[:, xx]
    ker = T.constant(ker[xx,xx,:,:].astype(np.float32))
    xbatch = x.reshape((x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]))
    xup = conv(xbatch, ker, 'valid', transpose=True, stride=scale)
    return xup.reshape((x.shape[0], x.shape[1], xup.shape[2], xup.shape[3]))

def filter_transpose(w):
    '''
    Transposes and filps a set of filters.
    (output_maps, input_maps, h, w) -> (input_maps, output_maps, h, w)
    and each filter is rotated by 180deg in (h, w).
    '''
    return w.transpose((1,0,2,3))[:,:,::-1,::-1]

_conv_mode_transpose = {'valid': 'full', 'full': 'valid', 'same': 'same'}

def random_zero(x, p):
    '''
    Keeps 1-p entries of x and zeros out a random subset with prob p
    '''
    return x * theano_rng.binomial(size=x.shape,
                                   n=1,
                                   p=1-p,
                                   dtype=x.dtype)

def feature_map_vectors(x):
    '''
    Transpose/Reshape feature maps into (bsize*ni*nj, #feature maps)
    '''
    (bsize, nc, ni, nj) = x.shape
    return x.transpose((0,2,3,1)).reshape((bsize*ni*nj, nc))

def feature_map_maps(x, xshape):
    '''
    Transpose/Reshape feature map vectors back to xshape == (bsize, nc, ni, nj)
    '''
    (bsize, nc, ni, nj) = xshape
    return x.reshape((bsize, ni, nj, nc)).transpose((0,3,1,2))


### Machine class for tracking training state etc. ###

_unit_types = {}
def register_unit_class(cls):
    typename = getattr(cls, 'type', cls.__name__.lower())
    _unit_types[typename] = cls
    return cls

class Machine(object):
    def __init__(self, conf, state_subdir_name='state', **kwargs):
        self.conf = conf
        self.bsize = self.conf.getint('train', 'bsize')
        self.state_dir = logutil.Subdir(state_subdir_name)
        self.units = []
        self.define_machine(**kwargs)

    def create_unit(self, sec, cls=None, name=None, load_key=None, **kwargs):
        conf_sec = self.conf.get_section(sec)
        if cls is None:
            cls = _unit_types[conf_sec.get('type')]
        if name is None:
            name = sec
        if load_key is None:
            load_key = conf_sec.get('load_key', name)
        kwargs['name'] = name
        kwargs['load_key'] = load_key
        kwargs['machine'] = self
        unit = cls(conf_sec, **kwargs)
        self.units.append(unit)
        return unit

    def define_machine(self):
        raise NotImplementedError


class MachinePart(object):
    __slots__ = ('vars',)

    def __init__(self, vars, exclude=('self',)):
        self.vars = dict((k,v) for (k,v) in vars.iteritems()
                                if k not in exclude)

    def __getattr__(self, k):
        if k in self.vars:
            return self.vars[k]
        return self.__getattribute__(k)

    def __getitem__(self, k):
        return getattr(self, k)

    def __setattr__(self, k, v):
        if k in self.__slots__:
            object.__setattr__(self, k, v)
        self.vars[k] = v

    def __setitem__(self, k, v):
        return setattr(self, k, v)


def import_module(mod_file, modpath=''):
    import importlib
    (fpath, fname) = os.path.split(mod_file)
    (modname, ext) = os.path.splitext(fname)
    modpath = os.path.join(modpath, fpath)
    sys.path.insert(0, modpath)
    try:
        mod = importlib.import_module(modname, modpath)
    finally:
        sys.path.remove(modpath)
    assert (os.path.realpath(os.path.dirname(mod.__file__)) ==
                os.path.realpath(modpath)), 'module path does not match'
    return mod


def create_machine(module_fn, config_fn, params_dir=None,
                   edit_conf=None, load_saved_params=True):
    
    # get configuration
    conf = configuration.read_config(config_fn)
    conf.set_eval_environ(section='config')

    # edit conf to load params from load dir
    if load_saved_params:
        assert params_dir, 'must supply params dir'
        if not conf.has_section('load'):
            conf.add_section('load')
        conf.set('load', 'all', params_dir)

    # user-supplied config edits
    if edit_conf:
        edit_conf(conf)

    # load definition module
    mod = import_module(module_fn)

    # construct machine class
    machine = getattr(mod, 'machine')(conf)

    return machine
    

### Units with parameters and inference methods ###

class Unit(object):
    def __init__(self, conf, name, load_key=None, machine=None, tie_params={}):
        self.conf = conf
        self.name = name
        self.load_key = load_key
        self.params = None
        self.grads = None
        self.constraints = {}
        self.tie_params = tie_params
        self.machine = machine

    def infer(self, x):
        raise NotImplementedError

    def add_constraint(self, param, constraint):
        if param in self.constraints:
            prev = self.constraints[param]
            self.constraints[param] = lambda x: constraint(prev(x))
        else:
            self.constraints[param] = constraint

    def _params_filename(self):
        return 'params-%s.pk' % self.name

    def _check_file(self, dir, fn, check_state_dir=True):
        if dir is None:
            return None
        fpaths = [os.path.join(dir, fn)]
        if check_state_dir:
            fpaths.append(os.path.join(dir, 'state', fn))
        for fpath in fpaths:
            if os.path.exists(fpath):
                return fpath
        return None

    def init_params(self, *args, **kwargs):
        '''
        Initializes parameters, either from a file or from initialization code
        for the unit.  This looks for parameters to use in the following
        order (highest precedence first):

        * load overrides for debug and interactive sessions
            1. params_file in unit config
            2. load_key in [load] config section
            3. default load dir ("all" in [load] config section)

        * params saved during training, loaded when resuming a run
            4. current training state in output
            5. current output directory

        * initializations, loaded once nothing was found for resuming
            6. load_key in [init] config section
            7. default init dir ("all" in [init] config section)

        * initialize by calling unit init code (since no was file specified)
            8. call unit _init_params()
        '''
        params_dir = None
        params_file = None
        fn = self._params_filename()

        # first check if a file is explicitly specified in unit config
        # if so, use it (even if it doesn't exist -- that case should error)
        case = 'in_config'
        params_file = self.conf.get('params_file', None)

        # if not, look in the dir for the load key specified for this unit
        if self.conf.parent.has_section('load'):
            if params_file is None and self.load_key is not None:
                case = 'load_key'
                params_dir = self.conf.parent.get('load', self.load_key, None)
                params_file = self._check_file(params_dir, fn)

            # then check in the default load dir
            if params_file is None:
                case = 'load_default'
                params_dir = self.conf.parent.get('load', 'all', None)
                params_file = self._check_file(params_dir, fn)

        # check current training state and output dir if the run is resumptive
        if self.conf.parent.getboolean('train', 'resumptive', True):
            if params_file is None:
                case = 'resume_current'
                params_dir = logutil.filename(self.machine.state_dir.current)
                params_file = self._check_file(params_dir, fn,
                                               check_state_dir=0)

            if params_file is None:
                case = 'resume_current'
                params_dir = logutil.filename(logutil.output_dir())
                params_file = self._check_file(params_dir, fn,
                                               check_state_dir=0)

        # next, look for initializations by key, then default init
        if self.conf.parent.has_section('init'):
            if params_file is None and self.load_key is not None:
                case = 'init_key'
                params_dir = self.conf.parent.get('init', self.load_key, None)
                params_file = self._check_file(params_dir, fn)

            if params_file is None:
                case = 'init_default'
                params_dir = self.conf.parent.get('init', 'all', None)
                params_file = self._check_file(params_dir, fn)

        # if we did not find a params file, init with _init_params()
        if params_file is None:
            case = 'none'

        kwargs['tie_params'] = self.tie_params
        for (k, x) in self.tie_params.iteritems():
            setattr(self, k, x)

        # load the params file, if we found one
        if params_file is not None:
            assert case != 'none'
            self.load_params(params_file)
            self.loaded = case in ('in_config', 'load_key', 'load_default')
            self.resumed = case in ('resume_current',)
            self.init_from_load = case in ('init_key', 'init_default')
        else:
            self.params = []
            self._init_params(*args, **kwargs)
            self.loaded = False
            self.resumed = False
            self.init_from_load = False

    def _save_params(self, dir=None, fn=None, attrs=[]):
        if fn is None:
            fn = self._params_filename()
        if dir:
            fn = os.path.join(dir, fn)
        pdict = dict((x, getattr(self, x)) for x in attrs)
        if self.params:
            pdict.update((p.name, p) for p in self.params)
            pdict['params'] = [p.name for p in self.params]
        with logutil.open(fn, 'w') as f:
            cPickle.dump(pdict, f, cPickle.HIGHEST_PROTOCOL)

    def _load_params(self, fn):
        _log.info('Loading parameters from %s' % fn)
        with logutil.consistent_dir(os.path.dirname(fn)):
            with open(fn, 'r') as f:
                pdict = cPickle.load(f)
        params = pdict.pop('params', [])
        for (name, value) in pdict.iteritems():
            setattr(self, name, value)
        self.params = [pdict[x] for x in params]

    save_params = _save_params
    load_params = _load_params

    def get_updates(self, cost, learning_rate, momentum):
        if not self.params:
            self.learning_rate = T.constant(0)
            return {}

        if self.grads is None:
            self.grads = [theano.shared(np.zeros_like(p.get_value()))
                          for p in self.params]

        # compute the gradients of the cost with respect to the parameters
        gparams = T.grad(cost, self.params, disconnected_inputs='ignore')
        grad_mult = self.conf.geteval('grad_mult', None)
        if grad_mult is not None:
            grad_mult = T.constant(grad_mult, dtype=floatX)
            gparams = [g * grad_mult for g in gparams]

        clip = self.conf.getfloat('grad_clip', None)
        if clip is not None:
            gparams = [T.clip(g, -clip, clip) for g in gparams]

        self.gparams = gparams

        # generate the list of updates
        gupdates = OrderedDict()
        pupdates = OrderedDict()

        self.learning_rate = self.conf.getfloat('learning_rate', None)
        if self.learning_rate:
            self.learning_rate = T.constant(self.learning_rate)
        else:
            self.learning_rate = learning_rate
        for (gparam, param, gold) in zip(gparams, self.params, self.grads):
            lrscale = self.conf.getfloat(
                                'learning_rate_scale_%s' % param.name,
                                None)
            if lrscale is None:
                lrscale = self.conf.getfloat('learning_rate_scale', 1.0)
            decay = self.conf.getfloat('weight_decay_%s' % param.name, 0.0)

            lr = self.learning_rate
            if lrscale != 1.0:
                lr *= lrscale

            if decay:
                gparam += decay * param

            if momentum:
                gnew = momentum * gold + gparam
                gupdates[gold] = gnew
                pupdates[param] = param - lr * gnew
            else:
                gupdates[gold] = gparam
                pupdates[param] = param - lr * gparam

        # apply update constraints
        for (p, constraint) in self.constraints.iteritems():
            pupdates[p] = constraint(pupdates[p])

        return OrderedDict(gupdates.items() + pupdates.items())


@register_unit_class
class MaxPool(Unit):
    def __init__(self, conf, **kwargs):
        Unit.__init__(self, conf, **kwargs)
        self.conf = conf
        self.vis_shape = kwargs.get('vis_shape', None)
        self.poolsize = self.conf.geteval('poolsize', None)
        self.poolstride = self.conf.geteval('poolstride', None)

    def pool(self, y):
        '''apply pooling to unpooled output'''
        if self.vis_shape is None:
            self.vis_shape = test_shape(y)[-2:]
        (p_y, p_inds) = pooling.maxpool2d(y, winsize=self.poolsize,
                                             stride=self.poolstride)
        return (p_y, p_inds)

    infer = pool

    def unpool(self, y, inds):
        '''unpool pooled output'''
        y = pooling.index_unpool_2d(y, inds,
                                    winsize=self.poolsize,
                                    stride=self.poolstride,
                                    output_shape=self.vis_shape[-2:])
        return y


@register_unit_class
class SumPool(Unit):
    def __init__(self, conf, **kwargs):
        Unit.__init__(self, conf, **kwargs)
        self.conf = conf
        self.vis_shape = kwargs.get('vis_shape', None)
        self.average = self.conf.getboolean('average', False)
        self.poolsize = self.conf.geteval('poolsize', None)
        self.poolstride = self.conf.geteval('poolstride', None)

    def pool(self, y):
        '''apply pooling to unpooled output'''
        self.vis_shape = self.vis_shape or test_shape(y)[-2:]
        p_y = pooling.sumpool2d(y, winsize=self.poolsize,
                                   stride=self.poolstride,
                                   average=self.average)
        return p_y

    infer = pool

    def unpool(self, y):
        '''unpool pooled output'''
        y = pooling.sum_unpool_2d(y,
                                  winsize=self.poolsize,
                                  stride=self.poolstride,
                                  average=self.average,
                                  output_shape=self.vis_shape[-2:])
        return y


@register_unit_class
class Conv(Unit):
    def __init__(self, conf, init_W=None, **kwargs):
        Unit.__init__(self, conf, **kwargs)
        self.conf = conf
        assert self.conf.get('type') == 'conv'
        self.filter_shape = self.conf.geteval('filter_shape')
        self.conv_mode = self.conf.get('conv_mode', 'valid')

        self.transpose = self.conf.getboolean('transpose', False)
        self.have_bias = self.conf.getboolean('bias', True)
        self.stride = self.conf.getint('stride', 1)

        self.init_params(init_W)

    def _init_params(self, init_W, tie_params):
        (nfilt, fc, fi, fj) = self.filter_shape

        if 'W' not in tie_params:
            if init_W is None:
                w_shape = self.filter_shape
                init_W = self.conf.geteval('init_W')(w_shape).astype(floatX)
            self.W = theano.shared(value=init_W, name='W')
            self.params.append(self.W)

        if self.have_bias and 'b' not in tie_params:
            init_b = self.conf.geteval('init_b', 0)
            nb = nfilt if not self.transpose else fc
            self.b = theano.shared(init_b + np.zeros(nb, dtype=floatX),
                                   name='b')
            self.params.append(self.b)

    def infer(self, x):
        (nfilt, fc, fi, fj) = self.filter_shape
        if (fi, fj) == (1, 1):
            W = self.W.reshape((nfilt, fc))
            (bsize, nc, ni, nj) = x.shape
            xvec = x.transpose((1,0,2,3)).reshape((nc, bsize*ni*nj))
            if self.transpose:
                y = T.dot(W.T, xvec)
                y = y.reshape((fc, bsize, ni, nj)).transpose((1,0,2,3))
            else:
                y = T.dot(W, xvec)
                y = y.reshape((nfilt, bsize, ni, nj)).transpose((1,0,2,3))
            y = thutil.gpu_contiguous(y)
        else:
            y = conv(x, self.W, border_mode=self.conv_mode,
                                transpose=self.transpose,
                                stride=self.stride)
        if self.have_bias:
            y += self.b.reshape((1, self.b.shape[0], 1, 1))
        return y


@register_unit_class
class Full(Unit):
    def __init__(self, conf, ninput, init_W=None, **kwargs):
        Unit.__init__(self, conf, **kwargs)
        self.conf = conf
        assert self.conf.get('type') == 'full'

        self.ninput = ninput
        self.noutput = self.conf.getint('noutput')
        self.transpose = self.conf.getboolean('transpose', False)
        self.have_bias = self.conf.getboolean('bias', True)

        self.init_params(init_W)

    def _init_params(self, init_W, tie_params):
        if 'W' not in tie_params:
            if init_W is None:
                w_shape = (self.ninput, self.noutput)
                init_W = self.conf.geteval('init_W')(w_shape).astype(floatX)
            self.W = theano.shared(value=init_W, name='W')
            self.params.append(self.W)

        if self.have_bias and 'b' not in tie_params:
            nbias = self.noutput if not self.transpose else self.ninput
            init_b = self.conf.geteval('init_b', 0)
            init_b = self.conf.geteval('init_bias', init_b)
            self.bias = theano.shared(init_b + np.zeros(nbias, dtype=floatX),
                                      name='bias')
            self.params.append(self.bias)

    def infer(self, x):
        W = self.W
        if self.transpose:
            W = W.T
        y = T.dot(x, W)
        if self.have_bias:
            y += self.bias.reshape((1, self.bias.size))
        return y

