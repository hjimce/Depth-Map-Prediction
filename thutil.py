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
import sys
import time
import numpy as np
import operator
import types
import ipdb
import inspect
import traceback

import theano
import theano.tensor as T

from theano import Op, Apply

from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet import conv
from theano.gof import local_optimizer

from common import imgutil, logutil

_log = logutil.getLogger()

use_gpu = theano.config.device.startswith('gpu')

checkgrad = False

if use_gpu:
    from theano.sandbox.cuda import GpuOp, gpu_from_host, host_from_gpu, \
                                    CudaNdarrayType, CudaNdarray
    from theano.sandbox.cuda.basic_ops import gpu_contiguous

class Eval(object):
    def __init__(self, globals=None, locals=None):
        self.globals = globals or {}
        self.locals = locals or sys._getframe(1).f_locals

    def __getitem__(self, key):
        return eval(key, self.globals, self.locals)

def c_contiguous(x):
    if x.is_c_contiguous():
        return x
    return x.copy()

def isvalid(x):
    return T.all(T.logical_not(T.logical_or(T.isnan(x), T.isinf(x))))

def maximum(x, y):
    if checkgrad:
        return x + y
    return T.maximum(x, y)

def minimum(x, y):
    if checkgrad:
        return x + y
    return T.minimum(x, y)

def named(x, name):
    x.name = name
    return x

def test_value(x):
    if isinstance(x, np.ndarray):
        return x
    return theano.gof.op.get_test_value(x)

def test_shape(x):
    return tuple(test_value(x.shape))

def theano_function(*vars_by_pos, **kwargs):
    '''theano function decorator'''
    mode = kwargs.pop('mode', 'FAST_RUN')
    check_valid = kwargs.pop('check_valid', False)
    checks = kwargs.pop('checks', ())
    vars_by_name = kwargs
    def compile_func(f):
        argnames = f.func_code.co_varnames[:f.func_code.co_argcount]
        if any([a in vars_by_name for a in argnames[:len(vars_by_pos)]]):
            raise ValueError('Argument supplied twice to %s' % f.func_name)
        varspec = dict(vars_by_name)
        varspec.update(zip(argnames[:len(vars_by_pos)], vars_by_pos))
        argvars = []
        for name in argnames:
            spec = varspec[name]
            if isinstance(spec, (tuple, list)):
                (var, test_val) = spec
            else:
                var = spec
                test_val = None
            assert isinstance(var, T.Variable)
            var.name = name
            if test_val is not None:
                var.tag.test_value = test_val
            argvars.append(var)
        return function(argvars, f(*argvars),
                        check_valid=check_valid,
                        checks=checks,
                        mode=mode)
    return compile_func

def function(inputs, outputs=None, check_valid=False, checks=(), **kwargs):
    input_names = None
    output_names = None
    if isinstance(inputs, dict):
        if inputs:
            (input_names, inputs) = zip(*inputs.iteritems())
        else:
            (input_names, inputs) = ((), ())
    if isinstance(outputs, dict):
        if outputs:
            (output_names, outputs) = zip(*outputs.iteritems())
        else:
            (output_names, outputs) = ((), ())

    if check_valid or checks:
        updates = kwargs.setdefault('updates', {})
        asserts = [assert_(c, 'check failed: %s' % c) for c in checks]

        if check_valid:
            if outputs:
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                asserts += (assert_(isvalid(x),
                                    'output invalid: %d (%s)' % (i, x.name))
                            for (i, x) in enumerate(outputs))

            if updates:
                asserts += (assert_(isvalid(xnew),
                                    'update invalid: variable %s' % str(x))
                               for (x, xnew) in updates.iteritems())

        checks_passed = theano.shared(np.int8(1), name='checks_passed')
        updates[checks_passed] = \
            T.all(T.as_tensor_variable(asserts)).astype('int8')

        f = _CheckedFunction(inputs, outputs, **kwargs)
    else:
        f = theano.function(inputs, outputs, **kwargs)
        if hasattr(f.fn, 'clear_storage'):
            f.clear_storage = f.fn.clear_storage
        else:
            _log.warn('Function %s has no clear_storage: disabling', f.fn)
            f.clear_storage = lambda: None

    if input_names is not None or output_names is not None:
        return NamedInputOutputFunction(input_names, output_names, f)
    return f

class NamedInputOutputFunction(object):
    def __init__(self, input_names, output_names, f):
        self.input_names = input_names
        self.output_names = output_names
        self.f = f

        if output_names:
            class _NamedOutputs(object):
                __slots__ = output_names

                def __init__(self, vals):
                    [setattr(self, k, v) for (k,v) in zip(self.__slots__, vals)]
                
                def __eq__(self, other):
                    return type(self) == type(other) and \
                           self.items() == other.items()

                def __getitem__(self, k):
                    return getattr(self, k)

                def iteritems(self):
                    return ((s, self[s]) for s in self.__slots__)

                __iter__ = iteritems

                def items(self):
                    return list(self.iteritems())

            self._NamedOutputs = _NamedOutputs

        if hasattr(f.fn, 'clear_storage'):
            self.clear_storage = f.fn.clear_storage
        else:
            _log.warn('Function %s has no clear_storage: disabling', f.fn)
            self.clear_storage = lambda: None

    def __call__(self, *args, **kwargs):
        inputs = args
        if self.input_names:
            assert not inputs, \
                   'theano function with kw args cannot take positional args'
            inputs = [kwargs[k] for k in self.input_names]

        outputs = self.f(*inputs)

        if self.output_names:
            outputs = self._NamedOutputs(outputs)

        return outputs

class _CheckedFunction(object):
    def __init__(self, inputs, outputs, **kwargs):
        self.f = theano.function(inputs, outputs,
                                 inplace_updates=False,
                                 **kwargs)
        self.dbg_kwargs = dict(kwargs)
        self.dbg_kwargs.update(inputs=inputs,
                               outputs=outputs,
                               inplace_updates=False,
                               mode='DEBUG_MODE')
        self.f_dbg = None
        self.fn = self.f.fn
        self.clear_storage = self.f.fn.clear_storage

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        except AssertionError:
            _log.exception('assertion failed in function %s' % self.f.name)
            if self.f_dbg is None:
                _log.info('creating debug function for %s' % self.f.name)
                self.f_dbg = theano.function(**self.dbg_kwargs)
            _log.error('calling debug function for %s' % self.f.name)
            self.f_dbg(*args, **kwargs)
            _log.error('debug version seems to have passed' % self.f.name)
            raise

class Assert(theano.Op):
    view_map = {0: [0]}

    def __init__(self, msg=None):
        self.msg = msg

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.msg == other.msg)

    def __hash__(self):
        return reduce(operator.xor, map(hash, (type(self), self.msg)))

    def make_node(self, input):
        output = T.as_tensor_variable(input).type()
        return theano.Apply(self, (input,), (output,))

    def make_gpu_node(self, input):
        return Assert_GPU(self.msg)(input)

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def perform(self, node, (input,), (output,)):
        assert np.all(input), self.msg
        output[0] = input

    def grad(self, inputs, doutputs):
        return (None,)

def assert_(cond, msg=None):
    return Assert(msg)(cond)

class Constant(theano.Op):
    def __init__(self, ninputs):
        self.view_map = dict((i,[i]) for i in xrange(ninputs))

    def __eq__(self, other):
        return (type(self) == type(other) and
                len(self.view_map) == len(other.view_map))

    def __hash__(self):
        return reduce(operator.xor,
                      map(hash, (type(self), len(self.view_map))))

    def make_node(self, *inputs):
        outputs = tuple([T.as_tensor_variable(inp).type() for inp in inputs])
        return theano.Apply(self, inputs, outputs)

    def make_gpu_node(self, *inputs):
        return Constant_GPU(len(inputs))(*inputs)

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def perform(self, node, inputs, outputs):
        for (inp, out) in zip(inputs, outputs):
            out[0] = inp

    def grad(self, inputs, doutputs):
        return [T.DisconnectedType()() for _ in inputs]

def constant(*inputs):
    return Constant(len(inputs))(*inputs)


class _BreakpointVars(object):
    def __init__(self, th_vars, py_vars):
        self.th_vars = th_vars
        self.py_vars = py_vars

    def __getattr__(self, k):
        if k in self.th_vars:
            return self.th_vars[k]
        if k in self.py_vars:
            return self.py_vars[k]
        return object.__getattr__(self, k)
    
    def __repr__(self):
        s = []
        s.append('Theano runtime variables:')
        s += ('%-16s  %s' % (k, str(v.shape))
              for (k, v) in sorted(self.th_vars.items(), key=lambda (k,v): k))
        s.append('')
        s.append('Python creation-time variables:')
        s.append(', '.join(sorted(self.py_vars.keys())))
        s.append('')
        return '\n'.join(s)

class Breakpoint(theano.Op):
    view_map = {0: [0]}

    global_breakpoint_enable = False

    def __init__(self, var_names, cond, tb, py_vars,
                       breakpoint_grad, is_grad=False):
        self.var_names = var_names
        self.cond = cond
        self.tb = tb
        self.py_vars = py_vars
        self.nvars = len(var_names)
        self.breakpoint_grad = breakpoint_grad
        self.is_grad = is_grad

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.var_names == other.var_names and
                self.cond == other.cond and
                self.tb == other.tb)

    def __hash__(self):
        return reduce(operator.xor, map(hash, (
                   type(self), self.var_names, self.cond, self.tb)))

    def make_node(self, *inputs):
        output = T.as_tensor_variable(inputs[0]).type()
        return theano.Apply(self, inputs, (output,))

    def make_gpu_node(self, *inputs):
        return Breakpoint_GPU(
                    self.var_names, self.cond, self.tb, self.py_vars,
                    self.breakpoint_grad, self.is_grad)(*inputs)

    def infer_shape(self, node, input_shapes):
        return (input_shapes[0],)

    def perform(self, node, inputs, (output,)):
        output[0] = inputs[0]
        if not Breakpoint.global_breakpoint_enable:
            return
        x = inputs[0]
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if self.cond(x):
            vars = _BreakpointVars(
                        dict(zip(self.var_names, map(np.array, inputs[1:]))),
                        self.py_vars)
            if self.is_grad:
                place = 'theano gradient eval'
            else:
                place = 'theano eval'
            print >> sys.stderr, 'Breakpoint in %s, created at' % place
            print >> sys.stderr, '  ...'
            traceback.print_list(self.tb[-4:], sys.stderr)
            ipdb.set_trace()
            pass    # in theano breakpoint

    def grad(self, inputs, (doutput,)):
        if self.breakpoint_grad:
            doutput = Breakpoint(self.var_names, self.cond,
                                 self.tb, self.py_vars, True, True) \
                          (doutput, *inputs[1:])
        return [doutput] + [T.DisconnectedType()() for _ in xrange(self.nvars)]

_theano_types = (theano.tensor.basic.TensorConstant,
                 theano.tensor.basic.TensorVariable,
                 theano.compile.SharedVariable,
                 )

def is_theano_var(x):
    return isinstance(x, _theano_types)

def breakpoint(output, vars=None, cond=lambda v: True, grad=True):
    tb = tuple(traceback.extract_stack()[:-1])
    py_vars = {}
    if type(vars) not in (tuple, list, dict, types.NoneType):
        raise ValueError('vars keyword arg must be None, dict, list or tuple')
    if not isinstance(vars, dict):
        frame_locals = inspect.stack()[1][0].f_locals
        if vars is not None:
            frame_locals = dict((name, val)
                                for (name, val) in frame_locals.iteritems()
                                if name in vars or val in vars)
        vars = frame_locals
    assert isinstance(vars, dict)
    th_vars = dict((name, val) for (name, val) in vars.iteritems()
                               if isinstance(val, _theano_types))
    py_vars = dict((name, val) for (name, val) in vars.iteritems()
                               if name not in th_vars)
    (th_var_names, th_var_vals) = zip(*th_vars.iteritems())
    return Breakpoint(th_var_names, cond, tb, py_vars, grad) \
                     (output, *th_var_vals)

def enable_breakpoints(enable=True):
    Breakpoint.global_breakpoint_enable = enable

def cross(x, y, axis=None):
    ndim = x.ndim
    assert x.ndim == y.ndim
    if axis is None:
        axis = ndim - 1
    def _getindexslice(a, i):
        return a[tuple([slice(i,i+1) if d == axis else slice(None)
                        for d in xrange(ndim)])]
    x0 = _getindexslice(x, 0)
    x1 = _getindexslice(x, 1)
    x2 = _getindexslice(x, 2)
    y0 = _getindexslice(y, 0)
    y1 = _getindexslice(y, 1)
    y2 = _getindexslice(y, 2)

    res = T.concatenate((x1*y2 - x2*y1,
                         x2*y0 - x0*y2,
                         x0*y1 - x1*y0), axis=axis)
    return res


if use_gpu:

    class Constant_GPU(Constant, GpuOp):
        def make_node(self, *inputs):
            outputs = tuple([inp.type() for inp in inputs])
            return theano.Apply(self, inputs, outputs)

    class Assert_GPU(Assert, GpuOp):
        def make_node(self, input):
            output = input.type()
            return theano.Apply(self, (input,), (output,))

        def perform(self, node, (input,), (output,)):
            assert np.all(np.array(input))
            output[0] = input

    class Breakpoint_GPU(Breakpoint, GpuOp):
        def make_node(self, *inputs):
            output = inputs[0].type()
            return theano.Apply(self, inputs, (output,))

    @theano.sandbox.cuda.opt.register_opt()
    @theano.gof.local_optimizer(None)
    def local_gpu_togpu(node):
        if node.op == gpu_from_host:
            host_input = node.inputs[0]
            if host_input.owner and \
                    hasattr(host_input.owner.op, 'make_gpu_node'):
                try:
                    gpu_inputs = map(gpu_from_host, host_input.owner.inputs)
                except TypeError:
                    return False
                return [host_input.owner.op.make_gpu_node(*gpu_inputs)]
        elif hasattr(node.op, 'make_gpu_node') and \
                all([x.owner and x.owner.op == host_from_gpu
                     for x in node.inputs]):
            gpu_inputs = [x.owner.inputs[0] for x in node.inputs]
            return [host_from_gpu(node.op.make_gpu_node(*gpu_inputs))]
        return False

    @theano.sandbox.cuda.opt.register_opt()
    @theano.gof.local_optimizer([Breakpoint])
    def local_gpu_togpu_breakpoint(node):
        if isinstance(node.op, Breakpoint):
            result_input = node.inputs[0]
            if result_input.owner and result_input.owner.op == host_from_gpu:
                gpu_inputs = [x.owner.inputs[0]
                                if x.owner and x.owner.op == host_from_gpu
                                else x
                              for x in node.inputs]
                return [host_from_gpu(node.op.make_gpu_node(*gpu_inputs))]
        return False

