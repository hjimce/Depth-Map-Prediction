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
import numpy as np
import theano
import theano.tensor as T
from theano import Op, Apply
from theano.gradient import DisconnectedType

import thutil
from thutil import test_value, Eval

if thutil.use_gpu:
    import theano.sandbox.cuda
    from theano.sandbox.cuda import GpuOp, gpu_from_host, host_from_gpu, \
                                    CudaNdarrayType, CudaNdarray


def subsample2d(input, stride=(2,2), output_shape=None, transpose='n'):
    (bsize, ic, ix, iy) = input.shape
    (dx, dy) = stride
    if transpose.lower() == 't':
        if output_shape is None:
            output_shape = (ix * dx, iy * dy)
        out = T.zeros((bsize, ic,) + output_shape, dtype=input.dtype)
        out = T.set_subtensor(out[:, :, ::dx, ::dy], input)
    else:
        out = input[:, :, ::dx, ::dy]
    return out

def maxpool2d(input, winsize, stride=None, input_shape=None):
    if input_shape is None:
        input_shape = test_value(input).shape[-2:]
    inds = maxinds_2d(input, winsize, stride, input_shape)
    vals = index_pool_2d(input, inds, winsize, stride, input_shape)
    return (vals, inds)

def maxinds_2d(input, winsize, stride=None, input_shape=None):
    if input_shape is None:
        input_shape = test_value(input).shape[-2:]
    return MaxInds2D(input_shape, winsize, stride)(input)

def index_pool_2d(input, inds, winsize, stride=None,
                  input_shape=None):
    if input_shape is None:
        input_shape = test_value(input).shape[-2:]
    return IndexPool2D(input_shape, winsize, stride)(input, inds)

def index_unpool_2d(input, inds, winsize, stride=None,
                    input_shape=None, output_shape=None):
    if input_shape is None:
        input_shape = test_value(input).shape[-2:]
    return IndexUnpool2D(input_shape, winsize, stride,
                         output_shape=output_shape)(input, inds)

def sumpool2d(input, winsize, stride=None, input_shape=None, average=False):
    if input_shape is None:
        input_shape = test_value(input).shape[-2:]
    return SumPool2D(input_shape, winsize, stride, average=average)(input)

def sum_unpool_2d(input, winsize, stride=None,
                  input_shape=None, output_shape=None, average=False):
    if input_shape is None:
        input_shape = test_value(input).shape[-2:]
    return SumUnpool2D(input_shape, winsize, stride, output_shape,
                       average=average)(input)

def maxpool_features(input, winsize):
    if winsize == 1:
        return (input, T.zeros_like(input))
    (bsize, nc, ni, nj) = input.shape
    inp = input.transpose((0,2,3,1)).reshape((bsize, ni*nj, nc, 1))
    (vals, inds) = maxpool2d(inp, (winsize, 1))
    sz = vals.size / (bsize*ni*nj)
    vals = vals.reshape((bsize, ni, nj, sz)).transpose((0,3,1,2))
    return (vals, inds)

def index_unpool_features(input, inds, winsize):
    if winsize == 1:
        return input
    (bsize, nc, ni, nj) = input.shape
    inp = input.transpose((0,2,3,1)).reshape((bsize, ni*nj, nc, 1))
    vals = index_unpool_2d(inp, inds, (winsize,1))
    sz = vals.size / (bsize*ni*nj)
    vals = vals.reshape((bsize, ni, nj, sz)).transpose((0,3,1,2))
    return vals

def cmrnorm(x, winsize=5, scale=0.0001, pow=0.75, input_shape=None):
    if input_shape is None:
        input_shape = test_value(x.shape)[1:]
    return CMRNorm(input_shape, winsize, scale, pow, x.dtype)(x)

class PoolOp(Op):
    (is_pooling, is_unpooling) = (True, False)

    def __init__(self, input_shape, winsize, stride,
                       output_shape=None, dtype=None):
        if stride is None:
            stride = winsize
        self.input_shape = input_shape
        self.winsize = winsize
        self.stride = stride
        if output_shape is None:
            output_shape = self._infer_shape(input_shape)
        self.output_shape = output_shape
        self.output_dtype = dtype
        self._hash_key = (type(self), self.input_shape,
                          self.winsize, self.stride,
                          self.output_shape, self.output_dtype)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.input_shape == other.input_shape and
                self.winsize == other.winsize and
                self.stride == other.stride and
                self.output_shape == other.output_shape and
                self.output_dtype == other.output_dtype)

    def __hash__(self):
        return hash(self._hash_key)

    def _infer_shape(self, input_shape):
        (x, y) = input_shape
        (wx, wy) = self.winsize
        (sx, sy) = self.stride
        if self.is_pooling:
            return ((x-wx)//sx + 1, (y-wy)//sy + 1)
        else: # unpooling
            return ((x-1)*sx + wx, (y-1)*sy + wy)

    def infer_shape(self, node, input_shapes):
        s = input_shapes[0][:-2] + self.output_shape
        return (s,)

    def make_node(self, *inputs):
        inputs = tuple(map(T.as_tensor_variable, inputs))
        output = T.tensor4(dtype=(self.output_dtype or inputs[0].dtype))
        return Apply(self, inputs, (output,))

    def perform(self, node, inputs, (output,)):
        (bsize, nchan, unpooli, unpoolj) = inputs[0].shape
        (pooli, poolj) = self.output_shape
        (wi, wj) = self.winsize
        (si, sj) = self.stride
        output_dtype = self.output_dtype or node.inputs[0].dtype
        out = output[0]
        if out is None or out.dtype != output_dtype or \
           out.shape != (bsize, nchan, pooli, poolj):
            out = output[0] = np.empty((bsize, nchan, pooli, poolj),
                                       dtype=output_dtype)
        inp = inputs[0]
        poolfunc = self.perform_pool
        for b in xrange(bsize):
            for c in xrange(nchan):
                x = out[b, c].flat
                x[:] = [poolfunc(inp[b, c, i*si:i*si+wi, j*sj:j*sj+wj],
                                 inputs, b, c, i, j)
                        for i in xrange(pooli)
                        for j in xrange(poolj)]

    def _pool_c_code(self, node, name, input, output, body, sub,
                     output_type=None):
        (unpooli, unpoolj) = self.input_shape
        (pooli, poolj) = self.output_shape
        (wi, wj) = self.winsize
        (si, sj) = self.stride
        fail = sub['fail']
        if output_type is None:
            output_type = 'PyArray_ObjectType((PyObject*) %s, 0)' % input

        code = '''
        #define MIN(a,b) ((a) < (b) ? (a) : (b))
        int istart, jstart, iend, jend;
        int ind;
        int bsize = PyArray_DIMS(%(input)s)[0];
        int nchan = PyArray_DIMS(%(input)s)[1];
        npy_intp dims[4] = {0, 0, %(pooli)d, %(poolj)d};
        dims[0] = bsize;
        dims[1] = nchan;

        if (PyArray_NDIM(%(input)s) != 4) {
            PyErr_SetString(PyExc_ValueError, "input must be a 4d ndarray");
            %(fail)s;
        } 
        Py_XDECREF(%(output)s);
        %(output)s = (PyArrayObject*) PyArray_ZEROS(
                            4, dims, %(output_type)s, 0);

        for (int b = 0; b < bsize; ++b) {
            for (int c = 0; c < nchan; ++c) {
                for (int i = 0; i < %(pooli)d; ++i) {
                    istart = i * %(si)d;
                    iend = MIN(istart + %(wi)d, %(unpooli)d);
                    for (int j = 0; j < %(poolj)d; ++j) {
                        jstart = j * %(sj)d;
                        jend = MIN(jstart + %(wj)d, %(unpoolj)d);

                        %(body)s
                    }
                }
            }
        }
        ''' % locals()
        return code

class UnpoolOp(PoolOp):
    (is_pooling, is_unpooling) = (False, True)

    def perform(self, node, inputs, (output,)):
        vals = inputs[0]
        (bsize, nchan, pooli, poolj) = vals.shape
        (wi, wj) = self.winsize
        (si, sj) = self.stride
        (unpooli, unpoolj) = self.output_shape
        out = output[0] = np.zeros((bsize, nchan, unpooli, unpoolj),
                                   dtype=vals.dtype)
        for b in xrange(bsize):
            for c in xrange(nchan):
                for i in xrange(pooli):
                    for j in xrange(poolj):
                        x = out[b, c, i*si:i*si+wi, j*sj:j*sj+wj]
                        self.perform_unpool(x, vals[b,c,i,j],
                                            inputs, b, c, i, j)


class MaxInds2D(PoolOp):
    def perform_pool(self, pool_vals, inputs, b, c, i, j):
        return np.argmax(pool_vals)

    def make_gpu_node(self, input):
        return MaxInds2D_GPU(self.input_shape, self.winsize, self.stride) \
                            (input)

    def c_support_code(self):
        code = '''
        template<class T>
        inline int _argmax(PyArrayObject *x, int b, int c,
                           int istart, int iend, int jstart, int jend)
        {
            int k = 0, kmax = 0;
            T v, vmax;
            vmax = *(T*) PyArray_GETPTR4(x, b, c, istart, jstart);
            for (int i = istart; i < iend; ++i) {
                for (int j = jstart; j < jend; ++j, ++k) {
                    v = *(T*) PyArray_GETPTR4(x, b, c, i, j);
                    if (v > vmax) {
                        vmax = v;
                        kmax = k;
                    }
                }
            }
            return kmax;
        }
        '''
        return code

    def c_code(self, node, name, (input,), (output,), sub):
        output_type = {'int32': 'NPY_INT',
                       'float32': 'NPY_FLOAT32',
                       'float64': 'NPY_FLOAT64',
                       }[self.output_dtype or node.inputs[0].dtype]
        body = '''
        int v = _argmax<dtype_%(input)s>(
                      %(input)s, b, c, istart, iend, jstart, jend);
        *(dtype_%(output)s*) PyArray_GETPTR4(%(output)s, b, c, i, j)
                = (dtype_%(output)s) v;
        ''' % locals()
        return self._pool_c_code(node, name, input, output, body, sub)
    

class IndexPool2D(PoolOp):
    def perform_pool(self, pool_vals, inputs, b, c, i, j):
        return pool_vals.flat[int(inputs[1][b,c,i,j])]

    def grad(self, (vals, inds), (dvals,)):
        return (IndexUnpool2D(self.output_shape, self.winsize, self.stride,
                              output_shape=self.input_shape)(dvals, inds),
                DisconnectedType()(),)

    def make_gpu_node(self, input, inds):
        return IndexPool2D_GPU(self.input_shape, self.winsize, self.stride) \
                              (input, inds)

    def c_support_code(self):
        code = '''
        template<class T>
        inline T _select_ind(PyArrayObject *x, int b, int c,
                             int istart, int iend, int jstart, int jend,
                             int ind)
        {
            int jlen = jend - jstart;
            int i = istart + ind / jlen;
            int j = jstart + ind % jlen;
            return *(T*) PyArray_GETPTR4(x, b, c, i, j);
        }
        '''
        return code

    def c_code(self, node, name, (input, inds), (output,), sub):
        body = '''
        int ind = (int) *(dtype_%(inds)s*) 
                     PyArray_GETPTR4(%(inds)s, b, c, i, j);
        dtype_%(input)s v = _select_ind<dtype_%(input)s>(
                                %(input)s, b, c,
                                istart, iend, jstart, jend,
                                ind);
        *(dtype_%(output)s*)
            PyArray_GETPTR4(%(output)s, b, c, i, j) = v;
        ''' % locals()
        return self._pool_c_code(node, name, input, output, body, sub)


class IndexUnpool2D(UnpoolOp):
    def perform_unpool(self, unpool_vals, pool_val, inputs, b, c, i, j):
        unpool_vals.flat[int(inputs[1][b,c,i,j])] += pool_val

    def make_gpu_node(self, input, inds):
        return IndexUnpool2D_GPU(self.input_shape, self.winsize, self.stride,
                                 self.output_shape)(input, inds)

    def grad(self, (vals, inds), (doutput,)):
        return (IndexPool2D(self.output_shape, self.winsize, self.stride)
                           (doutput, inds),
                DisconnectedType()(),)

    def c_support_code(self):
        code = '''
        #define MIN(a,b) ((a) < (b) ? (a) : (b))
        template<class T>
        inline void _add_ind(
                           PyArrayObject *x, int b, int c,
                           int istart, int iend, int jstart, int jend,
                           int ind, T val)
        {
            int jlen = jend - jstart;
            int i = istart + ind / jlen;
            int j = jstart + ind % jlen;
            *(T*) PyArray_GETPTR4(x, b, c, i, j) += val;
        }
        '''
        return code

    def c_code(self, node, name, (input, inds), (output,), sub):
        (unpooli, unpoolj) = self.output_shape
        (pooli, poolj) = self.input_shape
        (wi, wj) = self.winsize
        (si, sj) = self.stride
        fail = sub['fail']

        code = '''
        int istart, jstart;
        int ind;
        int bsize = PyArray_DIMS(%(input)s)[0];
        int nchan = PyArray_DIMS(%(input)s)[1];
        npy_intp dims[4] = {0, 0, %(unpooli)d, %(unpoolj)d};
        dims[0] = bsize;
        dims[1] = nchan;
        dtype_%(output)s v;

        if (PyArray_NDIM(%(input)s) != 4) {
            PyErr_SetString(PyExc_ValueError, "input must be a 4d ndarray");
            %(fail)s;
        } 
        Py_XDECREF(%(output)s);
        %(output)s = (PyArrayObject*) PyArray_ZEROS(
                            4, dims,
                            PyArray_ObjectType((PyObject*) %(input)s, 0),
                            0);

        for (int b = 0; b < bsize; ++b) {
            for (int c = 0; c < nchan; ++c) {
                for (int i = 0; i < %(pooli)d; ++i) {
                    istart = i * %(si)d;
                    for (int j = 0; j < %(poolj)d; ++j) {
                        jstart = j * %(sj)d;

                        ind = (int) *(dtype_%(inds)s*) 
                                PyArray_GETPTR4(%(inds)s, b, c, i, j);
                        v = *(dtype_%(input)s*)
                                PyArray_GETPTR4(%(input)s, b, c, i, j);
                        _add_ind<dtype_%(output)s>(
                                  %(output)s, b, c,
                                  istart, MIN(istart + %(wi)d, %(unpooli)d),
                                  jstart, MIN(jstart + %(wj)d, %(unpoolj)d),
                                  ind, v);
                    }
                }
            }
        }
        ''' % locals()
        return code


class SumPool2D(PoolOp):
    def __init__(self, *args, **kwargs):
        self.average = kwargs.pop('average', False)
        PoolOp.__init__(self, *args, **kwargs)
        self._hash_key = self._hash_key + (self.average,)

    def __eq__(self, other):
        return PoolOp.__eq__(self, other) and self.average == other.average

    def perform_pool(self, pool_vals, inputs, b, c, i, j):
        if self.average:
            return np.mean(pool_vals)
        return np.sum(pool_vals)

    def make_gpu_node(self, input):
        return SumPool2D_GPU(self.input_shape, self.winsize, self.stride,
                             average=self.average)(input)

    def grad(self, (vals,), (dvals,)):
        return (SumUnpool2D(self.output_shape, self.winsize, self.stride,
                            output_shape=self.input_shape,
                            average=self.average)
                       (dvals),
                )

    def c_support_code(self):
        code = '''
        template<class T, bool average>
        inline T _sum_window(PyArrayObject *x, int b, int c,
                             int istart, int iend, int jstart, int jend)
        {
            T vsum = 0;
            for (int i = istart; i < iend; ++i) {
                for (int j = jstart; j < jend; ++j) {
                    vsum += *(T*) PyArray_GETPTR4(x, b, c, i, j);
                }
            }
            if (average)
                vsum /= (iend - istart) * (jend - jstart);
            return vsum;
        }
        '''
        return code

    def c_code(self, node, name, (input,), (output,), sub):
        output_type = {'int32': 'NPY_INT',
                       'float32': 'NPY_FLOAT32',
                       'float64': 'NPY_FLOAT64',
                       }[self.output_dtype or node.inputs[0].dtype]
        average = int(self.average)
        body = '''
        dtype_%(input)s v = _sum_window<dtype_%(input)s, %(average)d>(
                                %(input)s, b, c, istart, iend, jstart, jend);
        *(dtype_%(output)s*) PyArray_GETPTR4(%(output)s, b, c, i, j)
                = (dtype_%(output)s) v;
        ''' % locals()
        return self._pool_c_code(node, name, input, output, body, sub)


class SumUnpool2D(UnpoolOp):
    def __init__(self, *args, **kwargs):
        self.average = kwargs.pop('average', False)
        UnpoolOp.__init__(self, *args, **kwargs)
        self._hash_key = self._hash_key + (self.average,)

    def __eq__(self, other):
        return UnpoolOp.__eq__(self, other) and self.average == other.average

    def perform_unpool(self, unpool_vals, pool_val, inputs, b, c, i, j):
        if self.average:
            unpool_vals += pool_val / float(unpool_vals.size)
        else:
            unpool_vals += pool_val

    def make_gpu_node(self, input):
        return SumUnpool2D_GPU(self.input_shape, self.winsize, self.stride,
                               self.output_shape,
                               average=self.average)(input)

    def grad(self, (vals,), (dvals,)):
        return (SumPool2D(self.output_shape, self.winsize, self.stride,
                          average=self.average)
                       (dvals),
                )

    def c_support_code(self):
        code = '''
        #define MIN(a,b) ((a) < (b) ? (a) : (b))
        template<class T, bool average>
        inline void _add_val_to_window(
                           PyArrayObject *x, int b, int c,
                           int istart, int iend, int jstart, int jend,
                           T val)
        {
            if (average) {
                val /= (T) ((iend - istart) * (jend - jstart));
            }
            for (int i = istart; i < iend; ++i) {
                for (int j = jstart; j < jend; ++j) {
                    *(T*) PyArray_GETPTR4(x, b, c, i, j) += val;
                }
            }
        }
        '''
        return code

    def c_code(self, node, name, (input,), (output,), sub):
        (unpooli, unpoolj) = self.output_shape
        (pooli, poolj) = self.input_shape
        (wi, wj) = self.winsize
        (si, sj) = self.stride
        average = self.average
        fail = sub['fail']

        code = '''
        int istart, jstart;
        int bsize = PyArray_DIMS(%(input)s)[0];
        int nchan = PyArray_DIMS(%(input)s)[1];
        npy_intp dims[4] = {0, 0, %(unpooli)d, %(unpoolj)d};
        dims[0] = bsize;
        dims[1] = nchan;
        dtype_%(output)s v;

        if (PyArray_NDIM(%(input)s) != 4) {
            PyErr_SetString(PyExc_ValueError, "input must be a 4d ndarray");
            %(fail)s;
        } 
        Py_XDECREF(%(output)s);
        %(output)s = (PyArrayObject*) PyArray_ZEROS(
                            4, dims,
                            PyArray_ObjectType((PyObject*) %(input)s, 0),
                            0);

        for (int b = 0; b < bsize; ++b) {
            for (int c = 0; c < nchan; ++c) {
                for (int i = 0; i < %(pooli)d; ++i) {
                    istart = i * %(si)d;
                    for (int j = 0; j < %(poolj)d; ++j) {
                        jstart = j * %(sj)d;

                        v = *(dtype_%(input)s*)
                                PyArray_GETPTR4(%(input)s, b, c, i, j);
                        /* add val to all elements in this pooling window */
                        _add_val_to_window<dtype_%(output)s, %(average)d>(
                                  %(output)s, b, c,
                                  istart, MIN(istart + %(wi)d, %(unpooli)d),
                                  jstart, MIN(jstart + %(wj)d, %(unpoolj)d),
                                  v);
                    }
                }
            }
        }
        ''' % locals()
        return code


class CMRNorm(Op):
    def __init__(self, input_shape, winsize, scale, pow, dtype=None):
        self.input_shape = tuple(input_shape)
        self.winsize = winsize
        self.scale = scale
        self.pow = pow
        self.output_dtype = dtype
        self.enable_grad = True
        self._hash_key = (type(self), self.__class__, self.input_shape,
                          self.winsize, self.scale, self.pow,
                          self.output_dtype)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.__class__ == other.__class__ and
                self.input_shape == other.input_shape and
                self.winsize == other.winsize and
                self.scale == other.scale and
                self.pow == other.pow and
                self.output_dtype == other.output_dtype)

    def __hash__(self):
        return hash(self._hash_key)

    def make_gpu_node(self, input):
        return CMRNorm_GPU(self.input_shape,
                            self.winsize, self.scale, self.pow,
                            dtype=self.output_dtype)(input)

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def make_node(self, *inputs):
        inputs = tuple(map(T.as_tensor_variable, inputs))
        output = T.tensor4(dtype=(self.output_dtype or inputs[0].dtype))
        return Apply(self, inputs, (output,))

    def perform(self, node, (input,), (output,)):
        (bsize, nchan, ni, nj) = input.shape
        output_dtype = self.output_dtype or node.inputs[0].dtype
        out = output[0]
        if out is None or out.dtype != output_dtype or \
           out.shape != (bsize, nchan, ni, ni):
            out = output[0] = np.empty((bsize, nchan, ni, nj),
                                       dtype=output_dtype)
        x = input
        x2 = x ** 2
        sums = np.zeros_like(x)
        for p in xrange(self.winsize):
            d = p - (self.winsize//2)
            sums[:,max(0,-d):min(nchan,nchan-d),:,:] += \
                    x2[:,max(0,d):min(nchan,nchan+d),:,:]
        out[:] = x * ((2 + self.scale * sums) ** (-self.pow))

    def grad(self, (x,), (dy,)):
        if not self.enable_grad:
            return [dy]
        return (CMRNormGrad(self.input_shape,
                            self.winsize, self.scale, self.pow,
                            dtype=self.output_dtype)
                       (x, self(x), dy),
                )

class CMRNormGrad(CMRNorm):
    def perform(self, node, (x, y, dy), (output,)):
        (bsize, nchan, ni, nj) = x.shape
        output_dtype = self.output_dtype or node.inputs[0].dtype
        dx = output[0]
        if dx is None or dx.dtype != output_dtype or \
           dx.shape != (bsize, nchan, ni, ni):
            dx = output[0] = np.empty((bsize, nchan, ni, nj),
                                      dtype=output_dtype)
        x2 = x ** 2
        sums = np.zeros_like(x)
        for p in xrange(self.winsize):
            d = p - (self.winsize//2)
            sums[:,max(0,-d):min(nchan,nchan-d),:,:] += \
                    x2[:,max(0,d):min(nchan,nchan+d),:,:]
        denom = (2 + self.scale * sums) ** (-self.pow)
        a = (-2 * self.scale * self.pow) * y * denom
        dx[:] = 0
        x_dy = x * dy
        for p in xrange(self.winsize):
            d = p - (self.winsize//2)
            # slices of "convolution" window sliding
            lhs = slice(max(0,-d), min(nchan,nchan-d))
            rhs = slice(max(0,d), min(nchan,nchan+d))
            dx[:,lhs,:,:] += x_dy[:,rhs,:,:]
        dx *= a
        dx += dy * denom

    def make_gpu_node(self, *inputs):
        return CMRNormGrad_GPU(self.input_shape,
                               self.winsize, self.scale, self.pow,
                               dtype=self.output_dtype)(*inputs)

    def infer_shape(self, node, input_shapes):
        return (input_shapes[0],)

    def grad(self, inputs, doutputs):
        raise NotImplementedError


if thutil.use_gpu:

    source_support_defs = '''
        #define MOD %
        #define MIN(a,b) ((a) < (b) ? (a) : (b))
        #define IDX4(n1, n2, n3, n4, i1, i2, i3, i4) \\
            ((i1)*(n2)*(n3)*(n4) + (i2)*(n3)*(n4) + (i3)*(n4) + (i4))
        #define IDX3(n1, n2, n3, i1, i2, i3) \\
            ((i1)*(n2)*(n3) + (i2)*(n3) + (i3))
        #define UNRAVEL_IDX4(ind, n1, n2, n3, n4, i1, i2, i3, i4) \\
            { \\
                i1 = (ind) / ((n2)*(n3)*(n4)); \\
                i2 = ((ind) MOD ((n2)*(n3)*(n4))) / ((n3)*(n4)); \\
                i3 = ((ind) MOD ((n3)*(n4))) / (n4); \\
                i4 = ((ind) MOD (n4)); \\
            }
        #define SETIF(var, val, cond) \
            (var = (val)*!!(cond) + (var)*!(cond))
        #define DIVUP(x,y) (1 + (((x) - 1) / (y)))

        static void launch_sizes(int nthreads, dim3 &grid_size, dim3 &block_size)
        {
            static const int min_threads = 16;
            static const int max_threads = 256;
            static const int max_blocks = 65535;

            int ngroups = (nthreads + min_threads - 1) / min_threads;

            if (ngroups == 1) {
                grid_size = dim3(1);
                block_size = dim3(min_threads);
            } else if (nthreads < max_blocks * min_threads) {
                grid_size = dim3(ngroups);
                block_size = dim3(min_threads);
            } else if (nthreads < max_blocks * max_threads) {
                grid_size = dim3(max_blocks);
                block_size = dim3((ngroups + max_blocks - 1)
                                  / max_blocks * min_threads);
            } else {
                grid_size = dim3(max_blocks);
                block_size = dim3(max_threads);
            }
        }
        '''

    class PoolGpuOp(GpuOp):
        def make_node(self, *inputs):
            output = CudaNdarrayType((False,) * 4)()
            return Apply(self, inputs, (output,))

        def c_support_code(self):
            if self.is_pooling:
                unpooled_shape = self.input_shape
                pooled_shape = self.output_shape
                pooled_stride_i = 1
                pooled_stride_j = 1
            else:
                unpooled_shape = self.output_shape
                pooled_shape = self.input_shape
                pooled_stride_i = 'DIVUP(wsize_i, stride_i)'
                pooled_stride_j = 'DIVUP(wsize_j, stride_j)'
            assert unpooled_shape[0] >= (pooled_shape[0] - 1) * self.stride[0] + self.winsize[0]
            assert unpooled_shape[1] >= (pooled_shape[1] - 1) * self.stride[1] + self.winsize[1]

            source = source_support_defs + '''

            #define unpooled_i %(unpooled_shape[0])d
            #define unpooled_j %(unpooled_shape[1])d
            #define pooled_i %(pooled_shape[0])d
            #define pooled_j %(pooled_shape[1])d
            #define wsize_i %(self.winsize[0])d
            #define wsize_j %(self.winsize[1])d
            #define stride_i %(self.stride[0])d
            #define stride_j %(self.stride[1])d
            #define pooled_stride_i %(pooled_stride_i)s
            #define pooled_stride_j %(pooled_stride_j)s
            #define pooled_si DIVUP(pooled_i, pooled_stride_i)
            #define pooled_sj DIVUP(pooled_j, pooled_stride_j)

            #define ntiles_per_call (pooled_si * pooled_sj)
            static __global__ void %(self.ker_name)s ( %(self.ker_args)s,
                                                       float *out,
                                                       int nimgs,
                                                       int pooled_start_i,
                                                       int pooled_start_j,
                                                       uint32_t randseed)
            {
                unsigned total_threads = gridDim.x * blockDim.x;
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int t, tnum, img, tile;
                int pi, pj, ui0, uj0;

                %(self.ker_defs)s

                for (t = tid; t < nimgs * ntiles_per_call; t += total_threads) {
                    tnum = t; /* tile number */
                    img = tnum / ntiles_per_call; /* image the tile is in */
                    tile = tnum MOD ntiles_per_call; /* tile in image */
                    pi = (tile / pooled_sj); /* pooled pixel indices */
                    pj = (tile MOD pooled_sj);
                    pi = pi * pooled_stride_i + pooled_start_i;
                    pj = pj * pooled_stride_j + pooled_start_j;
                    if (pi >= pooled_i || pj >= pooled_j)
                        continue;
                    ui0 = pi * stride_i; /* unpooled window top-left pixel */
                    uj0 = pj * stride_j;

                    %(self.ker_loop_body)s
                }
            }

            ''' % Eval()
            return source

        def c_code(self, node, nodename, inputs, outputs, sub):
            (output,) = outputs

            source = '''
            const int *input_dims = CudaNdarray_HOST_DIMS(%(inputs[0])s);
            const int *output_dims = %(output)s ?
                                        CudaNdarray_HOST_DIMS(%(output)s) :
                                        NULL;
            const int dims[] = { input_dims[0],
                                 input_dims[1],
                                 %(self.output_shape[0])s,
                                 %(self.output_shape[1])s };
            const int nimgs = dims[0] * dims[1]; /* imgs * channels */

            int ntiles = nimgs * ntiles_per_call; /* one thread per tile */
            dim3 grid_size, block_size;
            launch_sizes(ntiles, grid_size, block_size);

            CudaNdarray %(', '.join('*%s_contig' % inp for inp in inputs))s ;
            cudaError_t err;
            '''

            source += '''
            if (%(output)s == NULL
                    || !CudaNdarray_is_c_contiguous(%(output)s)
                    || %(output)s->nd != 4
                    || dims[0] != output_dims[0]
                    || dims[1] != output_dims[1]
                    || dims[2] != output_dims[2]
                    || dims[3] != output_dims[3]) {

                Py_XDECREF(%(output)s);
                %(output)s = (CudaNdarray*)CudaNdarray_New();
                if (%(output)s == NULL
                        || CudaNdarray_alloc_contiguous(%(output)s, 4, dims)) {
                    Py_XDECREF(%(output)s);
                    %(output)s = NULL;
                    %(sub['fail'])s;
                }
            }
            
            if (%(self.zero_output)d) {
                if (cudaMemset(
                        CudaNdarray_DEV_DATA(%(output)s),
                        0, CudaNdarray_SIZE(%(output)s) * sizeof(float))
                     != cudaSuccess) {
                    PyErr_Format(PyExc_MemoryError,
                                 "%(self.ker_name)s: Error in memset");
                    Py_XDECREF(%(output)s);
                    %(output)s = NULL;
                    %(sub['fail'])s;
                }

            }
            '''

            for inp in inputs:
                source += '''
                %(inp)s_contig = %(inp)s;
                if (!CudaNdarray_is_c_contiguous(%(inp)s)) {
                    %(inp)s_contig = (CudaNdarray*) CudaNdarray_Copy(%(inp)s);
                    assert(CudaNdarray_is_c_contiguous(%(inp)s_contig));
                }
                ''' % Eval()

            source += '''
            /* call kernel once for each offset within the pooled stride */
            for (int j = 0; j < pooled_stride_j; ++j) {
                for (int i = 0; i < pooled_stride_i; ++i) {
                    %(self.ker_name)s <<<grid_size, block_size>>> (
                        %(', '.join('CudaNdarray_DEV_DATA(%s_contig)' % x
                                    for x in inputs))s,
                        CudaNdarray_DEV_DATA(%(output)s),
                        nimgs,
                        i, j,
                        rand()
                        );
                }
            }
            CNDA_THREAD_SYNC;
            '''

            for inp in inputs:
                source += '''
                if (%(inp)s_contig != %(inp)s) {
                    Py_DECREF(%(inp)s_contig);
                }
                ''' % Eval()

            source += '''
            err = cudaGetLastError();

            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "Cuda error: %%s: %%s",
                             "%(self.ker_name)s", cudaGetErrorString(err));
                %(sub['fail'])s
            }
            '''

            source = source % Eval()
            return source


    class MaxInds2D_GPU(PoolGpuOp, MaxInds2D):
        ker_name = 'pool_maxind'

        ker_args = 'float *X'

        ker_defs = '''
            int u, ui, uj, ismax, max_ind;
            float val, max_val;
        '''

        zero_output = False

        ker_loop_body = '''
            max_val = -1000000;
            for (u = 0; u < wsize_i * wsize_j; ++u) {
                ui = ui0 + u / wsize_j;
                uj = uj0 + u MOD wsize_j;
                if ((ui < unpooled_i) && (uj < unpooled_j)) {
                    val = X[IDX3(nimgs, unpooled_i, unpooled_j,
                                 img, ui, uj)];
                    ismax = (val > max_val);
                    SETIF(max_val, val, ismax);
                    SETIF(max_ind, u, ismax);
                }
            }

            out[IDX3(nimgs, pooled_i, pooled_j,
                     img, pi, pj)]
                = max_ind;
        '''

        def perform(self, node, (input,), (output,)):
            output_host = [None]
            MaxInds2D.perform(self, node, (np.array(input),), (output_host,))
            output[0] = CudaNdarray(output_host[0].astype(np.float32))


    class IndexPool2D_GPU(PoolGpuOp, IndexPool2D):
        ker_name = 'pool_index'

        ker_args = 'float *input, float *inds'

        ker_defs = '''
            int ui, uj, ind;
            float val;
        '''

        zero_output = False

        ker_loop_body = '''
            ind = (int) inds[IDX3(nimgs, pooled_i, pooled_j,
                                  img, pi, pj)];

            ui = ui0 + ind / wsize_j;
            uj = uj0 + ind MOD wsize_j;
            val = input[IDX3(nimgs, unpooled_i, unpooled_j,
                             img, ui, uj)];

            out[IDX3(nimgs, pooled_i, pooled_j,
                     img, pi, pj)] = val;
        '''

        def perform(self, node, inputs, (output,)):
            output_host = [None]
            IndexPool2D.perform(self, node,
                                map(np.array, inputs), (output_host,))
            output[0] = CudaNdarray(output_host[0].astype(np.float32))


    class IndexUnpool2D_GPU(PoolGpuOp, IndexUnpool2D):
        ker_name = 'unpool_index'

        ker_args = 'float *input, float *inds'

        ker_defs = '''
            int ui, uj, ind;
            float val;
        '''

        zero_output = True

        ker_loop_body = '''
            ind = (int) inds[IDX3(nimgs, pooled_i, pooled_j,
                                  img, pi, pj)];
            val = input[IDX3(nimgs, pooled_i, pooled_j,
                             img, pi, pj)];

            ui = ui0 + ind / wsize_j;
            uj = uj0 + ind MOD wsize_j;
            out[IDX3(nimgs, unpooled_i, unpooled_j,
                     img, ui, uj)] += val;
        '''

        def perform(self, node, inputs, (output,)):
            output_host = [None]
            IndexUnpool2D.perform(self, node,
                                  map(np.array, inputs), (output_host,))
            output[0] = CudaNdarray(output_host[0].astype(np.float32))


    class SumPool2D_GPU(PoolGpuOp, SumPool2D):
        ker_name = 'pool_sum'

        ker_args = 'float *X'

        ker_defs = '''
            int u, ui, uj, usize_i, usize_j;
            float vsum;
        '''

        zero_output = False

        def __init__(self, *args, **kwargs):
            super(SumPool2D_GPU, self).__init__(*args, **kwargs)

            self.ker_loop_body = '''
                vsum = 0;
                for (u = 0; u < wsize_i * wsize_j; ++u) {
                    ui = ui0 + u / wsize_j;
                    uj = uj0 + u MOD wsize_j;
                    if (ui < unpooled_i && uj < unpooled_j)
                        vsum += X[IDX3(nimgs, unpooled_i, unpooled_j,
                                       img, ui, uj)];
                }

                if (%(average)s) {
                    usize_i = MIN(ui0 + wsize_i, unpooled_i) - ui0;
                    usize_j = MIN(uj0 + wsize_j, unpooled_j) - uj0;
                    vsum /= (usize_i * usize_j);
                }

                out[IDX3(nimgs, pooled_i, pooled_j,
                         img, pi, pj)]
                    = vsum;
            ''' % {'average': int(self.average)}

        def perform(self, node, (input,), (output,)):
            output_host = [None]
            SumPool2D.perform(self, node, (np.array(input),), (output_host,))
            output[0] = CudaNdarray(output_host[0].astype(np.float32))


    class SumUnpool2D_GPU(PoolGpuOp, SumUnpool2D):
        ker_name = 'unpool_sum'

        ker_args = 'float *input'

        ker_defs = '''
            int u, ui, uj, usize_i, usize_j;
            float val;
        '''

        zero_output = True

        def __init__(self, *args, **kwargs):
            super(SumUnpool2D_GPU, self).__init__(*args, **kwargs)

            self.ker_loop_body = '''
                val = input[IDX3(nimgs, pooled_i, pooled_j,
                                 img, pi, pj)];
                if (%(average)d) {  /* average? */
                    usize_i = MIN(ui0 + wsize_i, unpooled_i) - ui0;
                    usize_j = MIN(uj0 + wsize_j, unpooled_j) - uj0;
                    val /= (float) (usize_i * usize_j);
                }

                for (u = 0; u < wsize_i * wsize_j; ++u) {
                    ui = ui0 + u / wsize_j;
                    uj = uj0 + u MOD wsize_j;
                    if (ui < unpooled_i && uj < unpooled_j)
                        out[IDX3(nimgs, unpooled_i, unpooled_j,
                                 img, ui, uj)] += val;
                }
            ''' % {'average': int(self.average)}

        def perform(self, node, (input,), (output,)):
            output_host = [None]
            SumUnpool2D.perform(self, node, (np.array(input),), (output_host,))
            output[0] = CudaNdarray(output_host[0].astype(np.float32))


    class CMRNorm_GPU(GpuOp, CMRNorm):
        def __init__(self, *args, **kwargs):
            CMRNorm.__init__(self, *args, **kwargs)
            self._define_kernel_code()

        def make_node(self, *inputs):
            output = CudaNdarrayType((False,) * 4)()
            return Apply(self, inputs, (output,))

        def perform(self, node, inputs, (output,)):
            output_host = [None]
            CMRNorm.perform(self, node,
                            map(np.array, inputs), (output_host,))
            output[0] = CudaNdarray(output_host[0].astype(np.float32))

        def c_support_code(self):
            source = source_support_defs + '''

            static __global__ void %(self.ker_name)s ( %(self.ker_args)s,
                                                       float *output,
                                                       int nimgs,
                                                       int nchan,
                                                       int ni, int nj)
            {
                unsigned total_threads = gridDim.x * blockDim.x;
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int t, img, chan, i, j;

                for (t = tid; t < nchan * ni * nj * nimgs; t += total_threads) {
                    UNRAVEL_IDX4(t, nimgs, nchan, ni, nj,
                                      img,  chan,  i,  j);

                    %(self.ker_loop_body)s
                }
            }

            ''' % Eval()
            return source

        zero_output = False

        ker_name = 'cmrnorm'

        ker_args = 'float *input'

        def _define_kernel_code(self):
            self.ker_loop_body = '''

            int d;
            float sum = 0;
            float x;

            for (d = -%(self.winsize / 2)d; d <= %(self.winsize / 2)d; ++d) {
                if (chan + d >= 0 && chan + d < nchan) {
                    x = input[IDX4(nimgs, nchan, ni, nj,
                                     img, chan + d, i, j)];
                    sum += x * x;
                }
            }

            x = input[IDX4(nimgs, nchan, ni, nj,
                             img,  chan,  i,  j)];
            output[IDX4(nimgs, nchan, ni, nj,
                          img, chan,  i,  j)]
                = x * __powf(2 + %(self.scale)s * sum, -%(self.pow)s);

            ''' % Eval()

        def c_code_cache_version(self):
            return (1, hash(self))

        def c_code(self, node, nodename, inputs, outputs, sub):
            (output,) = outputs

            source = '''
            const int *dims = CudaNdarray_HOST_DIMS(%(inputs[0])s);
            const int *output_dims = %(output)s ?
                                        CudaNdarray_HOST_DIMS(%(output)s) :
                                        NULL;

            const int nchan = dims[0];
            const int ni    = dims[1];
            const int nj    = dims[2];
            const int nimgs = dims[3];

            int nelems = nimgs * nchan * ni * nj; /* one thread elem */
            dim3 grid_size, block_size;
            launch_sizes(nelems, grid_size, block_size);

            CudaNdarray %(', '.join('*%s_contig' % inp for inp in inputs))s ;
            cudaError_t err;
            '''

            source += '''
            if (%(output)s == NULL
                    || !CudaNdarray_is_c_contiguous(%(output)s)
                    || %(output)s->nd != 4
                    || dims[0] != output_dims[0]
                    || dims[1] != output_dims[1]
                    || dims[2] != output_dims[2]
                    || dims[3] != output_dims[3]) {

                Py_XDECREF(%(output)s);
                %(output)s = (CudaNdarray*)CudaNdarray_New();
                if (%(output)s == NULL
                        || CudaNdarray_alloc_contiguous(%(output)s, 4, dims)) {
                    Py_XDECREF(%(output)s);
                    %(output)s = NULL;
                    %(sub['fail'])s;
                }
            }
            
            if (%(self.zero_output)d) {
                if (cudaMemset(
                        CudaNdarray_DEV_DATA(%(output)s),
                        0, CudaNdarray_SIZE(%(output)s) * sizeof(float))
                     != cudaSuccess) {
                    PyErr_Format(PyExc_MemoryError,
                                 "%(self.ker_name)s: Error in memset");
                    Py_XDECREF(%(output)s);
                    %(output)s = NULL;
                    %(sub['fail'])s;
                }

            }
            '''

            for inp in inputs:
                source += '''
                %(inp)s_contig = %(inp)s;
                if (!CudaNdarray_is_c_contiguous(%(inp)s)) {
                    %(inp)s_contig = (CudaNdarray*) CudaNdarray_Copy(%(inp)s);
                    assert(CudaNdarray_is_c_contiguous(%(inp)s_contig));
                }
                ''' % Eval()

            source += '''
            %(self.ker_name)s <<<grid_size, block_size>>> (
                %(', '.join('CudaNdarray_DEV_DATA(%s_contig)' % x
                            for x in inputs))s,
                CudaNdarray_DEV_DATA(%(output)s),
                nimgs, nchan, ni, nj
                );

            CNDA_THREAD_SYNC;
            '''

            for inp in inputs:
                source += '''
                if (%(inp)s_contig != %(inp)s) {
                    Py_DECREF(%(inp)s_contig);
                }
                ''' % Eval()

            source += '''
            err = cudaGetLastError();

            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "Cuda error: %%s: %%s",
                             "%(self.ker_name)s", cudaGetErrorString(err));
                %(sub['fail'])s
            }
            '''

            source = source % Eval()
            return source

    class CMRNormGrad_GPU(CMRNorm_GPU, CMRNormGrad):
        def __init__(self, *args, **kwargs):
            CMRNormGrad.__init__(self, *args, **kwargs)
            self._define_kernel_code()

        def perform(self, node, inputs, (output,)):
            output_host = [None]
            CMRNormGrad.perform(self, node,
                                map(np.array, inputs), (output_host,))
            output[0] = CudaNdarray(output_host[0].astype(np.float32))

        zero_output = False

        ker_name = 'cmrnormgrad'

        ker_args = 'float *input, float *ys, float *dys'

        def _define_kernel_code(self):
            self.ker_loop_body = '''

            int d;
            float sum = 0;
            float x, denom, a, y, dx, x_d, dy_d;

            for (d = -%(self.winsize / 2)d; d <= %(self.winsize / 2)d; ++d) {
                if (chan + d >= 0 && chan + d < nchan) {
                    x_d = input[IDX4(nimgs, nchan, ni, nj,
                                       img, chan + d, i, j)];
                    sum += x_d * x_d;
                }
            }

            x = input[IDX4(nimgs, nchan, ni, nj,
                             img,  chan,  i,  j)];
            y = ys[IDX4(nimgs, nchan, ni, nj,
                          img,  chan,  i,  j)];

            denom = __powf(2 + %(self.scale)s * sum, -%(self.pow)s);
            a = (-2 * %(self.scale)s * %(self.pow)s) * y * denom;

            dx = 0;
            for (d = -%(self.winsize / 2)d; d <= %(self.winsize / 2)d; ++d) {
                if (chan + d >= 0 && chan + d < nchan) {
                    x_d = input[IDX4(nimgs, nchan, ni, nj,
                                       img, chan + d,  i,  j)];
                    dy_d = dys[IDX4(nimgs, nchan, ni, nj,
                                      img, chan + d,  i,  j)];
                    dx += x_d * dy_d;
                }
            }

            dx *= a;
            dx += denom * dys[IDX4(nimgs, nchan, ni, nj,
                                     img,  chan,  i,  j)];

            output[IDX4(nimgs, nchan, ni, nj,
                          img,  chan,  i,  j)] = dx;

            ''' % Eval()


def test_pooling():
    from theano.tests.unittest_tools import verify_grad

    winsize = (5,5)
    stride = (3,3)

    xtest = np.random.rand(3,2,16,30)
    xtest = xtest.astype(theano.config.floatX)

    x = T.tensor4('x', dtype=theano.config.floatX)
    x.tag.test_value = xtest

    # max pool/unpool

    xinds = maxinds_2d(x, winsize, stride=stride)
    indf = theano.function([x], xinds, mode='DEBUG_MODE')
    theano.printing.debugprint(indf)
    xinds_val = indf(xtest)

    xshape = xtest.shape[-2:]

    xmax = index_pool_2d(x, xinds, winsize, stride=stride)
    poolf = theano.function([x], xmax, mode='DEBUG_MODE')
    theano.printing.debugprint(poolf)
    xmax_val = poolf(xtest)

    unpoolf = theano.function([x], index_unpool_2d(xmax, xinds, winsize,
                                                   stride=stride,
                                                   input_shape=xmax_val.shape[-2:],
                                                   output_shape=xshape),
                            mode='DEBUG_MODE')
    theano.printing.debugprint(unpoolf)
    ux_val = unpoolf(xtest)
    if stride == winsize:
        assert np.sum(xtest == ux_val) == np.prod(xmax_val.shape)

    # sum pool/unpool

    xsum = sumpool2d(x, winsize, stride)
    poolf = theano.function([x], xsum, mode='DEBUG_MODE')
    theano.printing.debugprint(poolf)
    xsum_val = poolf(xtest)
    assert xsum_val.shape == xmax_val.shape

    xavg = sumpool2d(x, winsize, stride, average=True)
    poolf = theano.function([x], xavg, mode='DEBUG_MODE')
    theano.printing.debugprint(poolf)
    xavg_val = poolf(xtest)
    assert xavg_val.shape == xsum_val.shape

    unpoolf = theano.function([x], sum_unpool_2d(xsum, winsize, stride,
                                                 input_shape=xsum_val.shape[-2:],
                                                 output_shape=xshape),
                              mode='DEBUG_MODE')
    theano.printing.debugprint(unpoolf)
    ux_val = unpoolf(xtest)


    T.verify_grad(lambda x: sumpool2d(x, winsize=winsize, stride=stride,
                                      input_shape=(16,30)),
                  (xtest,),
                  rng=np.random.RandomState(0))

    T.verify_grad(lambda xsum: sum_unpool_2d(xsum,
                                             winsize=winsize, stride=stride,
                                             input_shape=xsum_val.shape[-2:],
                                             output_shape=xshape),
                  (xsum_val,),
                  rng=np.random.RandomState(0))

    T.verify_grad(lambda x: sumpool2d(x, winsize=winsize, stride=stride,
                                      average=True,
                                      input_shape=(16,30)),
                  (xtest,),
                  rng=np.random.RandomState(0))

    T.verify_grad(lambda x: index_pool_2d(x, xinds_val,
                                          winsize=winsize, stride=stride,
                                          input_shape=(16,30))[0],
                  (xtest,),
                  rng=np.random.RandomState(0))

    T.verify_grad(lambda xmax: index_unpool_2d(xmax, xinds_val,
                                               winsize=winsize,
                                               stride=stride,
                                               input_shape=xmax_val.shape[-2:],
                                               output_shape=(16,30)),
                  (xmax_val,),
                  rng=np.random.RandomState(0))

def test_cmrnorm():
    from theano.tests.unittest_tools import verify_grad

    xtest = np.random.rand(2,8,3,4)
    xtest = xtest.astype(theano.config.floatX)

    x = T.tensor4('x', dtype=theano.config.floatX)
    x.tag.test_value = xtest

    y = cmrnorm(x, input_shape=xtest.shape[1:])
    f = theano.function([x], y, mode='DEBUG_MODE')
    f(xtest)

    f = theano.function([x], gpu_from_host(T.grad(T.sum(y), wrt=x)),
                        mode='DEBUG_MODE')
    f(xtest)
    theano.printing.debugprint(f)

    T.verify_grad(lambda x: cmrnorm(x, input_shape=xtest.shape[1:]),
                  (xtest,),
                  rng=np.random.RandomState(0))

    print 'cmrnorm passed'

if __name__ == '__main__':
    test_pooling()
    test_cmrnorm()
