'''
Misc image and filter manipulation utilities.

Author: deigen

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

import numpy as np

def rot180(x):
    '''180 degree matrix rotation for a 2D matrix'''
    return x[::-1, ::-1]

def scale_values(x, min=None, max=None, center=None):
    '''Scales values of x so min->0 and max->1.
       By default uses min(x) and max(x).  If min or max is supplied,
       clamps x first.
       If center is supplied, instead scales values so that center->0.5, and
       [min, max] fit within [0,1] (i.e. scales by max difference from center)
    '''
    min = min if min is not None else np.min(x.flat)
    max = max if max is not None else np.max(x.flat)
    if center is None:
        x = np.maximum(np.minimum(x, max), min)
        return (x - min) / (max - min)
    else:
        x = (x - center)/np.maximum(np.abs(min - center), np.abs(max - center))
        return 0.5 * (x + 1)

def boxslice((i0, j0), (i1, j1)):
    '''Given top-left and bottom-right corners, returns array index slices for
       the box formed by these two points.
    '''
    return (slice(i0, i1), slice(j0, j1))

def filter_truncate(i, j, xshape, yshape):
    '''Given (i,j) center of filter y placed in x, and shapes (ilen, jlen) of 
       image x and filter y, returns slices for x and y s.t. y gets truncated
       at x's boundary.  Example:
       (xbox, ybox) = filter_truncate(i, j, recons.shape, filter.shape)
       recons[xbox] += k * filter[ybox]
    '''
    (xi, xj) = xshape
    (yi, yj) = yshape

    xi0 = i - yi//2
    xi1 = i + yi//2 + (int(yi) % 2)
    xj0 = j - yj//2
    xj1 = j + yj//2 + (int(yi) % 2)
    yi0 = 0
    yi1 = yi
    yj0 = 0
    yj1 = yj

    if xi0 < 0:
        yi0 -= xi0
        xi0 = 0
    if xi1 > xi:
        yi1 -= (xi1 - xi)
        xi1 = xi
    if xj0 < 0:
        yj0 -= xj0
        xj0 = 0
    if xj1 > xj:
        yj1 -= (xj1 - xj)
        xj1 = xj

    return (boxslice((xi0, xj0), (xi1, xj1)),
            boxslice((yi0, yj0), (yi1, yj1)))

def montage(imgs, layout=None, fill=0, border=0):
    '''Tiles given images together in a single montage image.
       imgs is an iterable of (h, w) or (h, w, c) arrays.
    '''
    sz = imgs[0].shape
    assert all([sz == x.shape for x in imgs])
    if len(sz) == 3:
        (h, w, c) = sz
    elif len(sz) == 2:
        (h, w) = sz
        c = 1
    else:
        raise ValueError('images must be 2 or 3 dimensional')

    bw = bh = 0
    if border:
        try:
            (bh, bw) = border
        except TypeError:
            bh = bw = int(border)
    nimgs = len(imgs)

    if layout is None:
        (ncols, nrows) = (None, None)
    else:
        (nrows, ncols) = layout

    if not (nrows and nrows > 0) and not (ncols and ncols > 0):
        if w >= h:
            ncols = np.ceil(np.sqrt(nimgs * h / float(w)))
            nrows = np.ceil(nimgs / float(ncols))
        else:
            nrows = np.ceil(np.sqrt(nimgs * w / float(h)))
            ncols = np.ceil(nimgs / float(nrows))
    elif not (nrows and nrows > 0):
        nrows = np.ceil(nimgs / float(ncols))
    elif not (ncols and ncols > 0):
        ncols = np.ceil(nimgs / float(nrows))

    mw = w * ncols + bw * (ncols-1)
    mh = h * nrows + bh * (nrows-1)
    assert mh * mw >= w*h*nimgs, 'layout not big enough to for images'
    M = np.zeros((mh, mw, c))
    M += fill
    i = 0
    j = 0
    for img in imgs:
        M[i:i+h, j:j+w, :] = img.reshape((h, w, c))
        j += w + bw
        if j >= mw:
            i += h + bh
            j = 0
    if len(sz) == 1:
        M = M.reshape((mh, mw))
    return M

def colormap(x, m=None, M=None, center=0, colors=None):
    '''color a grayscale array (currently red/blue by sign)'''
    if center is None:
        center = 0
    if colors is None:
        colors = np.array(((0, 0.7, 1),
                           (0,   0, 0),
                           (1,   0, 0)),
                          dtype=float)
    if x.shape[-1] == 1:
        x = x[..., 0]
    x = scale_values(x, min=m, max=M, center=center)
    y = np.empty(x.shape + (3,))
    for c in xrange(3):
        y[..., c] = np.interp(x, (0, 0.5, 1), colors[:, c])
    return y

def chan_to_pix(x, nchan=3, imsize=(1,1)):
    return (x.reshape((-1, nchan,) + imsize)
             .transpose((0,2,3,1))
             .reshape((-1, nchan)))

def pix_to_chan(x, nchan=3, imsize=(1,1)):
    return (x.reshape((-1,) + imsize + (nchan,))
             .transpose((0,3,1,2))
             .reshape((-1, nchan*imsize[0]*imsize[1])))

def bcxy_from_bxyc(im):
    return im.transpose((0,3,1,2))

def bxyc_from_bcxy(im):
    return im.transpose((0,2,3,1))

def bxyc_from_cxyb(im):
    return im.transpose((3,1,2,0))

def cxyb_from_bxyc(im):
    return im.transpose((3,1,2,0))

def filter_montage(imgs, m=None, M=None, center=None):
    (nf, nc) = imgs.shape[:2]

    if nc == 1:
        return montage(
            colormap(bxyc_from_bcxy(imgs), m, M, center),
            border=1,
            fill=0.2)
    elif nc == 3:
        return image_montage(imgs, m, M, center)
    else:
        imgs = imgs.reshape((nf*nc, 1,) + imgs.shape[2:])
        return montage(
            colormap(bxyc_from_bcxy(imgs), m, M, center),
            layout=(nf, nc),
            border=1,
            fill=0.2)

def image_montage(imgs, m=None, M=None, center=None):
    imgs = bxyc_from_bcxy(imgs)
    return montage(
        scale_values(imgs, m, M, center),
        border=1)

def acts_montage(acts, scale=True, nimgs=16, m=None, M=None):
    if nimgs:
        acts = acts[:nimgs]
    if len(acts.shape) == 2:
        acts = acts[:, :, np.newaxis, np.newaxis]
    if scale:
        inner_fill = 0.2
        outer_fill = 1.0
    else:
        inner_fill = np.min(acts) + 0.2 * (np.max(acts) - np.min(acts))
        outer_fill = np.max(acts)
    return montage([montage(
                       (scale_values(x, min=m, max=M)
                        if scale
                        else x),
                       border=1,
                       fill=inner_fill)
                    for x in acts],
                   border=3,
                   fill=outer_fill)

