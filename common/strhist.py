'''
strhist.py

Prints histograms using text.
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

_strhist_chars = ' .oO@^'

def _gethist(x, bins, m, M):
    x = np.array(x)
    if m == None:
        m = x.min()
    if M == None:
        M = x.max()
    (h, hbins) = np.histogram(x, bins=bins, range=(m,M))
    h = h.astype(float)
    h /= np.sum(h)
    return (h, hbins, m, M)

def hist_chars(x, m=None, M=None, width=50):
    '''
    Prints a one-line histogram with one char per bin.  The bin count is
    quantized into only a few values and scaled to create a visual
    representation.  Min and max values are displayed on the ends.
    '''
    (h, hbins, m, M) = _gethist(x, width, m, M)
    nchars = len(_strhist_chars)
    if np.any(h > 0):
        hmin = np.min(h)
        hmax  = np.max(h)
        hchar = np.round((nchars-1)*(h - hmin)/(hmax - hmin))
        hstr = ''.join([_strhist_chars[int(i)] for i in hchar])
    else:
        hstr = ' ' * width
    return '% .5f |%s| %.5f' % (m, hstr, M)

def hist_bins(x, m=None, M=None, width=50, sep=''):
    '''
    Prints a one-line histogram with a percent in each bin.
    Min and max values are displayed on the ends.
    '''
    w = 7
    bins = width / w
    (h, hbins, m, M) = _gethist(x, bins, m, M)
    hstr = sep.join([str(int(np.round(x*100))).center(w-2) for x in h])
    return '% .2f ||%s|| %.2f' % (m, hstr, M)

def hist_bars(x, m=None, M=None, bins=10, width=50):
    '''
    Prints a histogram with one bin per line.
    '''
    (h, hbins, m, M) = _gethist(x, bins, m, M)
    barlengths = np.round(width * h / np.maximum(1e-8, np.max(h)))
    s = ['% .3f ~ % .3f | %s' % (hbins[i], hbins[i+1], '*' * barlengths[i])
         for i in xrange(len(h))]
    return '\n'.join(s)

strhist = hist_chars
hist = hist_chars

if __name__ == '__main__':
    x = np.random.randn(10000)
    for fname in ('hist_chars', 'hist_bins', 'hist_bars'):
        print fname
        print eval(fname)(x)
        print

