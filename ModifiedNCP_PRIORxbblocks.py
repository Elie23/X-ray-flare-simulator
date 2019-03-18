# Copyright 2013 Peter Williams
# Licensed under the GNU General Public License version 3 or higher

"""
Modified from PW codes: Now takes in the ncp_prior directly instead of using the one from Scargle 2013

xbblocks - extended Bayesian Blocks
Bayesian Blocks analysis for the "time tagged" case described by Scargle+ 2013.
Inspired by the bayesian_blocks implementation by Jake Vanderplas in the AstroML
package, but that turned out to have some limitations.
We have iterative determination of the best number of blocks (using an ad-hoc
routine described in Scargle+ 2013) and bootstrap-based determination of
uncertainties on the block heights (ditto).
"""

import numpy as np

__all__ = ['nlogn binbblock ttbblock bsttbblock']

## quickutil: holder
#- snippet: holder.py (2012 Sep 29)
#- SHA1: bc9ad74474ffc74f18a12675f7422f0c5963df59
class Holder (object):
    def __init__ (self, **kwargs):
        self.set (**kwargs)

    def __str__ (self):
        d = self.__dict__
        s = sorted (d.iterkeys ())
        return '{' + ', '.join ('%s=%s' % (k, d[k]) for k in s) + '}'

    def __repr__ (self):
        d = self.__dict__
        s = sorted (d.iterkeys ())
        return '%s(%s)' % (self.__class__.__name__,
                           ', '.join ('%s=%r' % (k, d[k]) for k in s))

    def set (self, **kwargs):
        self.__dict__.update (kwargs)
        return self

    def get (self, name, defval=None):
        return self.__dict__.get (name, defval)

    def setone (self, name, value):
        self.__dict__[name] = value
        return self

    def has (self, name):
        return name in self.__dict__

    def copy (self):
        new = self.__class__ ()
        new.__dict__ = dict (self.__dict__)
        return new
## end


def nlogn (n, dt):
    # I really feel like there must be a cleverer way to do this
    # scalar-or-vector possible-bad-value masking.

    if np.isscalar (n):
        if n == 0:
            return 0.
        return n * (np.log (n) - np.log (dt))

    n = np.asarray (n)
    mask = (n == 0)
    r = n * (np.log (np.where (mask, 1, n)) - np.log (dt))
    return np.where (mask, 0, r)


def binbblock (widths, counts, prior):#p0=0.05):
    widths = np.asarray (widths)
    counts = np.asarray (counts)
    ncells = widths.size
    #origp0 = p0

    if np.any (widths <= 0):
        raise ValueError ('bin widths must be positive')
    if widths.size != counts.size:
        raise ValueError ('widths and counts must have same size')
    #if p0 < 0 or p0 >= 1.:
    #    raise ValueError ('p0 must lie within [0, 1)')

    vedges = np.cumsum (np.concatenate (([0], widths))) # size: ncells + 1
    block_remainders = vedges[-1] - vedges # size: nedges = ncells + 1
    ccounts = np.cumsum (np.concatenate (([0], counts)))
    count_remainders = ccounts[-1] - ccounts

    prev_blockstarts = None
    best = np.zeros (ncells, dtype=np.float)
    last = np.zeros (ncells, dtype=np.int)

    #for _ in range (10):
        # Pluggable num-change-points prior-weight expression:
    ncp_prior = prior#4 - np.log (p0 / (0.0136 * ncells**0.478))

    for r in range (ncells):
        tk = block_remainders[:r+1] - block_remainders[r+1]
        nk = count_remainders[:r+1] - count_remainders[r+1]

        # Pluggable fitness expression:
        try:
            fit_vec = nlogn (nk, tk)
        except:
            print ('q', nk, tk)
            print ('r', widths, counts)
            raise

    # This incrementally penalizes partitions with more blocks:
        tmp = fit_vec - ncp_prior
        tmp[1:] += best[:r]

        imax = np.argmax (tmp)
        last[r] = imax
        best[r] = tmp[imax]

    # different semantics than Scargle impl: our blockstarts is similar to
    # their changepoints, but we always finish with blockstarts[0] = 0.

    work = np.zeros (ncells, dtype=int)
    workidx = 0
    ind = last[-1]

    while True:
        work[workidx] = ind
        workidx += 1
        if ind == 0:
            break
        ind = last[ind - 1]

    blockstarts = work[:workidx][::-1]

    #if prev_blockstarts is not None:
    #	if (blockstarts.size == prev_blockstarts.size and (blockstarts == prev_blockstarts).all ()):
#		break # converged

    #if blockstarts.size == 1:
    #	break # can't shrink any farther

    # Recommended ad-hoc iteration to favor fewer blocks above and beyond
    # the value of p0:
    #p0 = 1. - (1. - p0)**(1. / (blockstarts.size - 1))
    #prev_blockstarts = blockstarts

    assert blockstarts[0] == 0
    nblocks = blockstarts.size

    info = Holder ()
    info.ncells = ncells
    info.nblocks = nblocks
    #info.origp0 = origp0
    info.prior = prior
    info.blockstarts = blockstarts
    info.counts = np.empty (nblocks, dtype=np.int)
    info.widths = np.empty (nblocks)

    for iblk in range (nblocks):
        cellstart = blockstarts[iblk]
        if iblk == nblocks - 1:
            cellend = ncells - 1
        else:
            cellend = blockstarts[iblk+1] - 1

        info.widths[iblk] = widths[cellstart:cellend+1].sum ()
        info.counts[iblk] = counts[cellstart:cellend+1].sum ()

    info.rates = info.counts / info.widths
    return info


def ttbblock (tstarts, tstops, times, prior):
    tstarts = np.asarray (tstarts)
    tstops = np.asarray (tstops)
    times = np.asarray (times)

    if tstarts.size != tstops.size:
        raise ValueError ('must have same number of starts and stops')

    ngti = tstarts.size

    if ngti < 1:
        raise ValueError ('must have at least one goodtime interval')
    if np.any ((tstarts[1:] - tstarts[:-1]) <= 0):
        raise ValueError ('tstarts must be ordered and distinct')
    if np.any ((tstops[1:] - tstops[:-1]) <= 0):
        raise ValueError ('tstops must be ordered and distinct')
    if np.any (tstarts >= tstops):
        raise ValueError ('tstarts must come before tstops')
    if np.any ((times[1:] - times[:-1]) < 0):
        raise ValueError ('times must be ordered')
    if times.min () < tstarts[0]:
        raise ValueError ('no times may be smaller than first tstart')
    if times.max () > tstops[-1]:
        raise ValueError ('no times may be larger than last tstop')
    #for i in range (1, ngti):
    #    if np.where ((times > tstops[i-1]) & (times < tstarts[i]))[0].size:
    #        raise ValueError ('no times may fall in goodtime gap #%d' % i)
    #if p0 < 0 or p0 >= 1.:
    #    raise ValueError ('p0 must lie within [0, 1)')

    utimes, uidxs = np.unique (times, return_index=True)
    nunique = utimes.size
    #print times, utimes, uidxs
    counts = np.empty (nunique)
    counts[:-1] = uidxs[1:] - uidxs[:-1]
    counts[-1] = times.size - uidxs[-1]
    assert counts.sum () == times.size

    #print counts
    # we grow these arrays with concats, which will perform badly with lots of
    # GTIs. Not expected to be a big deal.
    widths = np.empty (0)
    ledges = np.empty (0)
    redges = np.empty (0)

    for i in range (ngti):
        tstart, tstop = tstarts[i], tstops[i]

        w = np.where ((utimes >= tstart) & (utimes <= tstop))[0]

        if not w.size:
            # No events during this goodtime! We have to insert a zero-count
            # event block. This may break assumptions within binbblock()?

            # j = idx of first event after this GTI
            wafter = np.where (utimes > tstop)[0]
            if wafter.size:
                j = wafter[0]
            else:
                j = utimes.size
            assert j == 0 or np.where (utimes < tstart)[0][-1] == j - 1

            counts = np.concatenate ((counts[:j], [0], counts[j:]))
            widths = np.concatenate ((widths, [tstop - tstart]))
            ledges = np.concatenate ((ledges, [tstart]))
            redges = np.concatenate ((redges, [tstop]))
        else:
            gtutimes = utimes[w]
            midpoints = 0.5 * (gtutimes[1:] + gtutimes[:-1]) # size: n - 1
            gtedges = np.concatenate (([tstart], midpoints, [tstop])) # size: n + 1
            gtwidths = gtedges[1:] - gtedges[:-1] # size: n
            #assert gtwidths.sum () == tstop - tstart
            widths = np.concatenate ((widths, gtwidths))
            ledges = np.concatenate ((ledges, gtedges[:-1]))
            redges = np.concatenate ((redges, gtedges[1:]))

    assert counts.size == widths.size
    #print widths
    info = binbblock (widths, counts, prior)
    info.ledges = ledges[info.blockstarts]
    # The right edge of the i'th block is the right edge of its rightmost
    # bin, which is the bin before the leftmost bin of the (i+1)'th block:
    info.redges = np.concatenate ((redges[info.blockstarts[1:] - 1], [redges[-1]]))
    info.midpoints = 0.5 * (info.ledges + info.redges)
    return info


def bsttbblock (times, tstarts, tstops, prior, nbootstrap=512):
    np.seterr ('raise')
    times = np.asarray (times)
    tstarts = np.asarray (tstarts)
    tstops = np.asarray (tstops)

    nevents = times.size
    if nevents < 1:
        raise ValueError ('must be given at least 1 event')

    info = ttbblock (tstarts, tstops, times, prior)

    # Now bootstrap resample to assess uncertainties on the bin heights. This
    # is the approach recommended by Scargle+.
    if nbootstrap > 0 :
        bsrsums = np.zeros (info.nblocks)
        bsrsumsqs = np.zeros (info.nblocks)

        for _ in range (nbootstrap):
            bstimes = times[np.random.randint (0, times.size, times.size)]
            bstimes.sort ()
            bsinfo = ttbblock (tstarts, tstops, bstimes, prior)
            blocknums = np.minimum (np.searchsorted (bsinfo.redges, info.midpoints),
                                bsinfo.nblocks - 1)
            samprates = bsinfo.rates[blocknums]
            #print 'here',info.nblocks,bsinfo.nblocks, bsinfo.redges, info.midpoints, blocknums,bsinfo.rates, samprates
            bsrsums += samprates
            bsrsumsqs += samprates**2

        bsrmeans = bsrsums / nbootstrap
        mask = bsrsumsqs / nbootstrap <= bsrmeans**2
        bsrstds = np.sqrt (np.where (mask, 0, bsrsumsqs / nbootstrap - bsrmeans**2))
        info.bsrates = bsrmeans
        info.bsrstds = bsrstds
    return info
