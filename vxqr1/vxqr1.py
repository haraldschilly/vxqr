# -*- coding: utf8 -*-
#
#  VXQR1
#
#  Copyright 2015  Harald Schilly <harald.schilly@univie.ac.at>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import absolute_import, division, with_statement

import logging

import numpy as np
import numpy.linalg as LA
from time import time

from .utils import create_logger, VXQR1Exception

double_info = np.finfo(np.double)
eps = double_info.eps
realmin = double_info.tiny
realmax = double_info.max


class Result(object):

    def __init__(self, fbest, xbest, nf_used):
        self.fbest = fbest
        self.xbest = xbest
        self.nf_used = nf_used
        # best function value f after nf function values
        self.flist = []

    def __lt__(self, other):
        if isinstance(other, Result):
            return self.fbest < other.fbest
        raise ValueError("other must be of type 'Result'")

    def __repr__(self):
        return "Result f: %.4f after %d evaluations\n  x: %s" \
               % (self.fbest, self.nf_used, self.xbest)


class VXQR1(object):

    class Config(object):

        def __init__(self,
                     iscale=None,
                     stop_nf_target=None,
                     stop_f_target=None,
                     stop_max_sec=np.inf,
                     gain_factor=None,
                     discount_factor=None,
                     alp_tolerance=None,
                     alp_eps=None,
                     reg_factor_config=None):
            # initial scale: initially, the solution is sought
            # in a box of radius iscale around xinit
            self.iscale = max(iscale or 1., realmin)

            self.stop_nf_target = stop_nf_target
            self.stop_f_target = stop_f_target
            self.stop_max_sec = stop_max_sec

            # factor for trial stepsize from expected gain
            self.gain_factor = gain_factor or 2
            # discount factor for expected gain
            self.discount_factor = max(eps, discount_factor or .5)
            # tolerance in line search (defining lambdamin)
            self.alp_tolerance = max(eps, alp_tolerance or 1e-2)
            # tolerance in line search (defining first alp)
            self.alp_eps = max(eps, alp_eps or 1e-12)
            # regularization factor for QR
            self.reg_factor_config = reg_factor_config or 1e-4
            self.reg_factor = None  # set in _vxinit (depends on dimension n)

    def __init__(self, config, log_level=None):
        log_level = log_level or logging.DEBUG
        self.func = None
        self.log = create_logger("VXQR1", level=log_level)
        self.config = config
        # self.log.info("init")

    def solve(self,
              func,
              starting_point,
              lower_bounds,
              upper_bounds):
        self.start_time = self.last_eval_print = time()

        self.iscale = self.config.iscale
        self.func = func
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # some more global state objects
        self.trydir = None
        self.S = None

        lb = self.lower_bounds
        ub = self.upper_bounds

        if starting_point is None:
            starting_point = np.random.rand(*lb.shape) * (ub - lb) + lb
        self.x = self.starting_point = starting_point

        self.n = starting_point.shape[0]

        self.vxinit()
        result = self.vxmain()
        self.vxfinish()
        return result

    def vxinit(self):
        self.config.reg_factor = max(
            self.n * eps, self.config.reg_factor_config)

        if self.config.stop_nf_target is not None:
            self.nfmax = np.max(self.config.stop_nf_target)
            #imax = np.argmax(self.config.stop_nf_target)
            #self.config.stop_f_target[imax] = np.inf
            #self.config.stop_f_target = np.inf
        else:
            self.nfmax = 4.8 * self.n ** 2 + 200 * self.n
            self.config.stop_nf_target = self.nfmax
            #self.config.stop_f_target = np.inf
            self.config.stop_max_sec = np.inf
        self.nfmax = max(2, self.nfmax)
        self.vxinitf()
        self.x = self.starting_point[:]

        # initialize function value table
        self.nflog = np.ceil(np.log10(self.nfmax))
        flist = 10. ** np.arange(self.nflog + 1)
        flist = np.r_[flist, 2 * flist, 5 * flist]
        np.sort(flist)
        flist = flist[flist <= self.nfmax]
        flist = np.row_stack([flist, np.full(flist.shape[0], np.nan)])
        dflist = np.full(10, np.nan)
        self.flist = flist
        self.dflist = dflist

        # initial best point information
        self.nfused = 0
        self.nfbest = self.nfused
        self.xbest = self.x
        self.fbest = np.inf
        self.ilist = 0
        self.nfi = flist[0, self.ilist]
        stat = []
        self.overcount = 1.5
        self.deltaf = 1e6
        self.usestat = 0
        self.ns = self.n

        self.vxeval()

        self.onn = np.ones(self.n)
        self.X = self.xbest[:, np.newaxis]  # affine basis
        # affine function value matrix
        # 1x1 matrix, which will be expanded later on (using np.pad)
        self.F = np.array([[self.fbest]])

    def vxeval(self):
        '''
        evaluate function f=f(x) and keep books,
        but best point information must be updated outside
        :return:
        '''
        if self.done:  # TODO or time > self.max_secs
            self.f = f = np.nan
            return f
        if self.nfused >= self.config.stop_nf_target:
            self.f = f = np.nan
            return f

        if np.any(np.isnan(self.x)):
            #self.log.warning("x contains NaN")
            self.f = f = np.nan
            return f
        # project into the bounding box!
        self.x = np.minimum(self.upper_bounds,
                           np.maximum(self.lower_bounds, self.x))

        f = self.func(self.x)
        self.nfused += 1
        if self.nfused == self.nfi:
            self.flist[1, self.ilist] = self.fbest
            self.ilist += 1
            if self.ilist > self.flist.shape[1]:
                self.nfi = np.inf
            else:
                self.nfi = self.flist[1, self.ilist]

        # safeguard for huge function values
        if f > 1e15:
            if np.isfinite(f):
                f = 1e15 * (1 + np.log(f))
            else:
                f = 1e15 * (1 + realmax)

        # logging of progress, in the interval `dt` seconds
        dt = 2
        if self.last_eval_print + dt <= time():
            self.last_eval_print = time()
            pps = (self.nfused - self.last_nfused) / dt
            eta = (self.config.stop_nf_target - self.nfused) / pps
            pct = 100. * self.nfused / self.config.stop_nf_target
            eta = np.nan if pct < 20 else eta
            self.last_nfused = self.nfused
            self.log.info(
                "nfused: %5d (%3.0f%%)  pps: %4.f  eta: %4.fs" % (self.nfused, pct, pps, eta))

        self.f = f
        return f

    @property
    def done(self):
        stop = self.nfused >= self.config.stop_nf_target or \
            self.fbest < self.config.stop_f_target
        return stop

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        """
        To capture cases, where it is set to an (n x 1)-vector
        instead of an n-vector.
        :param v:
        :return:
        """
        assert len(v.shape) == 1
        v = np.minimum(self.upper_bounds,
                            np.maximum(self.lower_bounds, v))
        if np.all(v < self.lower_bounds) or np.all(v > self.upper_bounds):
            raise VXQR1Exception("x out of bounds")
        self._x = v

    def vxinitf(self, *xx):
        '''
        evaluate function at list of x-vectors `xx` and add to history plot.
        initialize if xx is empty
        :return:
        '''
        if len(xx) == 0:  # init
            self.nfused = 0
            self.last_nfused = 0
            self.fbest = np.inf
            return
        if self.nfused >= self.nfmax:
            self.f = 1e150 + np.zeros(len(xx))
            raise VXQR1Exception("too many function evaluations")
        else:
            self.f = np.apply_along_axis(self.func, 1, xx)
        self.nfused += len(xx)

        self.f_min = np.min(self.f)
        if self.nfused > 1:
            self.improved = self.f_min < self.fbest
        else:
            self.improved = True
        if self.improved:
            self.log.info("fval improved at nfused=%4d to f=%.2f" %
                          (self.nfused, self.f_min))

    def vxmain(self):
        """

        :param starting_point:
        :return type: Result
        """
        n = self.n
        lb = self.lower_bounds
        ub = self.upper_bounds

        if n == 1:
            namax = 2
        else:
            # maximal size of affine basis
            namax = max(3, min((n + 1) / 2, 11))

        # this gives 2 decimals in 100*n (for small n fewer) function values
        self.defacc = eps
        self.shortfrac = .5  # fraction of short scout steps
        self.maxgline = 100

        # rough initial scaling --  smallest point in box
        xorig = np.select([lb > 0, ub < 0], [lb, ub], default=0.)
        p = xorig - self.xbest
        if np.any(p == 0):
            # scaling line esearch
            self.alp = 1
            keepalp = True
            self.vxline(p, keepalp)
            if self.done:
                return

        # main iteration

        self.nit = 0
        foldit = self.fbest
        stuck = 0
        self.nstuck = 0
        self.nscout = 0
        fscout = [self.fbest]
        self.trygline = self.maxgline

        while True:

            # stopping test
            if self.done:
                break

            # update counters
            self.nit += 1
            # print(self.X)
            self.na = self.X.shape[1]
            if self.fbest < foldit:
                if stuck > 1:
                    # stuck
                    pass

                stuck = 0
            elif self.na == namax:
                if self.fbest < foldit:
                    stuck = 0
                else:
                    stuck += 1

                if stuck:
                    self.log.debug('stuck = %s ' % stuck)

            # fscout[0, nit] = self.fbest
            fscout.append(self.fbest)
            self.nfold = self.nfused

            # hill step
            self.na = self.X.shape[1]
            if np.mod(self.nit + 1, namax) == 0 and self.trygline >= 0:
                # fbest
                # disp('global coordinate search before scout')
                for coord in range(n):
                    self.vxgline(coord)

            # scout step
            self.ns = n // 5

            self.vxscout(self.nit)
            if self.done:
                break

            # find column to replace
            if self.na < namax:
                self.rc = self.na  # + 1
            else:
                F_diag = np.diag(self.F)
                self.rc = np.argmax(F_diag)
                fmax = F_diag[self.rc]
                if self.f >= fmax:
                    # (is essential in some cases even when trygline<<0!)
                    self.log.debug('global coordinate search after scout')
                    for coord in range(n):
                        self.vxgline(coord)

            self.vxsub()
            if self.done:
                break

        return Result(self.fbest, self.xbest, self.nfused)

    def vxsub(self):
        """
        updates row and column rc of affine quadratic model of size na,
        assuming that the other rows and columns are ok.
        The new point is assumed to be x and its function value f.
        """

        # self.X    # affine basis, n x na
        # self.F    # symmetric function matrix, na x na
        # F(i,k)=fcn((x_i+x_k)/2)
        self.fold = self.fbest
        self.nfold = self.nfused

        # update column rc of affine basis
        self.xrc = self.x
        if self.X.shape[1] <= self.rc:
            # we have to expand X with zero columns
            #Y = np.zeros((self.X.shape[0], self.rc - self.X.shape[1] + 1))
            #self.X = np.c_[self.X, Y]
            dx = self.rc - self.X.shape[1] + 1
            self.X = np.pad(self.X, ((0, 0), (0, dx)), "constant")
        self.X[:, self.rc] = self.x
        self.na = self.X.shape[1]

        if False:  # never applies before many stucks
            # check hull
            xinf = self.X.min(axis=1)
            xsup = self.X.max(axis=1)
            qhull = max((xsup - xinf) / (abs(xsup) + abs(xinf) + realmin))
            if qhull < 1e-8:
                # affine restart
                self.X = self.xbest[:, np.newaxis]
                self.F = np.array([[self.fbest]])

        # update row and column rc of function matrix
        if self.F.shape[0] < self.rc + 1 or self.F.shape[1] < self.rc + 1:
            self.F = np.pad(self.F,
                            ((0, self.rc + 1 - self.F.shape[0]),
                             (0, self.rc + 1 - self.F.shape[1])),
                            "constant")

        self.F[self.rc, self.rc] = self.f
        for iu in range(self.na):
            if iu == self.rc:
                continue
            self.x = (self.xrc + self.X[:, iu]) / 2
            # evaluate function
            self.vxeval()
            if self.f < self.fbest:
                # update best point information
                self.nfbest = self.nfused
                self.xbest = self.x
                self.fbest = self.f
                self.log.debug('improvement in pair step')

            self.F[self.rc, iu] = self.f
            self.F[iu, self.rc] = self.f

        # now f(Xs)=(2*s'*F-diag(F)')*s if sum(s)=1

        # find index of best basis point x_rc
        self.d = d = self.F.diagonal()  # np.diag(F)
        self.rc = np.argmin(d)
        self.frc = d[self.rc]

        # toggle direction type to be tried
        trydirmax = False
        if self.trydir is None:
            self.trydir = 0
        else:
            self.trydir += 1
            if self.trydir > trydirmax:
                self.trydir = 0

        if self.trydir == 0:
            self.vxnewton()  # safeguarded Newton direction
        else:
            self.vxcov()     # covariance matrix based direction

        # search direction
        p = self.X.dot(self.s) - self.xbest  # s from vxnewton
        if np.all(p == 0):
            return

        # line search
        self.vxline(p, self.alp, keepalp=False)
        if self.done:
            return

    def vxnewton(self):
        """
        find safeguarded Newton direction
        """

        # negative gradient at best basis point x_rc
        b = self.d / 4 - self.F[:, self.rc]

        # steepest descent direction
        sS = b - np.mean(b)
        sS = sS / (LA.norm(sS, np.inf) + realmin)

        # Newton direction (is a good spanning direction even when singular)
        # warning off
        #ona = np.ones((self.na, 1))
        delta = LA.norm(self.d - self.frc, np.inf)
        #delta = delta[ona, 1]
        delta = np.repeat(delta, self.na).reshape(self.na, -1)
        # sN=[F-frc delta;delta' 0]\[b;0]'
        sN_A = np.vstack((
            np.hstack((self.F - self.frc, delta)),
            # instead of delta.T, make it 1d horizontal vector!
            np.hstack((delta.reshape(-1), 0))
        ))
        try:
            sN = LA.solve(sN_A, np.r_[b, 0])
        except LA.linalg.LinAlgError:
            sN = np.ones((b.shape[0] + 1,)) * np.inf
        sN = sN[:-1]  # remove last column
        sN = sN - np.mean(sN)
        sN = sN / LA.norm(sN, np.inf)
        # warning on

        if np.all(np.isfinite(sN)):
            # minimize in the 2D subspace spanned by sS and sN
            # f(xnew+Qt) = fnew - 4 c2^T t + 2 t^T G2 t
            Q = np.vstack((sS, sN))
            c2 = Q.dot(b)
            G2 = Q.dot(self.F).dot(Q.T)
            t, self.nalp = self.vx2quad(c2, G2, self.n * self.defacc)
            s = Q.T.dot(t)
        else:
            # unstable Newton direction, use only steepest descent
            s = sS

        # shift to affine coordinates
        s[self.rc] = s[self.rc] + 1
        self.s = s

        self.falp = np.nan  # function value at new point unknown;
        return s

    def vx2quad(self, c, G, delta):
        """
        minimizes 2D problem f(x) = - 2 c^T x + x^T G x
        with safeguards in the nearly indefinite case.
        delta>=0 is a definiteness threshold

        The optimum is at x=2^nalp*t, where t in [-1,1]^2
        and nalp=1 iff either G is nearly infinite,
        or the definite optimizer is in [-1,1]^2.

        :param c: 
        :param G: 
        :param delta: 
        :return:
        """

        definite = False

        if G[0, 0] > 0:
            detG = G[0, 0] * G[1, 1] - G[0, 1] ** 2

            if detG > delta * G[0, 1] ** 2:
                # G is sufficiently positive definite
                definite = True
                # try unconstrained minimizer by Cramer's rule
                t = np.array(
                    [c[0] * G[1, 1] - c[1] * G[1, 0],
                     G[0, 0] * c[1] - G[0, 1] * c[0]])
                self.t = t / detG
                tn = LA.norm(t, np.inf)
                if not np.isfinite(tn):  # 111
                    # no definite direction found
                    self.log.debug('definite, but no definite direction found')
                    definite = False
                elif tn <= 2:  # 222
                    self.nalp = 1
                else:  # 333
                    # rescale direction
                    self.nalp = np.ceil(np.log2(tn))
                    t = t * 2 ** (1 - self.nalp)

                self.f2 = t.dot(G.dot(t) / 2 - c)

        if not definite:  # 444
            # G is not positive definite
            # find minimum constrained to the box [-1,1]^2
            f2 = 0  # objective value at t=0
            for k in [0, 1]:
                i = 1 - k
                for tk in [-1, 1]:
                    ci = c[i] - tk * G[i, k]
                    Gii = G[i, i]
                    # minimize -ci ti + Gii ti^2/2 s.t. ti in [-1,1]
                    if Gii > np.abs(ci):
                        ti = ci / Gii
                        fti = -ci * ti / 2
                    elif ci >= 0:
                        ti = 1
                        fti = -ci + Gii / 2
                    else:
                        ti = -1
                        fti = ci + Gii / 2

                    if fti < f2:
                        f2 = fti
                        self.t = np.zeros((2, 1))
                        self.t[i, 0] = ti
                        self.t[k, 0] = tk

            if f2 >= 0:
                # no good direction found
                self.log.info('bad fti: G:%s, c:%s' % (G, c))
                self.t = np.array([[1], [1]])
                self.nalp = 1
                return self.t, self.nalp

        self.nalp = 10
        return self.t, self.nalp

    def vxcov(self):
        raise Exception("does not exist")

    def vxgline(self, coord):
        """
        global line search decreasing fcn(xbest+alp*p) 

        coord:     % k for coordinate search in direction k,
                   % `None` for random direction search
        """

        self.nalp = 10       # number of uniform points tried (>=4)
        # the random scale recipe can be tuned as well
        ub = self.upper_bounds
        lb = self.lower_bounds

        if coord is not None and coord >= 0:
            # coordinate direction
            pp = np.zeros((self.n, 1))
            pp[coord] = ub[coord] - lb[coord]
            ind = np.all(np.isfinite(pp))
            pp[ind] = 1
        else:
            # random search direction
            pp = 2 * (self.vxrand(lb, ub) - self.xbest)

        # convert n x 1 - vectors to an n-vector
        pp = pp.ravel()

        # search range alp in [-1,1]
        n2 = np.fix(self.nalp / 2)
        self.nalp = int(2 * n2 + 1)

        # global grid search
        glgood = 0

        for rep in range(10):
            fgline = self.fbest

            # random scale
            r = np.random.random() ** 2
            p = pp * r
            asorted = (
                np.arange(-n2, n2 + 1) + 0.8 * np.random.random(self.nalp) - 0.4) / n2
            asorted[n2 + 1] = 0
            fsorted = np.inf * asorted
            x0 = self.xbest

            for kk in range(self.nalp):
                alp = asorted[kk]
                if alp == 0:
                    fsorted[kk] = self.fbest
                    continue

                # function evaluation and list management
                self.x = x0 + asorted[kk] * p
                self.vxeval()  # evaluate f=f(x) and keep books
                if self.done:
                    break
                fsorted[kk] = self.f
                if self.f < self.fbest:
                    # update best point information
                    self.fbest = self.f
                    self.xbest = self.x
                    self.nfbest = self.nfused

            kbest = np.argmin(fsorted)
            ffbest = fsorted[kbest]
            if self.fbest < fgline:
                # best point moved on the grid
                glgood = glgood + 1
            else:
                break

        # now the best grid point is at alp=0

        bracket = 1
        nblist = 0
        blist = []
        fblist = []
        for kk in range(2, self.nalp - 2):

            f0 = fsorted[kk]
            f1 = fsorted[kk - 1]
            f2 = fsorted[kk + 1]
            if f0 > min(f1, f2):
                # not a local minimizer, do nothing
                continue

            # safeguarded quadratic interpolation step
            a00 = asorted[kk]
            a1 = asorted[kk - 1] - a00
            a2 = asorted[kk + 1] - a00
            self.vxquad()
            anew = [int(alp + a00)]

            # piecewise linear interpolation steps
            kink = -1
            kink = self.vxkink(kink, kk, asorted, fsorted)
            if kink:
                anew.append(self.alp)
            kink = +1
            kink = self.vxkink(kink, kk, asorted, fsorted)
            if kink:
                anew.append(self.alp)

            for alp in anew:
                # function evaluation and list management
                self.x = x0 + np.dot(alp, p)
                self.vxeval()  # evaluate f=f(x) and keep books
                if self.done:
                    break
                if self.f < self.fbest:
                    # update best point information
                    self.fbest = self.f
                    self.xbest = self.x
                    self.nfbest = self.nfused

                #nblist = nblist + 1
                blist.append(alp)
                fblist.append(self.f)

        if glgood > 0:
            self.trygline = self.maxgline
        else:
            self.trygline = self.trygline - 1

    def vxscout(self, nit):
        """
        creates an improved point using ns random line searches
        :return:
        """
        # self.log.debug('scout step')

        # note this currently fixes ns=n!

        if self.S is None:
            # create scout set
            ns = self.n
            self.S = 2 * np.random.rand(self.n, self.n) - 1
            self.sscale = self.iscale

        assert self.sscale is not None

        # create non-affine basis
        ns = self.S.shape[1]
        #ons = np.ones((ns, 1))
        #S, R = LA.qr(self.S - self.xbest[:, ons])
        self.S, self.R = LA.qr(self.S - self.xbest[:, np.newaxis])

        # determine scale
        if self.sscale == 0:
            scal = self.iscale
        else:
            scal = self.sscale

        sscale = scal / np.arange(1, self.n + 1)

        # search in each basis direction
        for k in range(self.n):
            # correctly scaled search direction
            p = sscale[k] * self.S[:, k]

            # line search towards scout reference point
            self.vxline(p, alp=1, keepalp=True)
            if self.done:
                break

            # update basis (the used part of the basis is affine)
            self.S[:, k] = self.x
            # fS[k]=self.f # TODO fS only used here, what's going on?
            sscale[k] = abs(self.alp) * sscale[k]

        self.log.debug('nf=%s, nit=%s, fbest=%s' %
                       (self.nfused, self.nit, self.fbest))

    def vxline(self, p, alp, keepalp):
        """
        line search decreasing fcn(xbest+alp*p)
        :return:
        """
        assert isinstance(p, np.ndarray)
        n = self.n

        nalist = 6  # maximal number of trial points (at least 4 needed)
        qlarge = 10  # asymmetry quotient considered large
        alpeps = 1e-12  # limit accuracy for convex regularizartion
        alpexp = 1  # damping factor for extrapolation
        # alptol could also be varied
        lineview = 0

        abest = 0
        a0 = 0
        convex = 0

        # select initial step
        if np.all(p == 0) or np.any(np.isfinite(p)):
            self.log.debug('replace zero direction')

            p = self.iscale * (np.random.rand(n) - 0.5)
            keepalp = False

        ind = self.xbest != 0
        ascale = np.min(np.abs(self.xbest[ind]) / (np.abs(p[ind]) + realmin))
        alpdist = 1e-8 * ascale
        if keepalp:
            # use current alp
            pass
        elif np.any(ind):
            alptol = 10. ** (np.random.random() - 2.)
            alp = alptol * ascale
        else:
            alp = self.iscale / LA.norm(p, np.inf)

        self.alist = alist = np.array([0])
        self.falist = falist = np.array([self.fbest])
        for iline in range(nalist):
            nalp = len(alist)
            alps = LA.norm(alist - abest, np.inf)
            if nalp > 2:
                # cut new step to size
                alp = max(-100 * alps, min(alp, 100 * alps))

            # if not np.isfinite(alp): # TODO Achtung, möglicherweise Änderung
            # xbest
            # p
            # ascale
            # alp=rand;
            # input('Next>');

            # function evaluation and list management
            self.x = self.xbest + alp * p
            if any(np.isnan(self.x)):
                self.log.debug("xbest: %s; alp: %s, p: %s" %
                               (self.xbest, alp, p))

            f = self.vxeval()  # evaluate f=f(x) and keep books
            if self.done:
                break
            alist = np.append(alist, abest + alp)
            falist = np.append(falist, f)
            newbest = f < self.fbest
            if newbest:
                # update best point information
                self.fbest = f
                self.xbest += alp * p
                abest = abest + alp
                self.nfbest = self.nfused

            # quit line search?
            if (convex and newbest) or len(alist) > nalist:
                # this may still be on a monotone stretch!

                # update dflist
                df = falist[0] - self.fbest
                if df > 0:
                    dflist = np.r_[df, self.dflist[:-1]]

                # quit line search
                break

            # determine next step size
            nalp = alist.shape[0]
            if nalp == 2:
                # extrapolate to better side
                if falist[0] < falist[1]:
                    alp = alpexp * (alist[0] - alist[1])
                else:
                    alp = alpexp * (alist[1] - alist[0])

                continue

            # sort points and find bracket
            a = alist - abest
            perm = np.argsort(a)
            asorted = a[perm]
            fsorted = falist[perm]
            kk, = np.nonzero(asorted == 0.0)
            a0 = asorted[kk]
            f0 = fsorted[kk]  # best value
            if kk > 0:
                left = True
                a1 = asorted[kk - 2]
                f1 = fsorted[kk - 2]
            else:
                left = False
                a1 = asorted[kk + 1]
                f1 = fsorted[kk + 1]

            if kk < nalp or kk < 2:
                right = True
                a2 = asorted[kk + 0]
                f2 = fsorted[kk + 0]
            else:
                right = False
                a2 = asorted[kk - 1]
                f2 = fsorted[kk - 1]

            self.bracket = left and right
            self.a0 = a0
            self.f0 = f0
            self.a1 = a1
            self.a2 = a2
            self.f1 = f1
            self.f2 = f2

            # linear kink interpolation applicable?
            if nalp == 5 and kk == 3:
                # if kk>=3 and kk<=nalp-2:
                kink = False  # without central point
                kink = self.vxkink(kink, kk, asorted, fsorted)
                if kink:
                    # if lineview, disp('kink step'); end;
                    continue

            # geometric mean step applicable?
            if nalp > 4:
                q = -a1 / a2
                if lineview and q > 0:
                    qq = max(q, 1 / q)
                if q > qlarge:
                    alp = -a2 * (1 + np.random.random() * np.sqrt(q))
                    self.log.debug('geometric mean step A')
                    continue
                elif 1 / q > qlarge:
                    alp = a1 / (1 + np.random.random() * np.sqrt(q))
                    self.log.debug('geometric mean step B')
                    continue

            # try safeguarded quadratic interpolant
            if nalp > 3:
                # restrict step by setting a gain target
                self.log.debug('restricted step')
                self.linetarget = 30 * \
                    max(self.dflist[0], np.median(self.dflist))
            else:
                self.linetarget = np.inf

            self.vxquad()

        # produce output referring to the original xbest
        if not abest:
            alp = self.a1
            f = self.f1
            x = self.xbest + self.a1 * p
        else:
            alp = abest
            f = self.fbest
            x = self.xbest

        self.alp = alp
        self.f = f
        self.x = x

    def vxquad(self):
        """
        finds alp for a safeguarded quadratic interpolation step
        through a0=0, a1, a2 with function values f0, f1, f2
        :return:
        """

        alpeps = 1e-12  # limit accuracy for convex regularization

        # get slopes
        da1 = (self.f1 - self.fbest) / self.a1
        da2 = (self.f2 - self.fbest) / self.a2

        # get interpolating quadratic model
        # f(xbest+alp*p)=fbest-alp*kappa+alp^2*lambda

        fbest = self.fbest
        a1 = self.a1
        a2 = self.a2
        f1 = self.f1
        f2 = self.f2

        try:
            alpf = max(self.falist) - fbest + eps * np.abs(fbest) / \
                max(abs(self.a1), abs(self.a2)) ** 2
        except:
            # required info not present -- replace by random step
            alp = a1 + np.random.random() * (a2 - a1)
            return

        lambdamin = alpeps * alpf
        lambda_ = (da2 - da1) / (a2 - a1)
        kappa = a1 * lambda_ - da1
        kappa2 = kappa / 2
        convex = self.bracket or (lambda_ > lambdamin)
        if False:
            condinv = np.min([(f1 - fbest) / (abs(f1) + abs(fbest)),
                              (f2 - fbest) / (abs(f2) + abs(fbest)),
                              (da2 - da1) / (abs(da2) + abs(da1))])

        if np.isfinite(self.linetarget):
            # get maximal step with predicted gain <= linetarget
            discr = kappa2 ** 2 - lambda_ * self.linetarget
            if discr > 0:
                if kappa2 < 0:
                    denom = kappa2 - np.sqrt(discr)
                else:
                    denom = kappa2 + np.sqrt(discr)

                alp = self.linetarget / denom
            elif lambda_ > 0:
                alp = kappa2 / lambda_
            else:
                alp = 0  # flat function

                # alp hier weiter
        else:
            # unrestricted case
            # get safeguarded convex quadratic model
            lambda_ = max(lambda_, lambdamin)
            kappa = a1 * lambda_ - da1
            # predicted optimal step size
            alp = kappa / (2 * lambda_)

        oldrep = alp == 0 or alp == a1 or alp == a2
        if oldrep:
            # replace by random step
            alp = a1 + np.random.random() * (a2 - a1)

        self.alp = alp

    def vxkink(self, kink, kk, asorted, fsorted):
        """
        finds alp for a piecewise linear kink interpolation step

        :return: boolean kink
        """
        kl = kr = None
        kk -= 1
        if kink == 0:
            # without central point
            kl = kk - 1
            kr = kk + 1
        elif kink == -1:
            # center to left
            kl = kk
            kr = kk + 1
        elif kink == 1:
            # center to right
            kl = kk - 1
            kr = kk

        # intersect interpolant of kk-{1,2} with that of kk+{1,2}
        al = asorted[kl]
        fl = fsorted[kl]
        dl = (fsorted[kl - 1] - fl) / (asorted[kl - 1] - al)
        ar = asorted[kr]
        fr = fsorted[kr]
        dr = (fsorted[kr + 1] - fr) / (asorted[kr + 1] - ar)
        kink = bool(dl * dr < 0)
        if kink:
            alp = (fl - dl * al - fr + dr * ar) / (dr - dl)
            if alp < asorted[kl]:
                alp = asorted[kr] / 100
            if alp > asorted[kr]:
                alp = asorted[kl] / 100
            self.alp = alp

        return kink

    def vxfinish(self):

        if self.f < self.fbest:
            # update best point information
            self.fbest = self.f
            self.xbest = self.x
            self.nfbest = self.nfused

        # reduce flist to its filled part
        # flist(:,isnan(flist(2,:)))=[];
        self.flist = self.flist[:, ~np.isnan(self.flist[1, :])]

    def vxrand(self, low, upp):
        """
        returns a random vector x in [low,upp]
        uniform in the components where upp-low is finite,
        and nonuniform otherwise
        """
        randn = np.random.randn
        rand = np.random.rand
        x = low + np.random.rand(len(low)) * (upp - low)

        ind, = np.nonzero(~np.isfinite(x))
        for i in ind:
            if np.isinf(upp[i]):
                if np.isinf(low[i]):
                    x[i] = randn()
                elif low[i] == 0:
                    x[i] = abs(randn())
                elif low[i] > 0:
                    x[i] = low[i] * (1 + abs(randn()))
                elif rand() < 0.5:
                    x[i] = rand() * low[i]
                else:
                    x[i] = abs(randn)

            else:  # low[i]=-inf, upp[i] finite
                if upp[i] == 0:
                    x[i] = -abs(randn())  # was -abs(rd)
                elif upp[i] < 0:
                    x[i] = upp[i] * (1 + abs(randn()))
                elif rand < 0.5:
                    x[i] = rand * upp[i]
                else:
                    x[i] = -abs(randn())

        return x
