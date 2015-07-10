"""
Client Library for the BBCOMP2015 challange.

Set your username/password in the environment variables
BBCOMP_USER and BBCOMP_PASS.
"""
from __future__ import print_function
from ctypes import *
from numpy.ctypeslib import ndpointer
from numpy import *
import sys
import platform

# get library name
dllname = ""
if platform.system() == "Windows":
    dllname = "./bbcomp.dll"
elif platform.system() == "Linux":
    dllname = "./libbbcomp.so"
elif platform.system() == "Darwin":
    dllname = "./libbbcomp.dylib"
else:
    sys.exit("unknown platform")

# initialize dynamic library
bbcomp = CDLL(dllname)
bbcomp.configure.restype = c_int
bbcomp.login.restype = c_int
bbcomp.numberOfTracks.restype = c_int
bbcomp.trackName.restype = c_char_p
bbcomp.setTrack.restype = c_int
bbcomp.numberOfProblems.restype = c_int
bbcomp.setProblem.restype = c_int
bbcomp.dimension.restype = c_int
bbcomp.budget.restype = c_int
bbcomp.evaluations.restype = c_int
bbcomp.evaluate.restype = c_int
bbcomp.evaluate.argtypes = [ndpointer(
    c_double, flags="C_CONTIGUOUS"), ndpointer(c_double, flags="C_CONTIGUOUS")]
bbcomp.history.restype = c_int
bbcomp.history.argtypes = [c_int, ndpointer(
    c_double, flags="C_CONTIGUOUS"), ndpointer(c_double, flags="C_CONTIGUOUS")]
bbcomp.errorMessage.restype = c_char_p


print("----------------------------------------------")
print("     VXQR1 BBComp 2015 competition client     ")
print("----------------------------------------------")
print()

# set configuration options (this is optional)
result = bbcomp.configure(1, "logs/")
if result == 0:
    sys.exit("configure() failed: " + bbcomp.errorMessage())

# login with demo account - this should grant access to the "trial" track
# (for testing and debugging)

import os
USER = os.environ.get("BBCOMP_USER")  # , "demoaccount")
PASS = os.environ.get("BBCOMP_PASS")  # , "demopassword")
result = bbcomp.login(USER, PASS)
if result == 0:
    sys.exit("login() failed: %s" % bbcomp.errorMessage())

print("login as %s successful" % USER)

# request the tracks available to this user (this is optional)
numTracks = bbcomp.numberOfTracks()
if numTracks == 0:
    sys.exit("numberOfTracks() failed: " + bbcomp.errorMessage())

print(numTracks, " track(s):")
for i in range(numTracks):
    trackname = bbcomp.trackName(i)
    if bool(trackname) == False:
        sys.exit("trackName() failed: %s" % bbcomp.errorMessage())

    print("  ", i, ": ", trackname)

# set the track to "trial"
#track = "trial"
#track = "BBComp2015CEC"
track = "BBComp2015GECCO"
result = bbcomp.setTrack(track)
if result == 0:
    sys.exit("setTrack() failed: %s" % bbcomp.errorMessage())
print("track set to %s" % track)

# obtain number of problems in the track
numProblems = bbcomp.numberOfProblems()
if numProblems == 0:
    sys.exit("numberOfProblems() failed: %s" % bbcomp.errorMessage())

print("The track consists of %s problems." % numProblems)


# global allocation of memory for a search point
# point = zeros(dim)
value = zeros(1)

# run the optimization loop
#bestValue = 1e100
#bestEvaluation = -1


from vxqr1 import VXQR1
from vxqr1.utils import VXQR1Exception
import numpy as np
import logging

for problemID in range(numProblems):
    result = bbcomp.setProblem(problemID)
    if result == 0:
        sys.exit("setProblem() failed: " + bbcomp.errorMessage())

    print("Problem %4d [of %4d] " % (problemID, numProblems), end="")

    bud = bbcomp.budget()
    if bud == 0:
        sys.exit("budget() failed: %s" % bbcomp.errorMessage())

    evals = bbcomp.evaluations()
    if evals < 0:
        sys.exit("evaluations() failed: %s" % bbcomp.errorMessage())

    if evals >= bud:
        print("all evaluations used up for problem %4d in %s!" %
              (problemID, track))
        continue

    # obtain problem properties
    dim = bbcomp.dimension()
    if dim == 0:
        sys.exit("dimension() failed: %s" % bbcomp.errorMessage())

    pctevals = 100. * evals / bud
    print("dim: %3d  budget: %5d  used evals: %5d  (%4.1f%%)" %
          (dim, bud, evals, pctevals))

    n = dim
    nf_max = bud - evals
    starting_point = np.random.rand(n)
    lower_bounds = np.zeros(n)
    upper_bounds = np.ones(n)

    vx1config = VXQR1.Config(
        iscale=np.max([np.linalg.norm(lower_bounds, np.inf),
                       np.linalg.norm(upper_bounds, np.inf)]),
        stop_nf_target=nf_max,  # np.array([nf_max, 0]),
        stop_f_target=-np.inf  # np.array([np.inf, f_target])
    )

    vx1 = VXQR1(vx1config, log_level=logging.INFO)

    def func(x):
        result = bbcomp.evaluate(x, value)
        if result == 0:
            print("x: %r" % x)
            raise VXQR1Exception("evaluate() failed: %s after %s" %
                                 (bbcomp.errorMessage(), bbcomp.evaluations()))
        v = value[0]
        return v

    result = vx1.solve(func,
                       starting_point,
                       lower_bounds=lower_bounds,
                       upper_bounds=upper_bounds
                       )
    print("VXQR1 finished. Result: %s" % result)


# for e in range(bud):
#     if e < evals:
#         # If evals > 0 then we have already optimized this problem to some point.
#         # Maybe the optimizer crashed or was interrupted.
#         #
#         # This code demonstrates a primitive recovery approach, namely to replay
#         # the history as if it were the actual black box queries. In this example
#         # this affects only "bestValue" since random search does not have any
#         # state variables.
#         # As soon as e >= evals we switch over to black box queries.
#         result = bbcomp.history(e, point, value)
#         if result == 0:
#             sys.exit("history() failed: " + bbcomp.errorMessage())
#
#         # In any real algorithm "point" and "value" would update the internals
#         # state.
#         if value[0] < bestValue:
#             bestValue = value[0]
#             bestEvaluation = e
#
#     else:
#         # define a search point, here uniformly at random
#         point = random.rand(dim)
#
#         # query the black box
#         result = bbcomp.evaluate(point, value)
#         if result == 0:
#             sys.exit("evaluate() failed: " + bbcomp.errorMessage())
#
#         print("[", e, "] ", value[0])
#
#         if value[0] < bestValue:
#             bestValue = value[0]
#             bestEvaluation = e

# print the best point
#print("best value: ", bestValue)
#print("best iteration: ", bestEvaluation)
#result = bbcomp.history(bestEvaluation, point, value)
# if result == 0:
#    sys.exit("history() failed: " + bbcomp.errorMessage())

#print("best point:", point)

# check that we are indeed done
#evals = bbcomp.evaluations()
# if evals == bud:
#    print("optimization finished.")
# else:
#    print(
#        "something went wrong: number of evaluations does not coincide with budget :(")
