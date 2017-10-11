import os
import os.path as op
import sys

import pandas as pd
import numpy as np

# from numerapi.numerapi import NumerAPI

def compareValid(va, vb):
    vp = np.sum(np.abs(va-vb))
    vp = vp.astype(float)
    voi = vp/len(vb)
    print("{:.2f} percent of the predictions are False.".format(voi*100.0))

class Modelerz(object):
    def __init__(self, prse):
        self.pj = prse

    def eMod(self, modL, descr):
        print(descr)
        modL.fit(self.pj.trainF, self.pj.trainT)
        self.pre = modL.predict(self.pj.validF)
        compareValid(self.pj.validT.values.astype(int).ravel(), self.pre) # Prints validation test results.
        yprob = modL.predict_proba(self.pj.predF)
        jm = pd.DataFrame({self.pj.outvar: yprob[:,1]}, index=self.pj.predF.index)
        return jm.reset_index()
