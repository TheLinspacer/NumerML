"""
    Parse the .
"""

import os
import os.path as op
import sys
import json
import pandas as pd
import numpy as np
import sklearn as sk
from datetime import datetime
from numerapi.numerapi import NumerAPI

# Use this source path as the anchor for the paths.
def homepath():
    return op.abspath(op.dirname(op.dirname(__file__)))

def dirmake(pth):
    if op.isdir(pth):
        os.mkdir(pth)
    return

class Parser(object):
    def __init__(self, hmpath=homepath()):

        self.thispath = hmpath
        self.datapath = op.join(self.thispath, "data")
        self.rsltpath = op.join(self.thispath, "rslts")
        dirmake(self.datapath)
        dirmake(self.rsltpath)

        self.nrslt = len(os.listdir(self.rsltpath))-1
        self.outfile = op.join(self.rsltpath, "prediction" + str(self.nrslt) + ".csv")
        self.outvar = "probability"
        self.dummyfile = op.join(self.rsltpath, "nowTemp.csv")
        self.traindata = op.join(self.datapath, "numerai_training_data.csv")
        self.predata = op.join(self.datapath, "numerai_tournament_data.csv")

# 
    def getData(self):
        dataT = pd.read_csv(self.traindata, header=0)
        dataP = pd.read_csv(self.predata, header=0)
        self.feats = [f for f in dataT.columns.values.tolist() if "feature" in f]
        self.typ = ["id"]
        self.targ = ["target"]
        self.trainF = dataT[self.feats]
        self.trainT = dataT[self.targ]
        self.trainT = self.trainT[self.targ].astype(np.int32)
        self.dataV = dataP[dataP["data_type"].isin(["validation"])]
        self.dataV = self.dataV.set_index(self.typ)
        self.validF = self.dataV[self.feats]
        self.validT = self.dataV[self.targ]
        dataP = dataP.set_index(self.typ)
        self.predF = dataP[self.feats]        

    def compareValid(self, predz):
        self.vp = np.sum(np.abs(self.validT.values-predz.values))/len(self.validT)
        print("{:.2f} percent of the predictions are False.".format(self.vp*100.0)) 

    def writeOut(self, predz):
        self.idout[self.outvar] = predz
        self.idout.reset_index()
        self.idout.to_csv(self.outfile, index=False)

# Initially, to set up the work call this function and it will download
# and unzip the datasets and set up the result and data folders.
def getCSV(hmpath=homepath()):
    napi = NumerAPI(verbosity="info")
    ps = Parser(hmpath)
    napi.download_current_dataset(dest_path=ps.datapath, unzip=True)
    return ps


#See how different two files are.
def twoFiles():
    pass