import os
import os.path as op
import sys
import json
import pandas as pd
import numpy as np
from sklearn import * 

from numerdd import parse, model

if len(sys.argv) > 2:
    idata = parse.getCSV()
else:
    idata = parse.Parser()

idata.getData()
modd = model.Modelerz(idata)

# Make a model with the data.
mdl = linear_model.LogisticRegression(tol=1e-6, n_jobs=-1)
ansnow = modd.eMod(mdl, "Logistic Regression")

#Write out or not.
g = int(input("Does the model look good enough to write out? (1 for yes 0 for no)"))
if (g):
    ansnow.to_csv(idata.outfile, index=False) 

