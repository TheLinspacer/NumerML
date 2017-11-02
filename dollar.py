import os
import os.path as op
import sys
import numpy as np
from sklearn import linear_model

from numerdd import parse, model

#def reduceModel()

if __name__ == "__main__":

    if len(sys.argv) > 2:
        idata = parse.getCSV()
    else:
        idata = parse.Parser()
    
    idata.getData()
    modd = model.Modelerz(idata)
    
    # Make a model with the data.
    mdl = linear_model.LogisticRegression(tol=1e-6, n_jobs=-1)
    a1 = modd.eMod(mdl, "Logistic Regression")
    
    # Transformer and then transformed validation and training data
    fd = {"mag": np.linalg.norm, "mx": np.max, "avg": np.mean, 'fft': np.max(np.fft.fft)} 
    modd.pj.trainF = model.summit(idata.trainF, fd)
    modd.pj.validF = model.summit(idata.validF, fd)
    modd.pj.predF = model.summit(idata.predF, fd)
    a2 = modd.eMod(mdl, "Reduced Logistic Regression")
    
    #Write out or not.
    g = int(input("Does the model look good enough to write out? (1 for yes 0 for no)  "))
    if (g):
        a2.to_csv(idata.outfile, index=False) 

