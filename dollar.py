import os
import os.path as op
import sys
import json
import pandas as pd
import numpy as np
import sklearn as sk

from numerdd import *

if len(sys.argv) > 2:
    idata = parse.getCSV()
else:
    idata = parse.Parser()

idata.getData()

model.#Do shit to make model
