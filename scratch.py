# scratch.py

import numpy as np
import math
import FinanceHeader as fh
from Sources import IncomeSource 
import matplotlib.pyplot as plt

years = 5
tmax = years * fh.DAYS_PER_YR
multiplier = np.ones(tmax)
paycheck = IncomeSource(1800,12,14,fh.RAISERATE,fh.ANNUAL,fh.SEMI, fh.NOEXP)

foo = int(math.floor((years*fh.DAYS_PER_YR)/paycheck.ChangeCadence))
for f in range(1,foo):
	multiplier[f*paycheck.ChangeCadence:] += paycheck.ChangeRate  

print("Done")