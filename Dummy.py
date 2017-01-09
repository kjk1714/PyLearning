
import numpy as np
import math
import FinanceHeader as fh
from Sources import CostSource 

years = 5
tmax = years * fh.DAYS_PER_YR
pStream1 = np.zeros(tmax)
pStream2 = np.zeros(tmax)
time = range(0,tmax)

foo = CostSource(500,1,30,0,fh.ANNUAL,0,fh.SEMI)
cash = CostSource(400,31,fh.MONTHLY,fh.INFRATE,fh.ANNUAL,1,0)

for t in time:
   pStream1[t] = foo.Value * (not (np.mod(t - foo.Offset, foo.Cadence)))
   pStream2[t] = cash.Value * (not (np.mod(t - cash.Offset, cash.Cadence)))

   foo.Value = foo.Value * (1+float((not (False or np.mod(t,foo.ChangeCadence)))*foo.ChangeRate))
   cash.Value = cash.Value * (1+float((not (False or np.mod(t,cash.ChangeCadence)))*cash.ChangeRate))

print("Done")