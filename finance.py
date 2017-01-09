# -*- coding: utf-8 -*-
"""
Created on Fri Jan 5 14:04:59 2017

@author: Keith
"""

import Sources
reload(Sources)
from Sources import * 
from operator import add
import math
import scipy
import SavitzkyGolay as sv
reload(sv)
import matplotlib.pyplot as plt
import numpy as np
import FinanceHeader as fh
reload(FinanceHeader)

# Constants
NUMYRS = 3
BANK_INIT = 3800

# Preamble
plt.close("all")

# Build Income Objects
paycheck = IncomeSource(1800,12,14,fh.RAISERATE,fh.ANNUAL,fh.SEMI, fh.NOEXP)
rent = IncomeSource(400,1,30,fh.INFRATE,fh.ANNUAL,0,fh.ANNUAL)
taxes = IncomeSource(1500,60,365,0.1,fh.ANNUAL,59, fh.NOEXP)

# Build Cost Objects
sky = CostSource(1000,17,fh.MONTHLY,fh.INFRATE,fh.ANNUAL,1,fh.NOEXP)
discover = CostSource(300,1,fh.MONTHLY,fh.INFRATE,fh.ANNUAL,1,fh.NOEXP)
mortgage = CostSource(1100,3,fh.MONTHLY,0,fh.ANNUAL,1,fh.NOEXP)
rge = CostSource(200,28,fh.MONTHLY,0,fh.ANNUAL,0,fh.NOEXP)
car = CostSource(415,15,fh.MONTHLY,0,fh.ANNUAL,0,2*fh.ANNUAL)
cash = CostSource(100,31,fh.WEEKLY,fh.INFRATE,fh.ANNUAL,1,fh.NOEXP)
lifeins = CostSource(100,5,fh.MONTHLY,0,fh.ANNUAL,1,fh.NOEXP)

Garbage = CostSource(99,62,fh.QUARTERLY,0,fh.ANNUAL,0,fh.NOEXP)
Water = CostSource(55,53,fh.QUARTERLY,0,fh.ANNUAL,0,fh.NOEXP)
Cable = CostSource(10,26,fh.MONTHLY,0,fh.ANNUAL,0,fh.NOEXP)
Data = CostSource(75,31,fh.MONTHLY,0,fh.ANNUAL,0,fh.NOEXP)
Phone = CostSource(79,31,fh.MONTHLY,0,fh.ANNUAL,0,fh.NOEXP)
CarIns = CostSource(180,23,fh.MONTHLY,0,fh.ANNUAL,0,fh.NOEXP)
Golf = CostSource(280,15,fh.MONTHLY,fh.INFRATE,fh.ANNUAL,1,fh.NOEXP)

# Simulate Bank Account
allIncome = ( paycheck.calcStream(NUMYRS) +
              rent.calcStream(NUMYRS) +
              taxes.calcStream(NUMYRS) )

# Consolidate bills on credit card and reconcile cadence
citi = CostSource();
citi.Offset = 15

citiStream = ( Garbage.calcStream(NUMYRS) +
               Water.calcStream(NUMYRS) +
               Cable.calcStream(NUMYRS) +
               Data.calcStream(NUMYRS) +
               Phone.calcStream(NUMYRS) +
               CarIns.calcStream(NUMYRS) +
               Golf.calcStream(NUMYRS) )

citiMonthly = np.zeros(len(citiStream))
acc = 0
for i in range(1,len(citiStream)):
   acc += citiStream[i]
   if not(np.mod(i - citi.Offset,fh.MONTHLY)):
      citiMonthly[i] = acc
      acc = 0

allCosts = ( sky.calcStream(NUMYRS) + 
             discover.calcStream(NUMYRS) +
             mortgage.calcStream(NUMYRS) +
             car.calcStream(NUMYRS) +
             lifeins.calcStream(NUMYRS) +
             rge.calcStream(NUMYRS) +
             cash.calcStream(NUMYRS) +
             citiMonthly )

bankStream = np.zeros(allIncome.shape[0])
ax = np.zeros(allIncome.shape[0])
bankStream[0] = BANK_INIT

for t in range(0,NUMYRS*fh.ANNUAL-1):
   ax[t] = t
   bankStream[t+1] = bankStream[t] + allIncome[t] - allCosts[t]

plt.figure(1)
plt.plot(bankStream)
plt.ylabel('Balance')
plt.xlabel('Time')

plt.show()

