# -*- coding: utf-8 -*-
"""
Created on Fri Jan 5 14:04:59 2017

@author: Keith 
"""

import math
import SavitzkyGolay as sv
import numpy as np

class CostSource:
    def __init__(self, v=0, o=0, c=0, cr=0, cc=365, co=0, e=float("inf")):
        self.Value = v
        self.Offset = o
        self.Cadence = c
        self.ChangeRate = cr
        self.ChangeCadence = cc
        self.ChangeOffset = co
        self.Exp = e;

    def calcStream(self, years):
        DAYS_PER_YR = 365
        tmax = years * DAYS_PER_YR
        pStream = np.zeros(tmax)
        time = range(1,tmax)

        for t in time:
            if t <= self.Exp:
                pStream[t] = self.Value * (not (np.mod(t - self.Offset, self.Cadence)))
                self.Value = self.Value*(1+float((not (False or np.mod(t,self.ChangeCadence)))*self.ChangeRate))
        return np.array(pStream)

    def sumSources(self,s1,s2,s3,s4,s5,s6,s7):
        self.Value += (s1.Value + s2.Value + s3.Value + s4.Value +
                       s5.Value + s6.Value + s7.Value)

    # def sumSourcesMonthly(self,s1,s2,s3,s4,s5,s6,s7):
    #     for i in nargin
    #         tmp = kwargs[]
    #         if tmp.Cadence != fh.MONTHLY:
    #             tmpVal = tmp.Value

    def smoothStream(self, n, window, order):
        tmp = self.calcStream(n)
        sv.runFilter(tmp, window, order)


class IncomeSource:
    def __init__(self, v=0, o=0, c=0, cr=0, cc=365, co=0, e=float("inf")):
        self.Value = v
        self.Offset = o
        self.Cadence = c
        self.ChangeRate = cr
        self.ChangeCadence = cc
        self.ChangeOffset = co
        self.Exp = e;

    def calcStream(self, years):
        DAYS_PER_YR = 365
        tmax = years * DAYS_PER_YR
        pStream = np.zeros(tmax)
        time = range(1,tmax)

        for t in time:
            if t <= self.Exp:
                pStream[t] = self.Value * (not (np.mod(t - self.Offset, self.Cadence)))
                self.Value = self.Value*(1+float((not (False or np.mod(t,self.ChangeCadence)))*self.ChangeRate))
        return pStream


