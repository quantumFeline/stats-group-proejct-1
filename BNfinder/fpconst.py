import math

NaN = float('nan')
PosInf = float('inf')
NegInf = float('-inf')

def isNaN(v):
    return v != v

def isPosInf(v):
    return v == PosInf

def isNegInf(v):
    return v == NegInf

def isFinite(v):
    return not isNaN(v) and not isPosInf(v) and not isNegInf(v)