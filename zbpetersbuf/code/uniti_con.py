"""
uniti_con.py
This moduale contains the relevent code for the unix time convertion
"""
import re
import os
import numpy as np

def gt_unix(mod_name):
    """gives the unix time for the start of the data collection"""
    fpath='zbpetersbuf/data/'.strip()
    filpath = os.path.join(fpath, mod_name)

    with open(filpath, 'r', encoding='utf-8') as f:
        ftime = np.delete(re.findall(r'\d+', f.read()),0)

    matime = np.zeros(ftime.shape[0])
    for i in range(ftime.shape[0]):
        matime[i] = int(ftime[i])

    awf = [0,31,59,90,120,151,181,212,243,273,304,334]
    dfg = awf[int(matime[1])-1]
    day = 365*(matime[0] - 1970) + matime[2] + dfg + ((matime[0]-1972)//4)
    hour = matime[3] + matime[7]
    minit = 60*hour + matime[4]
    sec = 86400*day + 60*minit + matime[5] + matime[6]*(10**(-3))

    return sec
