# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 20:12:43 2018

@author: Ankush
"""

import pandas as pd
dataset = pd.read_csv("data.csv")
x = dataset.iloc[0,1:4]