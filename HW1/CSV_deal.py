#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.

"""
this file is use to test how to 
use pandsa to deal with CSV file
"""
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv',encoding='ISO-8859-1')
print(data.head())
print(data.columns)
print(data.index)
print(data.iloc[9::18,3:14])#here is using iloc to index the PM2.5 items

#print(data["PM2.5"])
#data.plot()
#print(data)

#data.to_csv("output") #save as a CSV file
