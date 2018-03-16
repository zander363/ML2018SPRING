#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.

"""
this python script is to visualize 
the relation between PM2.5 and each feature
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv',encoding='ISO-8859-1')
PM25 = data.iloc[9:500:18,3:14]
feature1 = data.iloc[1:500:18,3:14]
feature2 = data.iloc[2:500:18,3:14]
feature3 = data.iloc[3:500:18,3:14]
feature4 = data.iloc[4:500:18,3:14]
feature5 = data.iloc[5:500:18,3:14]
feature6 = data.iloc[6:500:18,3:14]
feature7 = data.iloc[7:500:18,3:14]
feature8 = data.iloc[8:500:18,3:14]
feature10 = data.iloc[10:500:18,3:14]
feature11 = data.iloc[11:500:18,3:14]
feature12 = data.iloc[12:500:18,3:14]
feature13 = data.iloc[13:500:18,3:14]
feature14 = data.iloc[14:500:18,3:14]
feature15 = data.iloc[15:500:18,3:14]
feature16 = data.iloc[16:500:18,3:14]
feature17 = data.iloc[17:500:18,3:14]
feature18 = data.iloc[18:500:18,3:14]

plt.plot(feature10,PM25[:],'o')
#plt.scatter(PM25,feature1)
plt.show()

