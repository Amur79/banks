#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
# для карты банкоматов
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans

from cairo import ImageSurface, FORMAT_ARGB32, Context


file_gazprom_csv = "/home/maxim/py/bankomats/Moskow.csv"

file_opendata_csv = "/home/maxim/py/bankomats/torgovl_stat.csv"

def image_bank(file_gazprom,file_opendata):
   dat = pd.read_csv(file_gazprom, sep='\t')

   dat.head()

   print(dat["Широта"].min())
   print(dat["Широта"].max())
   print(dat["\Долгота"].min())
   print(dat["\Долгота"].max())
   dat.info()
   dat = dat[dat["Область"] == "Москва"]
   dat["lat"] = dat["Широта"]
   dat["long"] = dat["\Долгота"]
   dat = dat.drop(["\Долгота", "Широта"], axis=1)

   dat.head()

   open_data = pd.read_csv(file_opendata)
   types = []
   lats = []
   longs = []

   for line in open_data.geoData:
      typ = line.split(", ")[0].split('=')[1]
      lat = line.split(", ")[1].split('[')[1]
      long = line.split(", ")[2].split(']')[0]
      types.append(typ)
      lats.append(lat)
      longs.append(long)

   open_df = pd.DataFrame({'types':types,
                       'lat':lats,
                       'long':longs})

   open_df = open_df.astype({'lat': 'float64', 'long': 'float64'})

   print(open_df.lat.min())
   print(open_df.lat.max())
   print(open_df.long.min())
   print(open_df.long.max())


   open_df["lat_int"] = np.round(open_df["lat"],2)
   open_df["long_int"] = np.round(open_df["long"],2)


   open_df.head()

# обучение усреднением
   kmeans = KMeans(n_clusters=253)
   kmeans.fit(open_df[["lat","long"]])

   y_means = kmeans.predict(open_df[["lat","long"]])

   bankomats = kmeans.cluster_centers_

   plt.scatter(open_df.lat, open_df.long, c='green', s=5)
   plt.scatter(bankomats[:, 0], bankomats[:, 1], c='cyan', s=10)
   plt.scatter(dat.long, dat.lat, c='red', s=10)


if __name__ == "__main__":
    image_bank(file_gazprom_csv,file_opendata_csv)


