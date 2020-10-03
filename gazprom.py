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

# для карты
import io
import urllib.request
import random
import mercantile
 
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


# отображение карты мспользуя OpenStreetMaps
def plugmap(w,s,e,n):
  west = open_df.long.min()
  south =open_df.lat.min()
  east = open_df.long.max()
  north = open_df.lat.max()
  zoom = 5 # 

  tiles = list(mercantile.tiles(west, south, east, north, zoom))
  print(tiles)

  tile_size = (256, 256)
  # создаем пустое изображение в которое как мозайку будем вставлять тайлы
  # для начала просто попробуем отобразить все четыре тайла в строчку
  map_image = ImageSurface(FORMAT_ARGB32, tile_size[0] * len(tiles), tile_size[1])

  # создаем контекст для рисования
  ctx = Context(map_image)

  for idx, t in enumerate(tiles):
    server = random.choice(['a', 'b', 'c'])  # у OSM три сервера, распределяем нагрузку
    url = 'http://{server}.tile.openstreetmap.org/{zoom}/{x}/{y}.png'.format(
        server=server,
        zoom=t.z,
        x=t.x,
        y=t.y
    )
    # запрашиваем изображение
    response = urllib.request.urlopen(url)

    # создаем cairo изображние 
    img = ImageSurface.create_from_png(io.BytesIO(response.read()))

    # рисуем изображение, с правильным сдвигом по оси x
    ctx.set_source_surface(img, idx * tile_size[0], 0)
    ctx.paint()

  # сохраняем собраное изображение в файл
  with open("map.png", "wb") as f:
    map_image.write_to_png(f)

if __name__ == "__main__":
    image_bank(file_gazprom_csv,file_opendata_csv)


