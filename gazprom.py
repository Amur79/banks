#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
# для карты банкоматов
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.cbook as cbook 

from sklearn.cluster import KMeans

import typing

import requests

class MapParams(object):
    def __init__(self):
        self.lat = 55.665279  # Координаты центра карты на старте. Задал координаты университета
        self.lon = 35.813492
        self.zoom = 5  # Масштаб карты на старте. Изменяется от 1 до 19
        self.type = "map" # Другие значения "sat", "sat,skl"
 
    # Преобразование координат в параметр ll, требуется без пробелов, через запятую и без скобок
    def ll(self):
        return str(self.lon)+","+str(self.lat)
 
    def update(self, event):
        my_step = 0.008             
        if event.key == 280 and self.zoom < 19:  # Page_UP
            self.zoom += 1
        elif event.key == 281 and self.zoom > 2:  # Page_DOWN
            self.zoom -= 1
        elif event.key == 276:  # LEFT_ARROW
            self.lon -= my_step * math.pow(2, 15 - self.zoom)
        elif event.key == 275:  # RIGHT_ARROW
            self.lon += my_step * math.pow(2, 15 - self.zoom)
        elif event.key == 273 and self.lat < 85:  # UP_ARROW
            self.lat += my_step * math.pow(2, 15 - self.zoom)
        elif event.key == 274 and self.lat > -85:  # DOWN_ARROW
            self.lat -= my_step * math.pow(2, 15 - self.zoom)
            
            
file_gazprom_csv = "/home/maxim/py/bankomats/Moskow.csv"

file_opendata_csv = "/home/maxim/py/bankomats/torgovl_stat.csv"

# Создание карты с соответствующими параметрами.
def load_map(mp):
    map_request = "http://static-maps.yandex.ru/1.x/?ll={ll}&z={z}&l={type}".format(ll=mp.ll(), z=mp.zoom, type=mp.type)
    response = requests.get(map_request)
    if not response:
        print("Ошибка выполнения запроса:")
        print(map_request)
        print("Http статус:", response.status_code, "(", response.reason, ")")
        sys.exit(1)
 
    # Запись полученного изображения в файл.
    map_file = "map.png"
    try:
        with open(map_file, "wb") as file:
            file.write(response.content)
    except IOError as ex:
        print("Ошибка записи временного файла:", ex)
        sys.exit(2)
    return map_file
         
def image_bank(file_gazprom,file_opendata):
   dat = pd.read_csv(file_gazprom, sep='\t')

   dat.head()

#   print(dat["Широта"].min())
#   print(dat["Широта"].max())
#   print(dat["\Долгота"].min())
#   print(dat["\Долгота"].max())
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

#   print(open_df.lat.min())
#   print(open_df.lat.max())
#   print(open_df.long.min())
#   print(open_df.long.max())


   open_df["lat_int"] = np.round(open_df["lat"],2)
   open_df["long_int"] = np.round(open_df["long"],2)


   open_df.head()

# обучение усреднением
   kmeans = KMeans(n_clusters=253)
   kmeans.fit(open_df[["lat","long"]])

   y_means = kmeans.predict(open_df[["lat","long"]])

   bankomats = kmeans.cluster_centers_
   
   #open_df_wm = open_df.to_crs({'init' :'epsg:3857'})
   #непосредственно преобразование проекции
   
   # звгружаем файл Яндекса
   mp = MapParams()
   file_map = os.getcwd() + "/" + load_map(mp)
  
   with cbook.get_sample_data(file_map) as image_file:
    image = plt.imread(image_file) 
   
   fig, ax = plt.subplots()

# Если убрать комментарий то появится карта
#   ax.imshow(image)
         
   ax.scatter(open_df.lat, open_df.long, c='green', s=5)
   ax.scatter(bankomats[:, 0], bankomats[:, 1], c='cyan', s=10)
   ax.scatter(dat.long, dat.lat, c='red', s=10)
   
   fig.set_figwidth(10)     #  ширина и
   fig.set_figheight(10)    #  высота "Figure"

#   ax.set_xlim(30, 40)
#   ax.set_ylim(50, 60)
   ax.set_title('Расположение банкоматов')
   
   plt.show()
   
if __name__ == "__main__":
    image_bank(file_gazprom_csv,file_opendata_csv)


