#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Программа расчета эффективности расположения банкоматов в городе
Используются данные на сайте банка Газпром в открытом доступе о расположении
банкоматов в городе, в частности г. Москва
Второй набор данных - открытые данные расположения торговых центров этом городе
Для расчета используется кластерный анализ.
Результаты выводятся в графическом виде

Метод запуска, для простоты через терминал с передаче параметров в командной строке
Пример:

    python3 gazprom.py Moskow.csv torgovl_stat.csv

Требования: версия Python не ниже 3.6, файлы данных ы формате CSV

Файлы программы находятся на "https://github.com/Amur79/banks.git"
Для использования достаточно все скопировать в одну папку и распаковать открытые данные 
"""
# для карты банкоматов
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from collections import Counter

from sklearn.cluster import KMeans

# Входные данные
# данные расположения банкоматов Газпром в формате CSV
file_gazprom_csv = sys.argv[1] #"Moskow.csv"
# открытые данные расположения торговых центров ы формате CSV
file_opendata_csv = sys.argv[2] # "torgovl_stat.csv"

def data_process():
  """
  Основной модуль расчета эффективности расположения банкоматов
  """
  dat = pd.read_csv(file_gazprom_csv, sep='\t')
  dat = dat[dat["Область"] == "Москва"]
  dat["lat"] = dat["Широта"]
  dat["long"] = dat["\Долгота"]
  dat = dat.drop(["\Долгота", "Широта"], axis=1)

  open_data = pd.read_csv(file_opendata_csv)
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

  open_df = pd.DataFrame({'types': types,
                          'lat': lats,
                          'long': longs})

  open_df = open_df.astype({'lat': 'float64', 'long': 'float64'})
  print(open_df.head())

  x = np.array([open_df["lat"], open_df["long"]])
  x = np.transpose(x)
  
  print("Вычисление...")
  # кластерный анализ и получение результатов
  bankomats, labels = find_clusters(x, 100)
  
  # отображение результатов
  fig, ax = plt.subplots()
  
  ax.scatter(open_df.lat, open_df.long, c='green', s=5, label = "ТЦ") # расположение торговых центров
  ax.scatter(bankomats[:, 0], bankomats[:, 1], c='cyan', s=10, label = "Новые банкоматы") # вычисленное расположение банкоматов
  ax.scatter(dat.long, dat.lat, c='red', s=10, label = "Текущение банкоматы") # текущее расположение банкоматов

  ax.legend() # включаем легенду
  
  fig.set_figheight(10) # размер по высоте
  fig.set_figwidth(10) # размер по ширине
  plt.show() # отображение

def find_clusters(X, 
                  n_clusters, # количество кластеров
                  rseed=3, 
                  max_iters=50, # максимальное количество итераций
                  weight_koef=0.000002): # весовые коэфициенты
  """
  Модуль кластерного анализа
  """
  rng = np.random.RandomState(rseed)
  i = rng.permutation(X.shape[0])[:n_clusters]
  centers = X[i]
  # print(X[i])

  for iter in range(max_iters):
    # print(centers)
    labels = pairwise_distances_argmin(X, centers, metric='manhattan')
    # weights = pairwise_distances(X, centers, metric='manhattan')
    elems_count = Counter(labels)
    lengths = []
    for x_iter in range(X.shape[0]):
        weights = []
        for center_id in range(len(centers)):
            weight = abs(X[x_iter, 0] - centers[center_id][0]) + abs(X[x_iter, 1] - centers[center_id][1])
            # Поправка на очереди у банкомата
            weight += weight_koef * elems_count[center_id]
            weights.append(weight)
        lengths.append(weights)
    labels_res = []
    for x in lengths:
        labels_res.append(np.argmin(x))
    labels = np.array(labels_res)

    new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
    length = len(new_centers[np.isnan(new_centers)]) // 2
    # lat_rand = np.array([X[:, 0].min()]*length) + (X[:, 0].max() - X[:, 0].min()) * np.random.random(length)
    # long_rand = np.array([X[:, 1].min()]*length) + (X[:, 1].max() - X[:, 1].min()) * np.random.random(length)
    # arr = np.transpose(np.array([lat_rand, long_rand]))
    i = rng.permutation(X.shape[0])[:length]
    new_centers[np.isnan(new_centers[:, 0])] = X[i]

    if np.all(centers == new_centers):
      break

    centers = new_centers
    print(iter)
    
  return centers, labels


# if __name__ == "__main__":
#     image_bank(file_gazprom_csv, file_opendata_csv)

# Запуск основного модуля
data_process()

