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
    
1. Moskow.csv - данные по терминалам банка города
2. torgovl_stat.csv - открытые данные по торговым центрам этого же города

Требования: версия Python не ниже 3.6, файлы данных ы формате CSV

Файлы программы находятся на "https://github.com/Amur79/banks.git"
Для использования достаточно все скопировать в одну папку и распаковать открытые данные 

На будущее:
есть возможность использовать данное приложение и для других городов,
с соответствующими входными данными и срезом города
"""
# для карты банкоматов
import json
import sys, requests, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook 
import matplotlib.ticker as ticker

from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from collections import Counter
from sklearn.cluster import KMeans

# Входные данные
# данные расположения банкоматов Газпром в формате CSV
file_gazprom_csv = sys.argv[1] #"Moskow.csv"
# открытые данные расположения торговых центров ы формате CSV
file_opendata_csv = sys.argv[2] # "torgovl_stat.csv"

param_city = "Москва"


# отображение yandex карты 
class MapParams(object):
    def __init__(self, ln, lt):
        self.lat = lt # Координаты центра карты на старте. 
        self.lon = ln
        self.zoom = 5  # Масштаб карты на старте. Изменяется от 1 до 19
        self.type = "map" # Другие значения "sat", "sat,skl"
    # Преобразование координат в параметр ll, требуется без пробелов, через запятую и без скобок
    def ll(self):
        return str(self.lon)+","+str(self.lat)
    
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
        
def data_process():
    """
    Основной модуль расчета эффективности расположения банкоматов
    """
    dat = pd.read_csv(file_gazprom_csv, sep='\t')
    dat = dat[dat["Область"] == param_city]
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
  
    bankomats_current = np.array([dat["long"], dat["lat"]])
    bankomats_current = np.transpose(bankomats_current)
    current_weight = count_weight(x, bankomats_current)
    
    print("Текущее условное время обслуживания клиента {}".format(current_weight))
    print("Вычисление...")
  
    # кластерный анализ и получение результатов
    bankomats, labels = find_clusters(x, 100)

    new_weight = count_weight(x, bankomats)
    print("Рассчитанное условное время обслуживания клиента при введении системы: {}".format(new_weight))

    # подготовка координат для получения yandex карт, 
    # максимальные и минимальные коррдитанты открытых данных
    lt = (open_df.lat.max() - open_df.lat.min()) / 2 + open_df.lat.min()
    ln = (open_df.long.max() - open_df.long.min()) / 2 + open_df.long.min()
  
    mp = MapParams(lt, ln)
    # звгружаем файл Яндекса                 
    file_map = os.getcwd() + "/" + load_map(mp)
  
    with cbook.get_sample_data(file_map) as image_file:
        image = plt.imread(image_file) 
  
    # отображение результатов
    fig, ax = plt.subplots()
  
    ax.scatter(open_df.lat, open_df.long, c='green', s=5, label = "ТЦ") # расположение торговых центров
    ax.scatter(bankomats[:, 0], bankomats[:, 1], c='cyan', s=10, label = "Новые банкоматы") # вычисленное расположение банкоматов
    ax.scatter(dat.long, dat.lat, c='red', s=10, label = "Текущение банкоматы") # текущее расположение банкоматов

    ax.legend() # включаем легенду
    ax.set_xlabel('Широта')
    ax.set_ylabel('Долгота')  
    ax.set_title('Банкоматы Газпробанка')

    fig2, ax2 = plt.subplots()
    ax2.imshow(image) # отображаем карту
  
    fig.set_figheight(10) # размер по высоте
    fig.set_figwidth(10) # размер по ширине
    plt.show() # отображение
    fig.savefig('maps.png') # сохранение картинки
    fig2.savefig('mapcity.png') # сохранение картинки
  
def find_clusters(X,
                  n_clusters,  # количество кластеров
                  rseed=3,
                  max_iters=50,  # максимальное количество итераций
                  weight_koef=0.000002):  # весовые коэфициенты
    """
    Модуль кластерного анализа
    """
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    # print(X[i])

    for iter in range(max_iters):
        # Получаем номера ближайших банкоматов для всех точек в списке данных (ТЦ, адреса клиентов, стихийные рынки)
        # Для расчёта используем манхэттэнскую метрику, чтобы учесть движение клиентов по кварталам и улицам
        labels = pairwise_distances_argmin(X, centers, metric='manhattan')
        # Считаем, как много клиентов-точек приходится на каждый банкомат
        elems_count = Counter(labels)
        elems_count_list = []
        for i in range(n_clusters):
            if i in elems_count:
                elems_count_list.append(elems_count[i])
            else:
                elems_count_list.append(0)
        elems_count_list = np.array(elems_count_list)
        lengths = []
        weight_new = np.zeros([X.shape[0], centers.shape[0]])
        it = 0
        for x in X:
            # Считаем расстояние до ближайшего банкомата
            weight = np.abs(x[0] - centers[:, 0]) + np.abs(x[1] - centers[:, 1])
            # Но на этот раз прибавляем поправку, которая зависит от количества точек-клиентов на каждом банкомате
            weight += weight_koef * elems_count_list
            weight_new[it] = weight
            it += 1
        # Пересчитываем ближайшие банкоматы с учётом поправок
        labels = np.argmin(weight_new, axis=1)

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
        print(iter)  # вывод прогресса вычислений
    
    # вывод списка координат новых банкоматов
    print("Координаты новых банкоматов")
    k=1
    for i in centers:
        print(k,i)
        k=k+1
        
   
    return centers, labels # return def find_clusters

def count_weight(X, centers, weight_koef=0.000002):
    labels = pairwise_distances_argmin(X, centers, metric='manhattan')
    # Считаем, как много клиентов-точек приходится на каждый банкомат
    elems_count = Counter(labels)
    elems_count_list = []
    for i in range(centers.shape[0]):
        if i in elems_count:
            elems_count_list.append(elems_count[i])
        else:
            elems_count_list.append(0)
    elems_count_list = np.array(elems_count_list)
    lengths = []
    weight_new = np.zeros([X.shape[0], centers.shape[0]])
    it = 0
    for x in X:
        # Считаем расстояние до ближайшего банкомата
        weight = np.abs(x[0] - centers[:, 0]) + np.abs(x[1] - centers[:, 1])
        # Но на этот раз прибавляем поправку, которая зависит от количества точек-клиентов на каждом банкомате
        # Мат.ожидание при распределении Пуассона lambda = n*p, где p - вероятность, что потребуется банкомат
        weight += weight_koef * elems_count_list
        weight_new[it] = weight
        it += 1
    return(np.sum(np.min(weight_new, axis=1)))

# Запуск основного модуля
if __name__ == "__main__":
    data_process()
   
