#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import pygame, requests, sys, os, math
 
# Создаем оконное приложение, отображающее карту по к и в масштабеоординатам, который задаётся программно.
 
class MapParams(object):
    def __init__(self):
        self.lat = 55.665279  # Координаты центра карты на старте. 
        self.lon = 37.813492
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
         
def main():
    # Инициализируем pygame
    pygame.init()
    screen = pygame.display.set_mode((600, 450))
    mp = MapParams()
        
    running = True
    while running:
       for event in pygame.event.get():
          if event.type == pygame.KEYUP:  # Обрабатываем различные нажатые клавиши.
             mp.update(event)
          elif event.type == pygame.QUIT:  # Выход из программы
             running = False
          #Создаем файл
          map_file = load_map(mp)
          # Рисуем картинку, загружаемую из только что созданного файла.
          screen.blit(pygame.image.load(map_file), (0, 0))
          pygame.display.flip()
           
    pygame.quit()
    
    # Удаляем файл с изображением.
    os.remove(map_file) 
   
if __name__ == "__main__":
    main()