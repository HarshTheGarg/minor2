import numpy as np
import cv2
import pyautogui

def calc_bounding_box(image, landmarks):
  imw, imh = image.shape[1], image.shape[0]
  landar = np.empty((0, 2), int)
  for _, land in enumerate(landmarks.landmark):
    landx = min(int(land.x * imw), imw-1)
    landy = min(int(land.y * imh), imh-1)
    
    lanpoint = [np.array((landx, landy))]
    landar = np.append(landar, lanpoint, axis=0)
  
  x, y, w, h = cv2.boundingRect(landar)
  return [x, y, x+w, y+h]

def calc_land_image(image, landmarks):
  imw, imh = image.shape[1], image.shape[0]
  landar = []

  for land in landmarks.landmark:
    landx = min(int(land.x * imw), imw-1)
    landy = min(int(land.y * imh), imh-1)

    landar.append([landx, landy])
    
  
  return landar

def calc_land_screen(point):
  scw, sch = pyautogui.size()
  # print(scw, sch)
  x =min(int(point.x * scw), scw)
  y = min(int(point.y * sch), sch)
  return [x, y]

