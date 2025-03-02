#!./.venv/bin/python
import cv2
import mediapipe as mp
import csv
from collections import deque

# To resolve problem with OS using pyautogui
try:
  import subprocess
  subprocess.run(["/usr/bin/xhost", "+"])
except:
  pass

import pyautogui

from drawingUtil import drawBrect, drawInfo, drawLandMarks, drawZCircle

from preprocessingUtil import normalizeToBase, preprocessZ, normalizeXY

from calcUtil import calc_land_image, calc_bounding_box, calc_land_screen

from models.knn import knnPredict


def main():
  handDetector = mp.solutions.hands.Hands(
    max_num_hands=2
  )
  drawingUtil = mp.solutions.drawing_utils

  cap = cv2.VideoCapture(0)

  mode = 0
  zqueue, xqueue, yqueue = deque(maxlen=5),deque(maxlen=5),deque(maxlen=5)

  with open("./models/classes.csv", "r") as file:
    csvreader = csv.reader(file)
    labels = [row[0] for row in csvreader]
  

  while True:
    key = cv2.waitKey(10)

    # ESC
    if key == 27:
      break

    number, mode = setModeNumber(key, mode)

    ret, frame = cap.read()
    if not ret:
      break


    frame = cv2.flip(frame, 1)
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res = handDetector.process(rgbFrame)
    hands = res.multi_hand_landmarks

    if hands:
      for hand_lands, handedness in zip(hands, res.multi_handedness):
        
        boundRect = calc_bounding_box(frame, hand_lands)
        landmarkList = calc_land_image(frame, hand_lands)


        newZ, zqueue = preprocessZ(hand_lands.landmark[8].z, zqueue)
        # print(newZ)

        pointerLoc = calc_land_screen(hand_lands.landmark[8])
        pointerLoc, xqueue, yqueue = normalizeXY(pointerLoc, xqueue, yqueue)
        # print(pointerLoc)

        preprocessedLands = normalizeToBase(landmarkList)

        gesture = ""
        if mode == 0:
          pass
          # pyautogui.moveTo(*pointerLoc)
          # gestInd = knnPredict(preprocessedLands, "./models/dataset.csv")
          # gestInd = 0
          # gesture = labels[gestInd]

        if (mode == 1 and 0<=number<=9):
          log(number, preprocessedLands)


        # Drawings
        frame = drawBrect(frame, boundRect)
        frame = drawLandMarks(frame, landmarkList)
        frame = drawInfo(frame, boundRect, handedness, mode, gesture)
        frame = drawZCircle(frame, newZ, landmarkList)
        drawingUtil.draw_landmarks(frame, hand_lands)

    cv2.imshow("Video", frame)

def setModeNumber(key, mode):
  number = -1
  if 48 <= key <= 57:
    number = key -48
  
  # n
  if key == 110:
    mode = 0
  
  # k
  if key == 107:
    mode = 1
  
  return number, mode

def log(number, landmarkList):
  # print([number, *landmarkList])
  with open("./models/dataset.csv", "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([number, *landmarkList])


if __name__ == '__main__':
  main()