#!./.venv/bin/python
import cv2
import mediapipe as mp
import numpy as np
import copy
import itertools
import csv

def main():
  handDetector = mp.solutions.hands.Hands(
    max_num_hands=2
  )
  drawingUtil = mp.solutions.drawing_utils

  cap = cv2.VideoCapture(0)

  mode = 0

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
        # print(hand_lands)
        boundRect = calc_bounding_box(frame, hand_lands)
        # print(boundRect)
        landmarkList = calc_land_image(frame, hand_lands)
        # print(landmarkList)


        if (mode == 1 and 0<=number<=9):
          preprocessedLands = preprocess(landmarkList)
          log(number, preprocessedLands)


        # frame = 
        frame = drawBrect(frame, boundRect)
        frame = drawLandMarks(frame, landmarkList)
        frame = drawInfo(frame, boundRect, handedness, mode)
        drawingUtil.draw_landmarks(frame, hand_lands)
        # frame = cv2.flip(frame, 1)


        

    # cv2.imshow("Video", cv2.flip(frame, 1))
    cv2.imshow("Video", frame)

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

  for _, land in enumerate(landmarks.landmark):
    landx = min(int(land.x * imw), imw-1)
    landy = min(int(land.y * imh), imh-1)

    landar.append([landx, landy])
    
  
  return landar

def drawBrect(image, brect):
  cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0,0,0), 1)

  return image

def drawLandMarks(image, landmark_point):
  if len(landmark_point) > 0:
    color = (255, 255, 255)
    thikness = 2
    # Thumb
    cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
            color, thikness)

    # Index
    cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[5]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
            color, thikness)

    #Middle
    cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[9]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
            color, thikness)
    
    # Ring
    cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[13]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
            color, thikness)
    
    # Pinky
    cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[17]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
            color, thikness)

    # Palm
    cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[0]),
            color, thikness)
    cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[0]),
            color, thikness)

    cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[5]),
            color, thikness)

    cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[9]),
            color, thikness)

    cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
            color, thikness)
        
    cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
            color, thikness)
  return image

def drawInfo(image, brect, handedness, mode):
  cv2.rectangle(image, (brect[0], brect[1]-10), (brect[2], brect[1] - 32), (0,0,0), -1)

  text = handedness.classification[0].label[0:]
  cv2.putText(image, text, (brect[0] + 5, brect[1] - 14),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
              1)
  
  if mode == 1:
    cv2.putText(image, 
                "MODE: " + "Logging input",
                (10, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 0),
                2
                )
  return image

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
  with open("./dataset.csv", "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([number, *landmarkList])

def preprocess(landmarks):
  res = copy.deepcopy(landmarks)
  basex, basey = 0,0

  for index, landpoint in enumerate(res):
    if index == 0:
      basex, basey = landpoint[0], landpoint[1]
    
    res[index][0] = res[index][0] - basex
    res[index][1] = res[index][1] - basey

  res = list(itertools.chain.from_iterable(res))

  maxVal = max(list(map(abs, res)))

  def norm(n):
    return n/maxVal

  res = list(map(norm, res))

  return res

main()