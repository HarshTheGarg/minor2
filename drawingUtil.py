import cv2

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

def drawZCircle(image, z, landmarks):
  cv2.circle(image, tuple(landmarks[8]), int(z*2), (0, 255, 255), 1)

  return image