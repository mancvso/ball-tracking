import cv2
import numpy as np

class Capture(object):
  """docstring for Capture"""
  def __init__(self, cam_number):
    super(Capture, self).__init__()
    # create video capture
    self.cap = cv2.VideoCapture(cam_number)
    self.best_cnt = 1

  def run(self):
    while(1):

      # read the frames
      _,frame = self.cap.read()

      # smooth it
      frame = cv2.blur(frame,(3,3))

      # convert to hsv and find range of colors
      hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
      thresh = cv2.inRange(hsv,np.array((0, 80, 80)), np.array((20, 255, 255)))
      thresh2 = thresh.copy()

      # find contours in the threshold image
      contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

      # finding contour with maximum area and store it as best_cnt
      max_area = 0
      for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
          max_area = area
          self.best_cnt = cnt

      # finding centroids of best_cnt and draw a circle there
      M = cv2.moments(self.best_cnt)
      cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
      cv2.circle(frame,(cx,cy),5,100,3)
      cv2.circle(frame,(cx,cy-80),50,200,6)

      cv2.drawContours(frame,contours,-1,(0,255,0),3)

      #leftmost = tuple(contours[contours[:,:,0].argmin()][0])
      #rightmost = tuple(contours[contours[:,:,0].argmax()][0])
      #topmost = tuple(contours[contours[:,:,1].argmin()][0])
      #bottommost = tuple(contours[contours[:,:,1].argmax()][0])

      #cv2.circle(frame,leftmost,5,50,2)
      #cv2.circle(frame,rightmost,5,50,2)
      #cv2.circle(frame,topmost,5,50,2)
      #cv2.circle(frame,bottommost,5,50,2)

      # Show it, if key pressed is 'Esc', exit the loop
      cv2.imshow('frame',frame)
      cv2.imshow('thresh',thresh2)
      if cv2.waitKey(33)== 27:
        break

    # Clean up everything before leaving
    cv2.destroyAllWindows()
    self.cap.release()
