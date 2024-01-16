from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

JUGGLE_POINT = 90

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to optional video file")

# buffer for max amount of points (in this case balls?) that can be tracked
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the color "green"
# use the HSV color space
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
# greenLower = (29, 150, 6)
# greenUpper = (64, 255, 200)

# initialize the list of tracked points
pts = deque(maxlen=args["buffer"])

# if a video file was not supplied, grab the reference to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, initialize a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow camera/video file to load up
time.sleep(2.0)

# initialize juggle counter, variables for tracking juggles
juggleCount = 0
ballUp = False
highBall = 10000

# keep looping until user quits or video ends
while True:
	# grab the current frame
    frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
    # if video was supplied, frame is the second element; otherwise frame is frame
    frame = frame[1] if args.get("video", False) else frame

	# if a video frame is none, the video has ended
    if frame is None:
        break

	# resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for our color green, then perform a series of 
    # dilations and erosions to remove any small blobs left in the mask
    # is there a better way to handle this?
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
	
    # find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    
    # initialize variable to store the yValues (heights) of the balls
    yValues = []

    # for each contour (hopefully one of our balls) found
    for c in cnts:
        # grab the coordinates (x, y) of the center of the ball and the radius
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # only proceed if the radius meets a minimum size
        if radius > 10:
            yValues.append(y)
            # print(f'radius of c is: {radius}')
            # print(f'(x,y): {(x,y)}')
            # what is going on here?
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # update the list of tracked points
            pts.appendleft(center)

    # DETERMINE IF A JUGGLE CYCLE (one ball went up and then down) WAS COMPLETED
    # this can be turned into a function (and probably improved)
    if len(yValues) > 0:
        newMaxBall = min(yValues)
        if ballUp:
            allBallsDown = True
            for yValue in yValues:
                if yValue < JUGGLE_POINT:
                        allBallsDown = False
            if allBallsDown and highBall > 20 and highBall < 200: 
                ## no balls are above a certain height AND previous high ball was lower than a certain height (attempt to tackle ball going above view problem)
                ballUp = False
                juggleCount += 1
                print(juggleCount)
                # print(f'{highBall} -> {newMaxBall}')
        else: # ballUp is false
            for yValue in yValues:
                    if yValue < JUGGLE_POINT:
                        # print(f'ball up at {yValue}')
                        ballUp = True
        highBall = newMaxBall
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
	# break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()