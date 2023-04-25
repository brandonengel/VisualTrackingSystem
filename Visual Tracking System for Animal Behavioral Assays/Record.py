# Python program to save a
# video using OpenCV
import cv2
import time

BGR_COLOR = {'red': (0,0,255),
             'green': (127,255,0),
             'blue': (255,127,0),
             'yellow': (0,127,255),
             'black': (0,0,0),
             'white': (255,255,255)}

filename = input('Please Enter a name for the following recording:\n')
print("Recording will begin shortly. . .")
# Create an object to read
# from camera
video = cv2.VideoCapture(0)

# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
	print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter(filename + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
print("\nOpen Field Test Recording in Process\nPress 'S' to Stop & Save Recording\n(You must be engaged with the 'Frame' window)")	
ret, frame = video.read()
time.sleep(1)
frames = 0
while(True):     
	ret, frame = video.read()

	if ret == True:
                
		# Write the frame into the
		# file 'filename.avi'
		result.write(frame)
		frames = frames + 1
		timePass = frames * (1/30)
		cv2.putText(frame, str('%.2f' % timePass) + ' sec', (230,450), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['red'], 4)
		cv2.putText(frame, 'LIVE RECORDING', (190,50), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['red'], 4)
		#if timePass < 60:
                #cv2.putText(frame, str('%.1f' % timePass) + ' sec', (230,450), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['red'], 4)
##                else:
##                        minutes = timePass / 60
##                        sec = timePass % 60
##                        cv2.putText(frame, str('%.0f' % minutes) + ' min ' + str('%.1f' % sec) + ' sec', (210,450), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['red'], 4)
##                
		# Display the frame
		# saved in the file
		cv2.imshow('Open Field Test Live Recording', frame)
		#time = time.perf_counter()
		
		#cv2.putText(frame, time.time(), (5,420), cv2.FONT_HERSHEY_DUPLEX, 1, BGR_COLOR['white'])
		# Press S on keyboard
		# to stop the process
		if cv2.waitKey(1) & 0xFF == ord('s'):
			break

	# Break the loop
	else:
		break

# When everything done, release
# the video capture and video
# write objects
video.release()
result.release()
print("The video was successfully saved")
time.sleep(2)
# Closes all the frames
cv2.destroyAllWindows()

