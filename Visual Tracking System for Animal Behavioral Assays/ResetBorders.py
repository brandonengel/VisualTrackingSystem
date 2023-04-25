import numpy as np
import cv2
import glob, os, sys, time, datetime

# TODO corners save when offline processing
                                                                                                                                            #SET initial Values
ONLINE = True
CALIBRATE = False
RELATIVE_DESTINATION_PATH = str(datetime.date.today()) + '_Border_Reset/'
FPS = 60
THRESHOLD_WALL_VS_FLOOR = 80
THRESHOLD_ANIMAL_VS_FLOOR = 70
HD = 1280, 640
BGR_COLOR = {'red': (0,0,255),
             'green': (127,255,0),
             'blue': (255,127,0),
             'yellow': (0,127,255),
             'black': (0,0,0),
             'white': (255,255,255)}
WAIT_DELAY = 1

#######

perspectiveMatrix = dict()
croppingPolygon = np.array([[0,0]])
croppingPolygons = dict()
tetragons = []
name = ""

RENEW_TETRAGON = True

def counterclockwiseSort(tetragon):
    tetragon = sorted(tetragon, key=lambda e: e[0])
    tetragon[0:2] = sorted(tetragon[0:2], key=lambda e: e[1])
    tetragon[2:4] = sorted(tetragon[2:4], key=lambda e: e[1], reverse=True)
    return tetragon

# TODO pointlike tetragon moving instead drawing it by clicking
# mouse callback function for drawing a cropping polygon
def drawFloorCrop(event, x, y, flags, params):
    global perspectiveMatrix, name, RENEW_TETRAGON
    imgCroppingPolygon = np.zeros_like(params['imgFloorCorners'])
    if event == cv2.EVENT_RBUTTONUP:
        cv2.destroyWindow(f'Floor Corners for {name}')
    if len(params['croppingPolygons'][name]) > 4 and event == cv2.EVENT_LBUTTONUP:
        RENEW_TETRAGON = True
        h = params['imgFloorCorners'].shape[0]
        # delete 5th extra vertex of the floor cropping tetragon
        params['croppingPolygons'][name] = np.delete(params['croppingPolygons'][name], -1, 0)
        params['croppingPolygons'][name] = params['croppingPolygons'][name] - [h,0]
        
        # Sort cropping tetragon vertices counter-clockwise starting with top left
        params['croppingPolygons'][name] = counterclockwiseSort(params['croppingPolygons'][name])
        # Get the matrix of perspective transformation
        params['croppingPolygons'][name] = np.reshape(params['croppingPolygons'][name], (4,2))
        #UpLeft = params['croppingPolygons'][name][0,0]
        tetragonVertices = np.float32(params['croppingPolygons'][name])
        tetragonVerticesUpd = np.float32([[0,0], [0,h], [h,h], [h,0]])
        perspectiveMatrix[name] = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)
##        yes = list(params['croppingPolygons'][name])
##
        pointOneX = str(params['croppingPolygons'][name][0,0])
        pointOneY = str(params['croppingPolygons'][name][0,1])
        pointTwoX = str(params['croppingPolygons'][name][1,0])
        pointTwoY = str(params['croppingPolygons'][name][1,1])
        pointThreeX = str(params['croppingPolygons'][name][2,0])
        pointThreeY = str(params['croppingPolygons'][name][2,1])
        pointFourX = str(params['croppingPolygons'][name][3,0])
        pointFourY = str(params['croppingPolygons'][name][3,1])
        print('New Selected Coordinates (Sorted Counter-Clockwise from Top Left):\n')
        print(params['croppingPolygons'][name])
        file = open(RELATIVE_DESTINATION_PATH + name + '_New_Border_Coordinates.doc', 'w')
        file.write('BORDER RESET REPORT\n\nThe following are the coordinates that you selected for a new Open Field Test Boundary:\n\nFirst Point (Top Left): (' +
                   pointOneX + ',' + pointOneY + ')\nSecond Point (Bottom Left): (' +
                   pointTwoX + ',' + pointTwoY + ')\nThird Point (Bottom Right): (' +
                   pointThreeX + ',' + pointThreeY + ')\nFourth Point (Top Right): (' +
                   pointFourX + ',' + pointFourY + ')\n\nTo RESET the boundary for the Open Field Test you will need to EDIT the Analyze.py file and CHANGE the corresponding coordinate values.')
        file.close()
        print('\nThe New Border Coordinates will be available in a Word document in the ' + RELATIVE_DESTINATION_PATH + ' folder.\n\nPress ENTER to Close Window')
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(params['croppingPolygons'][name]) == 4 and RENEW_TETRAGON:
            params['croppingPolygons'][name] = np.array([[0,0]])
            RENEW_TETRAGON = False
        if len(params['croppingPolygons'][name]) == 1:
            params['croppingPolygons'][name][0] = [x,y]
        params['croppingPolygons'][name] = np.append(params['croppingPolygons'][name], [[x,y]], axis=0)
    if event == cv2.EVENT_MOUSEMOVE and not (len(params['croppingPolygons'][name]) == 4 and RENEW_TETRAGON):
        params['croppingPolygons'][name][-1] = [x,y]
        if len(params['croppingPolygons'][name]) > 1:
            cv2.fillPoly(
                imgCroppingPolygon,
                [np.reshape(
                    params['croppingPolygons'][name],
                    (len(params['croppingPolygons'][name]),2)
                )],
                BGR_COLOR['green'], cv2.LINE_AA)
            imgCroppingPolygon = cv2.addWeighted(params['imgFloorCorners'], 1.0, imgCroppingPolygon, 0.5, 0.)
            cv2.imshow(f'Floor Corners for {name}', imgCroppingPolygon)

def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return np.abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

def floorCrop(filename):
    global perspectiveMatrix, tetragons, name, croppingPolygons
    name = os.path.splitext(filename)[0]
    cap = cv2.VideoCapture(filename)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Take first non-null frame and find corners within it
    ret, frame = cap.read()
    while not frame.any():
        ret, frame = cap.read()

    frame = frame[:, w-h : w]
    

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                                                 #Transfroms ColorSpace to Gray
    kernelSize = (5,5)                                                                                                  #Define Convolution Size
    frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)                                                              #Function using frameGray input to kernelSize output of size 0
    retval, mask = cv2.threshold(frameBlur, THRESHOLD_WALL_VS_FLOOR, 255, cv2.THRESH_BINARY_INV)                        #Takes Source FrameBlur using threshold
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)                            


#    frame2 = frame[:, h-w : w]
#    frame2Gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)                                                                 #Transfroms ColorSpace to Gray
#    kernelSize = (3,3)
#    frame2Blur = cv2.GaussianBlur(frame2Gray, kernelSize, 0)                                                              #Function using frameGray input to kernelSize output of size 0
#    retval, mask = cv2.threshold(frame2Blur, THRESHOLD_WALL_VS_FLOOR, 255, cv2.THRESH_BINARY_INV)

    tetragons = []
    HALF_AREA = 0.5 * h * h
    for contour in contours:
        contourPerimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        contour = cv2.approxPolyDP(hull, 0.02 * contourPerimeter, True)

        # If the contour is convex tetragon
        # and its area is above a half of total frame area,
        # then it's most likely the floor
        if len(contour) == 4 and cv2.contourArea(contour) > HALF_AREA:
            contour = contour.reshape(-1, 2)
            max_cos = np.max([angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)])
            if max_cos < 0.3:
                tetragons.append(contour)
    frameGray = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)                                                            #Transfroms ColorSpace to RGB Scale
    imgSquare = np.zeros_like(frameGray)
    cv2.fillPoly(imgSquare, tetragons, BGR_COLOR['red'], cv2.LINE_AA)
    # cv2.add(frameGray, imgSquare / 2, frameGray)
    cv2.drawContours(frameGray, tetragons, -1, BGR_COLOR['red'], 2, cv2.LINE_AA)

    if len(tetragons) > 0:
        tetragonVertices = tetragons[0]
    else:
        tetragonVertices = np.float32([[0,0], [0,h], [h,h], [h,0]])
    # Sort the cropping tetragon vertices according to the following order:
    # [left,top], [left,bottom], [right,bottom], [right,top]
    tetragonVertices = counterclockwiseSort(tetragonVertices)
    croppingPolygons[name] = tetragonVertices
    tetragonVertices = np.float32(tetragonVertices)
    tetragonVerticesUpd = np.float32([[0,0], [0,h], [h,h], [h,0]])
    perspectiveMatrix[name] = cv2.getPerspectiveTransform(np.float32(croppingPolygons[name]), tetragonVerticesUpd)
    frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (h,h))
    
    imgFloorCorners = np.hstack([frame, frameGray])

    cv2.imshow(f'Floor Corners for {name}', imgFloorCorners)
    cv2.setMouseCallback(
        f'Floor Corners for {name}',
        drawFloorCrop,
        {'imgFloorCorners': imgFloorCorners, 'croppingPolygons': croppingPolygons},
        )
    k = cv2.waitKey(0)
    if k == 27:
        sys.exit()
    cv2.destroyWindow(f'Floor Corners for {name}')
    return tetragonVertices, perspectiveMatrix[name]

if len(sys.argv) > 1 and '--online' in sys.argv:
    ONLINE = True
if not os.path.exists(RELATIVE_DESTINATION_PATH):
    os.makedirs(RELATIVE_DESTINATION_PATH)
##
for filename in glob.glob('*.avi'):
        floorCrop(filename)
