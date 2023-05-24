# import numpy as np
# import cv2 as cv
# import time 
# cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
# cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
# cap.set(cv.CAP_PROP_FRAME_COUNT,30)

# # params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )

# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15, 15),
#                   maxLevel = 2,
#                   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# # Create some random colors
# color = np.random.randint(0, 255, (100, 3))

# # Take first frame and find corners in it
# time.sleep(1)
# ret, old_frame = cap.read()
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# # Apply Gaussian blur with a 3x3 kernel and sigma of 0
# old_gray = cv.GaussianBlur(old_gray, (3, 3), 0)
# mask = np.zeros_like(old_gray)

# # Define the ROI coordinates
# x, y, w, h = 210, 250, 229, 130

# # Set the ROI region in the mask to 1
# mask[y:y+h, x:x+w] = 1
# lower=np.array([72,46,36])
# upper=np.array([99,128,232])

# old_img=old_frame
# # mask = np.zeros_like(old_img)
# # old_img=cv.resize(old_img,[int(old_img.shape[0]/2),int(old_img.shape[1]/2)])
# old_img=cv.cvtColor(old_img,cv.COLOR_BGR2HSV)
# old_img=cv.inRange(old_img,lower,upper)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)

# # Create a dictionary to store the tracked points and their IDs
# points = {}
# id_counter = 0
# N=0
# while(1):
#     ret, frame = cap.read()
#     frame=cv.flip(frame,-1)
#     # if N==53:
#     #     print('Done')
#     #     break

#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
#     # frame_gray = frame_gray[y:y+h, x:x+w]
#     # calculate optical flow
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     # Update the ROI coordinates based on the new pixel positions
#     # x += int(p1.mean(axis=0)[0][0] - p0.mean(axis=0)[0][0])
#     # y += int(p1.mean(axis=0)[0][1] - p0.mean(axis=0)[0][1])
#     # Draw the updated ROI on the current frame
#     # cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     # Select good points
#     if p1 is not None:
#         good_new = p1[st==1]
#         good_old = p0[st==1]

#     # Draw the tracks and update the points dictionary
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
#         frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
#         # Update the tracked points dictionary
#         if i not in points:
#             points[i] = []
#         points[i].append((a, b))

#     img = cv.add(frame, mask)
#     new_velocity=0
#     a=np.array([])
#     for i, pts in points.items():
#         if len(pts) < 2:
#             continue
#         prev_point = pts[-2]
#         curr_point = pts[-1]
#         x1, y1 = prev_point
#         x2, y2 = curr_point
#         distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         elapsed_time = 1 / cap.get(cv.CAP_PROP_FPS)
#         velocity = distance / elapsed_time
#         a=np.append(a,velocity)
#         if velocity!=0:
#             i_current=i 
#     if a is not None:
#         print("velocity:", np.mean(a))
        
    
#     N+=1
#     cv.putText(img, "Velocity: {:.2f} pixels/frame".format(np.mean(a)), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv.imshow('frame', img)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break

#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)

# cv.destroyAllWindows()

#Code for hardware implementation
import numpy as np
import cv2 as cv

# Read in video file
cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv.CAP_PROP_FRAME_COUNT,30)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Read in first frame and select ROI around object
ret, frame = cap.read()
# roi = cv.selectROI(frame)
x, y, w, h = 210, 250, 229, 130
# Initialize variables
prev_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# define a kernel for dilation
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
# prev_frame= cv.dilate(prev_frame, kernel, iterations=1)
mask = np.zeros_like(prev_frame)
mask[y:y+h, x:x+w] = 1
p0= cv.goodFeaturesToTrack(prev_frame, mask = mask, **feature_params)
hsv = np.zeros_like(frame)
hsv[..., 1] = 255
num_frames = 0
N=0
# Create a dictionary to store the tracked points and their IDs
points = {}
id_counter = 0
# Loop over video frames
while True:
    # Read in next frame
    ret, frame = cap.read()
    frame=cv.flip(frame,-1)
    # if N==53:
    #     break
    
    # Convert frames to grayscale and calculate optical flow
    curr_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(prev_frame, curr_frame, p0, None, **lk_params)
    # Update the ROI coordinates based on the new pixel positions
    x += int(p1.mean(axis=0)[0][0] - p0.mean(axis=0)[0][0])
    y += int(p1.mean(axis=0)[0][1] - p0.mean(axis=0)[0][1])
    # Draw the updated ROI on the current frame
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    flow = cv.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        #Uncomment to see the feature points on the car being tracked
        cv.circle(frame, (int(a), int(b)), 5, (255,0,0), -1)
        if i not in points:
            points[i] = []
        points[i].append((a, b))
    # Compute the velocity of each car and print its ID and velocity
    V_list=np.array([])
    for i, pts in points.items():
        if len(pts) < 2:
            continue
        prev_point = pts[-2]
        curr_point = pts[-1]
        x1, y1 = prev_point
        x2, y2 = curr_point
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        elapsed_time = 1 / cap.get(cv.CAP_PROP_FPS)
        velocity = distance / elapsed_time
        if velocity!=0:
            V_list=np.append(V_list,velocity)
            # print("Object id",i,"velocity:", velocity)
    print("velocity:", np.mean(V_list))
    # # Draw optical flow lines on frame
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.rectangle(bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #Uncomment the next line to watch optical flow
    cv.putText(bgr, "Velocity: {:.2f} pixels/frame".format(np.mean(V_list)), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("frame_1", bgr)
    #Uncomment the next line to see the feature points being tracked on the car
    cv.putText(frame, "Velocity: {:.2f} pixels/frame".format(np.mean(V_list)), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("frame_2", frame)
    # Create a grid of points to display arrows
    step = 20
    h, w = frame.shape[:2]
    q, u = np.mgrid[int(step / 2):h:step, int(step / 2):w:step].reshape(2, -1).astype(int)
    # Calculate coordinates of end points of arrows
    fx, fy = flow[q, u].T
    e, f = np.round(u + fx).astype(int), np.round(q + fy).astype(int)
    # Draw arrows on the image
    for i, j, k, l in zip(u, q, e, f):
        cv.arrowedLine(frame, (i, j), (k+10, l+10), (0, 255, 0), 1)
    N+=1
    p0=p1
    #Use the next line to see the optical shade
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == 27:
            break
    
    # Update variables for next iteration
    prev_frame = curr_frame
    num_frames += 1
cv.destroyAllWindows()