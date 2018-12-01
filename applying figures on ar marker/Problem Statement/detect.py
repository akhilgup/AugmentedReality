import numpy as np
import cv2
import cv2.aruco as aruco
import math
import time
"""
**************************************************************************
*                  E-Yantra Robotics Competition
*                  ================================
*  This software is intended to check version compatiability of open source software
*  Theme: Thirsty Crow
*  MODULE: Task1.1
*  Filename: detect.py
*  Version: 1.0.0  
*  Date: October 31, 2018
*  
*  Author: e-Yantra Project, Department of Computer Science
*  and Engineering, Indian Institute of Technology Bombay.
*  
*  Software released under Creative Commons CC BY-NC-SA
*
*  For legal information refer to:
*        http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode 
*     
*
*  This software is made available on an “AS IS WHERE IS BASIS”. 
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*  
*  e-Yantra - An MHRD project under National Mission on Education using 
*  ICT(NMEICT)
*
**************************************************************************
"""

####################### Define Utility Functions Here ##########################
"""
Function Name : getCameraMatrix()
Input: None
Output: camera_matrix, dist_coeff
Purpose: Loads the camera calibration file provided and returns the camera and
         distortion matrix saved in the calibration file.
"""
def getCameraMatrix():
	with np.load('System.npz') as X:
		camera_matrix, dist_coeff, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
	return camera_matrix, dist_coeff

"""
Function Name : sin()
Input: angle (in degrees)
Output: value of sine of angle specified
Purpose: Returns the sine of angle specified in degrees
"""
def sin(angle):
	return math.sin(math.radians(angle))

"""
Function Name : cos()
Input: angle (in degrees)
Output: value of cosine of angle specified
Purpose: Returns the cosine of angle specified in degrees
"""
def cos(angle):
	return math.cos(math.radians(angle))



################################################################################


"""
Function Name : detect_markers()
Input: img (numpy array), camera_matrix, dist_coeff
Output: aruco list in the form [(aruco_id_1, centre_1, rvec_1, tvec_1),(aruco_id_2,
        centre_2, rvec_2, tvec_2), ()....]
Purpose: This function takes the image in form of a numpy array, camera_matrix and
         distortion matrix as input and detects ArUco markers in the image. For each
         ArUco marker detected in image, paramters such as ID, centre coord, rvec
         and tvec are calculated and stored in a list in a prescribed format. The list
         is returned as output for the function
"""
def detect_markers(img, camera_matrix, dist_coeff):
	markerLength = 100
	aruco_list = []
	######################## INSERT CODE HERE ########################
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)

	parameters = aruco.DetectorParameters_create()

	corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
	for i in range(len(ids)):

		rvec, tvec,_= aruco.estimatePoseSingleMarkers(corners[i], markerLength, camera_matrix, dist_coeff)
		respe=np.reshape(corners[i],(4,2))
		mean=tuple(np.mean(respe,0))
		
		

	
		t = (ids[i],mean,rvec,tvec)
		aruco_list.append(t)


	

 

	
	##################################################################
	return aruco_list

"""
Function Name : drawAxis()
Input: img (numpy array), aruco_list, aruco_id, camera_matrix, dist_coeff
Output: img (numpy array)
Purpose: This function takes the above specified outputs and draws 3 mutually
         perpendicular axes on the specified aruco marker in the image and
         returns the modified image.
"""
def drawAxis(img, aruco_list, aruco_id, camera_matrix, dist_coeff):
	for x in aruco_list:
		if aruco_id == x[0]:
			rvec, tvec = x[2], x[3]
	markerLength = 100
	m = markerLength/2
	pts = np.float32([[-m,m,0],[m,m,0],[-m,-m,0],[-m,m,m]])
	pt_dict = {}
	imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
	for i in range(len(pts)):
		pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
	src = pt_dict[tuple(pts[0])];   dst1 = pt_dict[tuple(pts[1])];
	dst2 = pt_dict[tuple(pts[2])];  dst3 = pt_dict[tuple(pts[3])];
	img = cv2.line(img, src, dst1, (0,255,0), 4)
	img = cv2.line(img, src, dst2, (255,0,0), 4)
	img = cv2.line(img, src, dst3, (0,0,255), 4)

	
	return img

"""
Function Name : drawCube()
Input: img (numpy array), aruco_list, aruco_id, camera_matrix, dist_coeff
Output: img (numpy array)
Purpose: This function takes the above specified outputs and draws a cube
         on the specified aruco marker in the image and returns the modified
         image.
"""
def drawCube(img, ar_list, ar_id, camera_matrix, dist_coeff):
	for x in ar_list:
		if ar_id == x[0]:
			rvec, tvec = x[2], x[3]
	markerLength = 100
	m = markerLength/2
	######################## INSERT CODE HERE ########################

	pts = np.float32([[-m,m,0],[-m,-m,0],[m,-m,0],[m,m,0],
                   [-m,m,m],[-m,-m,m],[m,-m,m],[m,m,m] ])

	

	imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
	
	# time.sleep(2323)

	imgpts = np.int32(imgpts).reshape(-1,2)

	img = cv2.drawContours(img, [imgpts[:4]],-1,(0,0,255),3)

	for i,j in zip(range(4),range(4,8)):
		img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),3)

	img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    

	##################################################################
	return img

"""
Function Name : drawCylinder()
Input: img (numpy array), aruco_list, aruco_id, camera_matrix, dist_coeff
Output: img (numpy array)
Purpose: This function takes the above specified outputs and draws a cylinder
         on the specified aruco marker in the image and returns the modified
         image.
"""
def drawCylinder(img, ar_list, ar_id, camera_matrix, dist_coeff):
	for x in ar_list:
		if ar_id == x[0]:
			rvec, tvec = x[2], x[3]
	markerLength = 100
	radius = markerLength/2; height = markerLength*1.5
	######################## INSERT CODE HERE ########################

	angle=[0,30,60,90,120,150,180,210,240,270,300,330,360]

	pts = np.float32([[0,0,0]])

	for i in angle:
		x=radius*cos(i)
		y=radius*sin(i)
		val=[[x,y,0]]
		pts=np.concatenate((pts,val))

	val=[[0,0,height]]
	pts=np.concatenate((pts,val))
	for i in angle:
		x=radius*cos(i)
		y=radius*sin(i)
		val=[[x,y,height]]
		pts=np.concatenate((pts,val))	


	# pts = np.float32([[0,0,0],[-radius,0,0],[0,-radius,0],[radius,0,0],[0,radius,0]])

	imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
	imgpts = np.int32(imgpts).reshape(-1,2)
	pt_dict = {}

	img = cv2.drawContours(img, [imgpts[:14]],-1,(255,0,0),3)

	for i in range(len(pts)//2):
		pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
	src = pt_dict[tuple(pts[0])];   dst1 = pt_dict[tuple(pts[1])];
	dst2 = pt_dict[tuple(pts[2])];  dst3 = pt_dict[tuple(pts[3])];
	dst4 = pt_dict[tuple(pts[4])];  dst5 = pt_dict[tuple(pts[5])];
	dst6 = pt_dict[tuple(pts[6])];  dst7 = pt_dict[tuple(pts[7])];
	dst8 = pt_dict[tuple(pts[8])];  dst9 = pt_dict[tuple(pts[9])];
	dst10 = pt_dict[tuple(pts[10])];  dst11 = pt_dict[tuple(pts[11])];
	dst12 = pt_dict[tuple(pts[12])];  dst13 = pt_dict[tuple(pts[13])];

	img = cv2.line(img, src, dst1, (255,0,0), 4)
	img = cv2.line(img, src, dst2, (255,0,0), 4)
	img = cv2.line(img, src, dst3, (255,0,0), 4)
	img = cv2.line(img, src, dst4, (255,0,0), 4)
	img = cv2.line(img, src, dst5, (255,0,0), 4)
	img = cv2.line(img, src, dst6, (255,0,0), 4)
	img = cv2.line(img, src, dst7, (255,0,0), 4)
	img = cv2.line(img, src, dst8, (255,0,0), 4)
	img = cv2.line(img, src, dst9, (255,0,0), 4)
	img = cv2.line(img, src, dst10, (255,0,0), 4)
	img = cv2.line(img, src, dst11, (255,0,0), 4)
	img = cv2.line(img, src, dst12, (255,0,0), 4)
	img = cv2.line(img, src, dst13, (255,0,0), 4)

	for i,j in zip(range(14),range(14,28)):
		img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255,0,0),3)

	for i in range(len(pts)//2,len(pts)):
		pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
	src = pt_dict[tuple(pts[14])];   dst1 = pt_dict[tuple(pts[15])];
	dst2 = pt_dict[tuple(pts[16])];  dst3 = pt_dict[tuple(pts[17])];
	dst4 = pt_dict[tuple(pts[18])];  dst5 = pt_dict[tuple(pts[19])];
	dst6 = pt_dict[tuple(pts[20])];  dst7 = pt_dict[tuple(pts[21])];
	dst8 = pt_dict[tuple(pts[22])];  dst9 = pt_dict[tuple(pts[23])];
	dst10 = pt_dict[tuple(pts[24])];  dst11 = pt_dict[tuple(pts[25])];
	dst12 = pt_dict[tuple(pts[26])];  dst13 = pt_dict[tuple(pts[27])];

	img = cv2.line(img, src, dst1, (255,0,0), 4)
	img = cv2.line(img, src, dst2, (255,0,0), 4)
	img = cv2.line(img, src, dst3, (255,0,0), 4)
	img = cv2.line(img, src, dst4, (255,0,0), 4)
	img = cv2.line(img, src, dst5, (255,0,0), 4)
	img = cv2.line(img, src, dst6, (255,0,0), 4)
	img = cv2.line(img, src, dst7, (255,0,0), 4)
	img = cv2.line(img, src, dst8, (255,0,0), 4)
	img = cv2.line(img, src, dst9, (255,0,0), 4)
	img = cv2.line(img, src, dst10, (255,0,0), 4)
	img = cv2.line(img, src, dst11, (255,0,0), 4)
	img = cv2.line(img, src, dst12, (255,0,0), 4)
	img = cv2.line(img, src, dst13, (255,0,0), 4)

	img = cv2.drawContours(img, [imgpts[14:]],-1,(255,0,0),3)
	



	
	##################################################################
	return img

"""
MAIN CODE
This main code reads images from the test cases folder and converts them into
numpy array format using cv2.imread. Then it draws axis, cubes or cylinders on
the ArUco markers detected in the images.
"""


if __name__=="__main__":
	cam, dist = getCameraMatrix()
	img = cv2.imread("../TestCases/image_1.jpg")
	aruco_list = detect_markers(img, cam, dist)
	for i in aruco_list:
		img = drawAxis(img, aruco_list, i[0], cam, dist)
		img = drawCube(img, aruco_list, i[0], cam, dist)
		img = drawCylinder(img, aruco_list, i[0], cam, dist)
	cv2.imshow("img", img)
	cv2.waitKey(0)
	# cv2.destroyAllWindows()
