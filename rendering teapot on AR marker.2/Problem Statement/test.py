from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
from PIL import Image
import cv2
import numpy as np
import cv2.aruco as aruco
import pygame

name = 'OpenGL Python Teapot'

texture_object = None
texture_background = None
camera_matrix = None
dist_coeff = None
cap = cv2.VideoCapture(0)
INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                           [-1.0,-1.0,-1.0,-1.0],
                           [-1.0,-1.0,-1.0,-1.0],
                           [ 1.0, 1.0, 1.0, 1.0]])

def getCameraMatrix():
        global camera_matrix, dist_coeff
        with np.load('Camera.npz') as X:
                camera_matrix, dist_coeff, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

def main():
        glutInit()
        getCameraMatrix()
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(625, 100)
        glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)
        window_id = glutCreateWindow("OpenGL")
        init_gl()
        glutDisplayFunc(display)
        glutIdleFunc(display)
        # glutReshapeFunc(resize)
        glutMainLoop()

"""
Function Name : init_gl()
Input: None
Output: None
Purpose: Initialises various parameters related to OpenGL scene.
"""  
def init_gl():
        global texture_object, texture_background
        glClearColor(0.0, 0.0, 0.0, 1.0)
        # glClearDepth(1.0) 
        # glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)   
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_TEXTURE_2D)
        glMatrixMode(GL_PROJECTION)
        # glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        gluLookAt(0,0,1,
                  0,0,0,
                  0,1,0)
        glPushMatrix()
        texture_background = glGenTextures(1)
        texture_object = glGenTextures(1)
"""
Function Name : resize()
Input: None
Output: None
Purpose: Initialises the projection matrix of OpenGL scene
"""
def resize(w,h):
        ratio = 1.0* w / h
        glMatrixMode(GL_PROJECTION)
        glViewport(0,0,w,h)
        gluPerspective(45, ratio, 0.1, 100.0)


def detect_markers(img):
        aruco_list = []
        ################################################################
        #################### Same code as Task 1.1 #####################
        ################################################################

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
            mean=np.mean(respe,0)
        
        

    
            t = (ids[i],mean,rvec,tvec)
            aruco_list.append(t)

        return aruco_list


def init_object_texture(image_filepath):
       
        tex=Image.open(image_filepath)
            
        ix = tex.size[0]
        iy = tex.size[1]
        tex = tex.tobytes("raw", "RGBA", 0, -1)
        print(ix,iy)

        glBindTexture(GL_TEXTURE_2D, texture_object)
        

        # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, tex)


        
        
        return None


def display():

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    # glLoadIdentity()
    ar_list = []
    ret, frame = cap.read()
    if ret == True:

        bg_image = cv2.flip(frame, 0)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)

        # draw background
        glBindTexture(GL_TEXTURE_2D, texture_background)
        glPushMatrix()
        glTranslatef(0.0,0.0,-10.0)
        draw_background()
        glPopMatrix()
        # ar_list = detect_markers(frame)
        # for i in ar_list:
        #     if i[0] == 8:
        #         overlay(frame, ar_list, i[0],"texture_1.png")
        #     if i[0] == 2:
        #         overlay(frame, ar_list, i[0],"texture_2.png")
        #     if i[0] == 7:
        #         overlay(frame, ar_list, i[0],"texture_3.png")
        #     if i[0] == 6:
        #         overlay(frame, ar_list, i[0],"texture_4.png")

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        # init_object_texture("texture_1.png")

        # glPushMatrix()
        # color = [1.0,0.,0.,1.]
        # # glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
        # # init_object_texture("texture_1.png")
        # # glRotatef(180,1,0,0);
        # # glRotatef(-45,0,1,0);
        # glutSolidTeapot(0.5) #-2,20,-20
        
        # glPopMatrix()
        image = overlay(frame)
        glutSwapBuffers()
    
    return

def draw_background():

    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
    glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
    glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
    glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
    glEnd()

def overlay(image):

    i_d = None
    centre = None
    rvecs = None
    tvecs = None

    try:
        # i_d, centre, rvecs, tvecs = detect_markers(image)
        ar_list = detect_markers(image)
        # print(type(int(ar_list[0][0])))
        i_d, centre, rvecs, tvec = int(ar_list[0][0]), ar_list[0][1], ar_list[0][2], ar_list[0][3]
        # print("aasdasdasdasdas",i_d, centre, rvecs, tvecs)
    
    
        # print(tvecs)
        rmtx = cv2.Rodrigues(rvecs)[0]
        
        print(tvecs)
     
        view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0][0][0]],
                                [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[0][0][1]],
                                [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[0][0][2]],
                                [0.0       ,0.0       ,0.0       ,1.0    ]])

        view_matrix = view_matrix * INVERSE_MATRIX

        view_matrix = np.transpose(view_matrix)

        if i_d == 8:
            init_object_texture("texture_1.png")
        if i_d == 2:
            init_object_texture("texture_2.png")
        if i_d == 7:
            init_object_texture("texture_3.png")
        if i_d == 6:
            init_object_texture("texture_4.png")

        glPushMatrix()
        # glLoadMatrixd(view_matrix)
        color = [1.0,0.,0.,1.]
        # glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
        
        # glRotatef(180,1,0,0);
        # glRotatef(-45,0,1,0);
        # glLoadMatrixd(view_matrix)
        glutSolidTeapot(0.1) #-2,20,-20
        
        glPopMatrix()


    except Exception as ex: 
        print("waah",ex)




if __name__ == '__main__': main()