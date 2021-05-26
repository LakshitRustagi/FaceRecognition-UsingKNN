import cv2 
import numpy as np 

# 1. Read and show video stream , capture images
# 2. Detect faces and show bounding box
# 3. For every 10th image flatten the largest face image array and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


# skip counter to check for every 10th frame
skip = 0
# for storing data of face sections
face_data = []
# location where npy file will be saved
path = './data/'
#  Input the name of the file 
filename = input("Enter the name of the person ")

# Classifier object
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    
    if ret==False:
        continue
    
    faces = face_cascade.detectMultiScale(frame,1.2,5,2)
    # Sorting the list in decreasing order on the basis of area (ie w*h )
    faces = sorted(faces,key= lambda f:f[2]*f[3],reverse=True)
    
    for face in faces:
        x,y,w,h = face
        # Creating a bounding box 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255))
        # Creating a section of the initial frame by giving a padding of 15 pixels on all sides
        offset = 10
        face_section = frame[y-offset:y+h+offset , x-offset:x+w+offset]
        # Resizing the section
        face_section = cv2.resize(face_section,(100,100))
        
        skip += 1
        # Storing every 10th frame in the list
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow('Video',frame)
    cv2.imshow('Face section',face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
# Creating a npy file to store the data of the largest faces
np.save(path+filename+'.npy',face_data)
print('Data is saved at '+path+filename+'.npy')


cap.release()
cv2.destroyAllWindows()
    
    