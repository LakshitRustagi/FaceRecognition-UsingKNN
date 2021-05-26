import numpy as np 
import cv2
import os


###  KNN 
def distance(p1,p2):
    return np.linalg.norm(p1-p2)

def knn(train,test,k):
    X_train = train[:,:-1]
    Y_train = train[:,-1]
    vals = []
    for i in range(train.shape[0]):
        dist = distance(X_train[i],test)
        vals.append((dist,Y_train[i]))
        
    vals = sorted(vals)
    vals = vals[:k]
    vals = np.array(vals)
    new_vals = np.unique(vals[:,1],return_counts=True)
    index = np.argmax(new_vals[1])
    return new_vals[0][index]

######
dataset_path = './data/'
labels = []
class_id = 0
names = {}
face_data = []
######

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        data_file = np.load(dataset_path+fx)
        face_data.append(data_file)
        names[class_id] = fx[:-4]
        
        class_level = class_id*np.ones(data_file.shape[0],)
        class_id += 1
        labels.append(class_level)

face_data = np.concatenate(face_data,axis=0)
labels = np.concatenate(labels,axis=0).reshape((-1,1))

training_data = np.concatenate((face_data,labels),axis=1)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret,frame = cap.read()
    
    if ret == False:
        continue
    
    faces = face_cascade.detectMultiScale(frame,1.3,2)
    
    if len(faces)==0:
        continue
    
    for face in faces:
        x,y,w,h = face
        ## Region of Interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        out = knn(training_data,face_section.flatten(),5)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),3)
        # Adding text to the frame
        frame = cv2.putText(frame,names[int(out)],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3,cv2.LINE_AA)
        
    cv2.imshow('Image',frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()
        