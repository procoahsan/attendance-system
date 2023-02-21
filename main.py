import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime


path = 'image attendance'
images = []
Classnames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    current_img = cv2.imread(f'{path}\{cl}')
    images.append(current_img)
    Classnames.append(os.path.splitext(cl)[0])
print(Classnames)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markattendance(name):
    with open('Attendance.csv','r+') as f:
        mydatalist= f.readlines()
        namelist=[]
        print(mydatalist)
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dstring=now.strftime('%m:%d:%y')
            tstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{tstring},{dstring}')
        else:
            print('',end='')
encodlistknown = findEncodings(images)
print(len(encodlistknown))
print("Ending process is completed ")

cap = cv2.VideoCapture(0)
while True:
   ret, img = cap.read()
   imgs = cv2.resize(img,(0,0),None,0.25,0.25)
   rect_faces = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   face_curframe = face_recognition.face_locations(imgs)
   encodecurrent_img = face_recognition.face_encodings(imgs,face_curframe)

   for encodeface,faceloc in zip(encodecurrent_img,face_curframe):
       matches = face_recognition.compare_faces(encodlistknown,encodeface,0.45)
       facedis = face_recognition.face_distance(encodlistknown,encodeface)
#       print(facedis)

       matchindex = np.argmin(facedis)
       name = Classnames[matchindex]
       if facedis[matchindex] < 0.50:
           name = Classnames[matchindex].upper()
           markattendance(name)
       else:
           name = 'Unknown'
       # print(name)
       y1, x2, y2, x1 = faceloc
       y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
       cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
       cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
       cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

   cv2.imshow('webcam',img)
   if cv2.waitKey(1) & 0xFF==ord('u'):
       break
cv2.waitKey(1)
cap.release()