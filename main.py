import face_recognition
import numpy as np
import cv2
import csv
import os
from datetime import datetime

#input from webcam
video_capture=cv2.VideoCapture(0)

#load known faces
vinayak_image=face_recognition.load_image_file("faces/vinayak.png")
vinayak_face_encoding=face_recognition.face_encodings(vinayak_image)[0]                           #encoding of images

vanshika_image=face_recognition.load_image_file("faces/Vanshika.jpg")
vanshika_face_encoding=face_recognition.face_encodings(vanshika_image) [0]

babita_image=face_recognition.load_image_file("faces/Babita.png")
babita_face_encoding=face_recognition.face_encodings(babita_image)[0]

rakesh_image=face_recognition.load_image_file("faces/Rakesh.png")
rakesh_face_encoding=face_recognition.face_encodings(rakesh_image)[0]

rahul_image=face_recognition.load_image_file("faces/Rahul Gandhi.jpg")
rahul_face_encoding=face_recognition.face_encodings(rahul_image)[0]

narendra_image=face_recognition.load_image_file("faces/Narendra Modi.jpg")
narendra_face_encoding=face_recognition.face_encodings(narendra_image)[0]

arvind_image=face_recognition.load_image_file("faces/Arvind.jpg")
arvind_face_encoding=face_recognition.face_encodings(arvind_image)[0]

known_face_encoding=[
    vinayak_face_encoding,
    vanshika_face_encoding,
    babita_face_encoding,
    rakesh_face_encoding,
    rahul_face_encoding,
    narendra_face_encoding,
    arvind_face_encoding
]

known_face_name=[
    "Vinayak Lathwal",
    "Vanshika",
    "Babita",
    "Rakesh",
    "Rahul Gandhi",
    "Narendra Modi",
    "Arvind Kejriwal"
]

#list of names of known faces(expected)
students=known_face_name.copy()

# the encoding of faces on the webcam at that moment
face_location=[]

# the encoding of faces on the webcam at that moment                 
face_encodings=[]   

# name=[]

s=True

#get current date and time
now = datetime.now()
current_date=now.strftime("%Y-%m-%d")

#creating the csv file where data will be stored
f=open(f"{current_date}.csv",'w+',newline='')
write=csv.writer(f)

#creating while infinite loop
while True:
    #capturing video
    _,frame=video_capture.read()
    #reszizng the frame
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    #converting frame to RGB as frame stores image in bgr
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    #recogonizing faces with knwon
    face_location=face_recognition.face_locations(rgb_small_frame)
    face_encodings=face_recognition.face_encodings(rgb_small_frame)
    #comparing captured faces with known faces
    for face_encoding in face_encodings:
        matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
        #calculating similarity
        face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
        best_match_index=np.argmin(face_distance)

        
        name=""
        if(matches[best_match_index]):
            name=known_face_name[best_match_index]

        #add text to frame if a person is present
        if name in known_face_name:
            font=cv2.FONT_ITALIC
            bottomleftcrorneroftext=(50,100)
            fontScale=1
            fontColor=(0,0,0)
            thickness=3
            linetype=2
            cv2.putText(frame,name+" "+"is Present",bottomleftcrorneroftext,font,fontScale,fontColor,thickness,linetype)
                                                                        
            #if name found and got the attendance remove name from list
            if name in students:
                students.remove(name)
                current_time=datetime.now().strftime("%H:%M:%S")
                write.writerow([name,current_time])

    cv2.imshow("Attendance",frame)
    if cv2.waitKey(1) & 0xFF == ord(';'):
        break        

video_capture.release()
cv2.destroyAllWindows()
f.close()


