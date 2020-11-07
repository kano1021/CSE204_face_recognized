import cv2
import sys
from PIL import Image
from keras.models import load_model
import model
import age_model
import eth_model
from load_data import IMAGE_SIZE

def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)
    
    # Video source: camera streaming
    cap = cv2.VideoCapture(camera_idx)                
    
    # Using OpenCV face classifier
    classifier = cv2.CascadeClassifier('/Users/YuYinfeng/Desktop/Semester 4/CSE_204_ML/Project/haarcascade_frontalface_alt2.xml')

    # The color of the frame containing the face
    color = (0, 255, 0)

    g_model = model.Gender_Model(None)
    g_model.load_model(file_name = '/Users/YuYinfeng/Desktop/Semester 4/CSE_204_ML/Project/gender_classifier.h5')
    print('g_model loaded')
    a_model = age_model.Age_Model(None)
    a_model.load_model(file_name = '/Users/YuYinfeng/Desktop/Semester 4/CSE_204_ML/Project/age_classifier.h5')
    print('a_model loaded')
    e_model = eth_model.Ethni_Model(None)
    e_model.load_model(file_name = '/Users/YuYinfeng/Desktop/Semester 4/CSE_204_ML/Project/sfullmodel.h5')
    print('e_model loaded')

    age_list = ["0-5", "6-10", "11-5", "16-20", "21-30", "31-40", "41-60", "61-80", "above 80"]
    ethni_list = ["Caucasian", "Negro", "Mongoloid", "Indian", "Others"]
        
    while cap.isOpened():
        ok, frame = cap.read() # capture a frame
        if not ok:           
            break         

        # transform the frame into grey image
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 
        
        # face detection
        faceRects = classifier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        
        if len(faceRects) > 0:            # face detected                              
            for faceRect in faceRects:  # find every face in the frame
                x, y, w, h = faceRect        

                # capture the image of the face
                image = frame[y - 15: y + h + 15, x - 15: x + w + 15]
                gender = g_model.predict(image)
                age = a_model.predict(image)
                eth = e_model.predict(image)

                # gender classifier and age classifier
                if gender == None or age == None or eth == None: continue
                if gender == 0:                                                        
                    
                    # labelling
                    cv2.putText(frame,'Male', 
                                (x + 30, y + 30),                      # coordinate
                                cv2.FONT_HERSHEY_SIMPLEX,              # font
                                1,                                     # fontsize
                                (255,0,0),                             # color
                                2)                                     # thickness
                elif gender == 1:                                                       
                    # labelling
                    cv2.putText(frame,'Female', 
                                (x + 30, y + 30),                      # coordinate
                                cv2.FONT_HERSHEY_SIMPLEX,              # font
                                1,                                     # fontsize
                                (255,0,255),                           # color
                                2)                                     # thickness

                # age labelling
                cv2.putText(frame, age_list[age], 
                            (x + 30, y + 70),                      # coordinate
                            cv2.FONT_HERSHEY_SIMPLEX,              # font
                            1,                                     # fontsize
                            (255,255,255),                         # color
                            2)                                     # thickness

                # ethnicity labelling
                cv2.putText(frame, ethni_list[eth], 
                            (x + 30, y + 100),                     # coordinate
                            cv2.FONT_HERSHEY_SIMPLEX,              # font
                            1,                                     # fontsize
                            (255,0,255),                         # color
                            2)                                     # thickness

                cv2.rectangle(frame, (x - 15, y - 15), (x + w + 15, y + h + 15), color, 2)


        # image display
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    # release the camera and destroy the window
    cap.release()
    cv2.destroyAllWindows() 
    
if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("Face recognition", 0)

