from ultralytics import YOLO
#from heic2png import HEIC2PNG
#import cv2

model = YOLO("best.pt")

#png = HEIC2PNG("IMG_2160.HEIC")
#png.save()
#print(png)

#for i in range(0,100):
#    cap = cv2.VideoCapture(i)
#    test = cap.read()
#    print(i)
#    print(test)
    

results = model(source = 0, show = True, max_det = 1, conf = 0.1, save = False, verbose=True) #source = 0 means webcam, show what the ai is doing (need this on for webcam), max_det is number of boxes, save is save the detection outputs, conf is minimum conf to show a box
    
