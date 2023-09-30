from ultralytics import YOLO
#from heic2png import HEIC2PNG

model = YOLO("best.pt")

#png = HEIC2PNG("IMG_2160.HEIC")
#png.save()
#print(png)

results = model(source = 0, show = True, max_det = 1, save = False, conf = 0.5) #source = 0 means webcam, show what the ai is doing (need this on for webcam), max_det is number of boxes, save is save the detection outputs, conf is minimum conf to show a box