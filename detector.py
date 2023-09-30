from ultralytics import YOLO
from heic2png import HEIC2PNG

model = YOLO("best.pt")

#png = HEIC2PNG("IMG_2160.HEIC")
#png.save()

#print(png)

results = model(source = 0, show = True, max_det = 1, save = False, conf = 0.5) #source = 0 means webcam