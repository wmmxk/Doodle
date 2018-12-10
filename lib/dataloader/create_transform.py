from .custom_transform import *



t1 = VerticalFlip()
t2 = HorizontalFlip()
t3 = Rotate()
#t4 = RandomCrop([720,730])
t5 = Shift()
t6 = ShiftScale()
t7 = ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, prob=1)
t8 = CLAHE()
t9 = RandomBrightness(limit=0.14)
t10 = RandomContrast(limit=0.14)
t11 = Crop(512)




t8 = RandomCrop([512,512])


ts = [t1,t2,t3,t5,t6,t7,t8,t9,t10,t11]
#the last transform is to Crop the size

train_transform = MultiCompose(ts,total = 1) 


