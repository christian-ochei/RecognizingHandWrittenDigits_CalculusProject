import cv2
import keyboard
import numpy as np
import mouse
import torch
import torchvision

from model import *

# mnist_model = model.MNISTModel()
mnist_model = torch.load('MNISTModel6.pt')

print('Press and Hold D to Draw')
print('Press and Hold E to Erase')
print('Press and Hold Enter to Check')


mat = np.zeros((28,28),dtype=np.uint8)
cv2.namedWindow('Draw a digit')


transform = torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize(
               (0.1307,), (0.3081,))
       ])

while True:
    cv2.imshow('Draw a digit',cv2.resize(mat,(400,400),cv2.INTER_NEAREST))
    cv2.waitKey(5)

    if keyboard.is_pressed('d') or keyboard.is_pressed('E'):
        rect = cv2.getWindowImageRect('Draw a digit')
        pos = mouse.get_position()

        x = ((pos[0]-rect[0])/400)*28
        y = ((pos[1]-rect[1])/400)*28

        if 0 < x < 28 and 0 < y < 28:
            mat[int(y),int(x)] = 255 if keyboard.is_pressed('d') else 0

    if keyboard.is_pressed('enter'):
        while keyboard.is_pressed('enter'):
            ...
        print(f' This looks like the number {mnist_model.predict( transform(mat) )}')

