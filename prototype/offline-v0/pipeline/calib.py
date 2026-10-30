import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))
import cv2, numpy as np, json
img = np.float32([
 [406.1,712.2],[1563.2,700.6],[1356.5,594.7],[621.6,599.1],   # doubles x short service
 [975.5,706.0],[985.0,597.0],                                  # center line x short service
 [1474.0,701.8],[493.9,711.1],[1299.3,595.3],[678.6,599.0]])   # singles x short service
crt = np.float32([
 [0,4.72],[6.10,4.72],[6.10,8.68],[0,8.68],
 [3.05,4.72],[3.05,8.68],
 [5.64,4.72],[0.46,4.72],[5.64,8.68],[0.46,8.68]])
H,_ = cv2.findHomography(img, crt, 0)
res = cv2.perspectiveTransform(img.reshape(-1,1,2),H).reshape(-1,2)-crt
print('residual m:',np.round(np.abs(res).max(axis=1),3))
chk = {'center line (972.8,904.7)':(972.8,904.7),'left post base (530,643)':(530,643),'right post base (1445,632)':(1445,632),'bottom-center (960,1077)':(960,1077),'bottom-left (0,1077)':(0,1077)}
for k,(x,y) in chk.items():
    p=cv2.perspectiveTransform(np.float32([[[x,y]]]),H)[0,0]; print(k,'->',np.round(p,2))
Hi=np.linalg.inv(H)
for name,(X,Y) in {'near baseline L':(0,0),'near baseline R':(6.1,0),'near long svc L':(0,0.76),'far baseline L':(0,13.4),'far baseline R':(6.1,13.4),'net L':(0,6.7),'net R':(6.1,6.7)}.items():
    p=cv2.perspectiveTransform(np.float32([[[X,Y]]]),Hi)[0,0]; print(name,np.round(p,1))
json.dump({'H':H.tolist(),'img_size':[1920,1078]},open(os.path.join(DATA, 'calib.json'),'w'))
