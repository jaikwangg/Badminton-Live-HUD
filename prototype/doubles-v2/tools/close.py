import cv2,numpy as np,json,sys
a,b,step,out=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
S=json.load(open('shuttle.json')); spos={}
for pa in S['paths']:
    for f,x,y in pa['pts']: spos[int(f)]=(x,y)
cap=cv2.VideoCapture('a540.mp4'); tiles=[]
for t in np.arange(a,b,step):
    f=int(round(t*30)); cap.set(1,f); ok,im=cap.read()
    for g in range(f-4,f+1):
        if g in spos: cv2.circle(im,(int(spos[g][0]),int(spos[g][1])),9 if g==f else 3,(0,0,255),2 if g==f else -1)
    c=im[200:900,0:540]; c=cv2.resize(c,(324,420)); cv2.putText(c,f'{t:.2f}',(5,22),0,0.7,(0,255,255),2); tiles.append(c)
while len(tiles)%6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+6]) for k in range(0,len(tiles),6)]))
