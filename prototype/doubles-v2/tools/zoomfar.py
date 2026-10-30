import cv2,numpy as np,json,sys
a,b,step,out=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
y0=int(sys.argv[5]) if len(sys.argv)>5 else 340; y1=int(sys.argv[6]) if len(sys.argv)>6 else 620
S=json.load(open('shuttle.json')); spos={}
for pa in S['paths']:
    for f,x,y in pa['pts']: spos[int(f)]=(x,y)
cap=cv2.VideoCapture('a540.mp4'); tiles=[]
for t in np.arange(a,b,step):
    f=int(round(t*30)); cap.set(1,f); ok,im=cap.read()
    if f in spos: cv2.circle(im,(int(spos[f][0]),int(spos[f][1])),7,(0,0,255),1)
    c=cv2.resize(im[y0:y1,40:500],None,fx=0.9,fy=0.9); cv2.putText(c,f'{t:.2f}',(4,18),0,0.55,(0,255,255),2); tiles.append(c)
while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+4]) for k in range(0,len(tiles),4)]))
