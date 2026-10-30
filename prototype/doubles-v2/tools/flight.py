import json,cv2,numpy as np,sys
a,b,out=float(sys.argv[1]),float(sys.argv[2]),sys.argv[3]
C=json.load(open('cands.json')); rows=[json.loads(l) for l in open('dets2.jsonl')]
def inbox(x,y,i):
    for d in rows[i]['d']:
        x1,y1,x2,y2=d['b']; w,h=x2-x1,y2-y1
        if x1-3<=x<=x2+3 and y1-3<=y<=y2+0.08*h+3: return True
    return False
cap=cv2.VideoCapture('a540.mp4'); cap.set(1,int((a+b)/2*30)); ok,bg=cap.read()
bg=(bg*0.55).astype(np.uint8)
big=cv2.resize(bg,None,fx=2,fy=2)
import colorsys
for i in range(int(a*30),int(b*30)):
    u=(i/30-a)/(b-a); r,g,bb=colorsys.hsv_to_rgb(0.8*u,1,1); col=(int(bb*255),int(g*255),int(r*255))
    for c in C[i]:
        if c[3]>=24 and not inbox(c[0],c[1],i):
            cv2.circle(big,(int(c[0]*2),int(c[1]*2)),3,col,-1)
for k,t in enumerate(np.arange(a,b,0.5)):
    u=(t-a)/(b-a); r,g,bb=colorsys.hsv_to_rgb(0.8*u,1,1)
    cv2.putText(big,f'{t:.1f}',(10+k*70,30),0,0.6,(int(bb*255),int(g*255),int(r*255)),2)
cv2.imwrite(out,big[0:1400])
