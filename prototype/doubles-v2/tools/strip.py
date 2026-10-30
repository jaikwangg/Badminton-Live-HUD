import json,cv2,numpy as np,sys
S=json.load(open('shuttle.json')); P=json.load(open('players.json')); C=json.load(open('cands.json'))
a,b,step,out=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
spos={}
for pa in S['paths']:
    for f,x,y in pa['pts']: spos[int(f)]=(x,y)
cap=cv2.VideoCapture('a540.mp4'); tiles=[]
COL=[(240,208,63),(160,227,75),(255,155,195),(107,138,255)]
for t in np.arange(a,b,step):
    f=int(round(t*30)); cap.set(1,f); ok,im=cap.read()
    for c in C[f]:
        if c[3]>=24: cv2.circle(im,(int(c[0]),int(c[1])),5,(0,255,255),1)
    for g in range(f-5,f+1):
        if g in spos: cv2.circle(im,(int(spos[g][0]),int(spos[g][1])),4 if g<f else 9,(0,0,255),-1 if g<f else 2)
    for j,p in enumerate(P['players']):
        bx=p['box'][f]
        if bx and bx[0] is not None: cv2.rectangle(im,(int(bx[0]),int(bx[1])),(int(bx[2]),int(bx[3])),COL[j],1)
    c=im[240:800]; c=cv2.resize(c,(300,311)); cv2.putText(c,f'{t:.2f}',(4,20),0,0.6,(0,255,255),2); tiles.append(c)
cols=int(sys.argv[5]) if len(sys.argv)>5 else 6
while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+cols]) for k in range(0,len(tiles),cols)]))
