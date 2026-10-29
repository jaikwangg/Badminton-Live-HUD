import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))
import cv2, numpy as np
m=cv2.imread('out/median.png')
hsv=cv2.cvtColor(m,cv2.COLOR_BGR2HSV)
H,S,V=cv2.split(hsv)
mask=((V>232)&(S<18)).astype(np.uint8)*255
mask[:535]=0
cv2.imwrite('out/mask.png',mask)
lines=cv2.HoughLinesP(mask,1,np.pi/720,80,minLineLength=60,maxLineGap=25)
L=[]
for x1,y1,x2,y2 in lines.reshape(-1,4):
    L.append((x1,y1,x2,y2))
# represent each as rho/theta
def rt(l):
    x1,y1,x2,y2=l
    th=np.arctan2(y2-y1,x2-x1)
    if th<0: th+=np.pi
    if th>=np.pi: th-=np.pi
    n=np.array([-np.sin(th),np.cos(th)])
    rho=n@np.array([x1,y1])
    return th,rho,np.hypot(x2-x1,y2-y1)
items=[(rt(l),l) for l in L]
# cluster
clusters=[]
for (th,rho,ln),l in sorted(items,key=lambda t:-t[0][2]):
    for c in clusters:
        dth=abs(c['th']-th); dth=min(dth,np.pi-dth)
        if dth<np.radians(1.5) and abs(c['rho']-rho)<8:
            c['segs'].append(l); c['len']+=ln; break
    else:
        clusters.append(dict(th=th,rho=rho,segs=[l],len=ln))
clusters=[c for c in clusters if c['len']>150]
out=m.copy()
for i,c in enumerate(sorted(clusters,key=lambda c:-c['len'])):
    pts=np.array([[s[0],s[1]] for s in c['segs']]+[[s[2],s[3]] for s in c['segs']],dtype=np.float32)
    vx,vy,x0,y0=cv2.fitLine(pts,cv2.DIST_L2,0,0.01,0.01).ravel()
    c['fit']=(vx,vy,x0,y0)
    xs=pts[:,0];ys=pts[:,1]
    print(i, f"deg={np.degrees(c['th']):6.1f} len={c['len']:6.0f} x[{xs.min():.0f},{xs.max():.0f}] y[{ys.min():.0f},{ys.max():.0f}] dir=({vx:.3f},{vy:.3f}) pt=({x0:.1f},{y0:.1f})")
    col=tuple(int(v) for v in np.random.RandomState(i).randint(0,255,3))
    p1=(int(x0-vx*2000),int(y0-vy*2000));p2=(int(x0+vx*2000),int(y0+vy*2000))
    cv2.line(out,p1,p2,(0,0,255),1)
    cv2.putText(out,str(i),(int(xs.mean()),int(ys.mean())),0,0.8,(0,255,255),2)
cv2.imwrite('out/lines.png',out)
np.save('clusters.npy',np.array([[c['fit'][0],c['fit'][1],c['fit'][2],c['fit'][3]] for c in sorted(clusters,key=lambda c:-c['len'])]))
