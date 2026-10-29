import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))
import numpy as np, scipy.signal as ss
from scipy.io import wavfile
import json
sr,x=wavfile.read(os.path.join(DATA, 'audio.wav')); x=x.astype(np.float32)/32768
sos=ss.butter(4,[2000,8000],btype='band',fs=sr,output='sos')
y=ss.sosfiltfilt(sos,x)
hop=int(sr*0.005); n=len(y)//hop
e=np.sqrt(np.mean(y[:n*hop].reshape(n,hop)**2,axis=1))+1e-6
db=20*np.log10(e)
# onset strength: rise over previous 30ms minimum
prev=np.array([db[max(0,i-6):i].min() if i>0 else db[0] for i in range(n)])
rise=db-prev
# adaptive baseline: rolling median of db over 2 s
from scipy.ndimage import median_filter
base=median_filter(db,size=401)
t=np.arange(n)*0.005
cand=[]
for i in range(1,n-1):
    if rise[i]>=10 and db[i]-base[i]>=14 and db[i]>=db[i-1] and db[i]>=db[i+1]:
        cand.append((t[i],float(db[i]-base[i]),float(rise[i])))
# min gap 0.18s keep strongest
hits=[]
for c in cand:
    if hits and c[0]-hits[-1][0]<0.18:
        if c[1]>hits[-1][1]: hits[-1]=c
    else: hits.append(c)
print(len(cand),len(hits))
for h in hits: print(f"{h[0]:7.3f} {h[1]:5.1f} {h[2]:5.1f}")
json.dump([{'t':h[0],'snr':h[1],'rise':h[2]} for h in hits],open(os.path.join(DATA, 'audio_hits.json'),'w'))
