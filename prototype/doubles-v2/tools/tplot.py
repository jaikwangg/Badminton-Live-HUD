import json,numpy as np,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
a,b,out=float(sys.argv[1]),float(sys.argv[2]),sys.argv[3]
C=json.load(open('cands.json')); rows=[json.loads(l) for l in open('dets2.jsonl')]; S=json.load(open('shuttle.json')); E=json.load(open('events.json')); P=json.load(open('players.json'))
def inbox(x,y,i):
    for d in rows[i]['d']:
        x1,y1,x2,y2=d['b']
        if x1-3<=x<=x2+3 and y1-3<=y<=y2+0.08*(y2-y1)+3: return True
    return False
T=[];X=[];Y=[]
for i in range(int(a*30),min(int(b*30),len(rows))):
    for c in C[i]:
        if c[3]>=24 and not inbox(c[0],c[1],i): T.append(i/30);X.append(c[0]);Y.append(c[1])
fig,ax=plt.subplots(2,1,figsize=(26,9),sharex=True)
ax[0].scatter(T,Y,s=5,c='0.5'); ax[1].scatter(T,X,s=5,c='0.5')
for pa in S['paths']:
    p=np.array(pa['pts']); m=(p[:,0]>=a*30)&(p[:,0]<=b*30)
    ax[0].plot(p[m,0]/30,p[m,2],'.',ms=4); ax[1].plot(p[m,0]/30,p[m,1],'.',ms=4)
cols=['c','g','m','r']
for j,pl in enumerate(P['players']):
    ts=np.arange(int(a*30),min(int(b*30),P['n']))
    yy=[pl['box'][t][1] if pl['box'][t] and pl['box'][t][0] is not None else np.nan for t in ts]
    xx=[(pl['box'][t][0]+pl['box'][t][2])/2 if pl['box'][t] and pl['box'][t][0] is not None else np.nan for t in ts]
    ax[0].plot(ts/30,yy,cols[j],lw=1,alpha=.6); ax[1].plot(ts/30,xx,cols[j],lw=1,alpha=.6,label=pl['name'] if False else str(j))
for o in json.load(open('onsets.json'))['onsets']:
    if a<=o['t']<=b:
        for x in ax: x.axvline(o['t'],color='r',alpha=.5)
for r in E['rallies']:
    for s in r['shots']:
        if a<=s['t']<=b:
            for x in ax: x.axvline(s['t'],color=cols[s['player']],ls='--',lw=2)
ax[0].invert_yaxis(); ax[1].legend()
for x in ax: x.set_xticks(np.arange(a,b,0.25)); x.grid(alpha=.3)
plt.tight_layout(); plt.savefig(out,dpi=50)
