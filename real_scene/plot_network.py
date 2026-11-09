# -*- coding: utf-8 -*-
import json, os, math, heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus']=False

D=os.path.dirname(os.path.abspath(__file__))
inst=json.load(open(os.path.join(D,'luoyang_stsp_20node.json')))
nodes=inst['nodes']; edges=inst['undirected_edges']
ways=json.load(open(os.path.join(D,'_osm_cache.json')))

lat0=sum(n['lat'] for n in nodes)/len(nodes); lon0=sum(n['lon'] for n in nodes)/len(nodes)
def XY(la,lo): return ((lo-lon0)*111320*math.cos(math.radians(lat0)), (la-lat0)*111320)
pos={n['id']:XY(n['lat'],n['lon']) for n in nodes}

# restriction color
WIN_STYLE={
 (7.0,9.0):('#e41a1c','-','早高峰 07:00–09:00'),
 (17.5,19.5):('#377eb8','-','晚高峰 17:30–19:30'),
 (7.5,8.5):('#984ea3','--','护学 07:30–08:30'),
 (6.0,22.0):('#111111',(0,(6,3)),'白天禁行 06:00–22:00'),
}
def estyle(e):
    for (a,b),(c,ls,_) in WIN_STYLE.items():
        if abs((e.get('a') or -1)-a)<1e-6 and abs((e.get('b') or -1)-b)<1e-6: return c,ls,3.0
    return '#c2c7cc','-',1.8
# need windows on undirected edges: pull from arcs by eid
win={}
for a in inst['arcs']:
    if a['restricted']: win[a['eid']]=(a['a'],a['b'],a['window'])
for e in edges:
    if e['eid'] in win: e['a'],e['b'],e['window']=win[e['eid']]
    else: e['a']=e['b']=e['window']=None

# ---------- Figure 1: geographic network ----------
fig,ax=plt.subplots(figsize=(14,11),dpi=160)
xs=[p[0] for p in pos.values()];ys=[p[1] for p in pos.values()]
xmin,xmax,ymin,ymax=min(xs),max(xs),min(ys),max(ys)
mg=140
# faint OSM basemap roads (cropped)
for wy in ways:
    gs=wy['geometry']
    if not any((xmin-mg)<=XY(g['lat'],g['lon'])[0]<=(xmax+mg) and (ymin-mg)<=XY(g['lat'],g['lon'])[1]<=(ymax+mg) for g in gs):continue
    xs2=[];ys2=[]
    for g in gs:
        x,y=XY(g['lat'],g['lon']);xs2.append(x);ys2.append(y)
    ax.plot(xs2,ys2,color='#e9ebed',lw=0.8,zorder=1)
# edges
def perp(u,v,d=55):
    x1,y1=pos[u];x2,y2=pos[v]
    dx,dy=x2-x1,y2-y1;L=math.hypot(dx,dy) or 1
    return -dy/L*d,dx/L*d
for e in edges:
    x1,y1=pos[e['u']];x2,y2=pos[e['v']]
    c,ls,lw=estyle(e)
    ax.plot([x1,x2],[y1,y2],color=c,lw=lw,ls=ls,zorder=3,solid_capstyle='round')
    if e['window']:
        ox,oy=perp(e['u'],e['v'],45)
        horiz=abs(x2-x1)>abs(y2-y1)
        ax.text((x1+x2)/2+ (0 if horiz else ox),(y1+y2)/2+(oy if horiz else 0),
                e['window'].replace('[','').replace(']',''),fontsize=7,color=c,
                ha='center',va='center',zorder=6,
                bbox=dict(boxstyle='round,pad=0.15',fc='white',ec=c,lw=0.7,alpha=0.95))
# nodes
short={2:'丹尼斯',4:'河科大二附院',7:'市中心医院',8:'新都汇/王府井',10:'中央百货',18:'大张超市'}
for n in nodes:
    x,y=pos[n['id']]
    if n['kind']=='depot':
        ax.scatter([x],[y],s=420,marker='s',fc='#d62728',ec='black',zorder=7)
        ax.text(x,y+95,'0 车场\n副食品批发市场',ha='center',va='bottom',fontsize=9,fontweight='bold',color='#d62728')
    elif n['kind']=='delivery':
        ax.scatter([x],[y],s=240,marker='^',fc='#ff7f0e',ec='black',zorder=7)
        ax.text(x,y-85,f"{n['id']} {short[n['id']]}",ha='center',va='top',fontsize=8.5,fontweight='bold',color='#b35a00')
    else:
        ax.scatter([x],[y],s=130,marker='o',fc='#ecf0f1',ec='#5d6d7e',zorder=6)
        ax.text(x,y,f"{n['id']}",ha='center',va='center',fontsize=8,color='#2c3e50',zorder=7)
ax.set_aspect('equal');ax.axis('off')
ax.set_xlim(xmin-mg,xmax+mg);ax.set_ylim(ymin-mg,ymax+mg)
ax.set_title('洛阳市西工区真实道路网络 STSP-TVRAR 实例（20节点：1车场+6配送点+13中转路口）\n'
             '节点坐标/路段长度取自 OpenStreetMap；限行时间窗依据洛阳市货运管控公开规定',fontsize=13)
legend_e=[Line2D([0],[0],color=c,lw=3,ls=ls,label=lab) for (_,_),(c,ls,lab) in
          {k:WIN_STYLE[k] for k in [(7.0,9.0),(17.5,19.5),(7.5,8.5),(6.0,22.0)]}.items()]
legend_n=[Line2D([0],[0],marker='s',color='w',markerfacecolor='#d62728',markersize=13,label='车场 Depot'),
          Line2D([0],[0],marker='^',color='w',markerfacecolor='#ff7f0e',markersize=13,label='配送点 Delivery(必访)'),
          Line2D([0],[0],marker='o',color='w',markerfacecolor='#ecf0f1',markeredgecolor='#5d6d7e',markersize=11,label='中转路口 Steiner(可经过)'),
          Line2D([0],[0],color='#c2c7cc',lw=2,label='普通路段(不限行)')]
ax.legend(handles=legend_n+legend_e,loc='lower left',fontsize=9,framealpha=0.95,ncol=1)
ax.text(0.99,0.01,'底图：© OpenStreetMap contributors',transform=ax.transAxes,ha='right',fontsize=7,color='#999')
plt.tight_layout()
plt.savefig(os.path.join(D,'fig1_network_map.png'),bbox_inches='tight')
print("saved fig1_network_map.png")

# ---------- Figure 2: time-dependent route demo ----------
out={i:[] for i in range(len(nodes))}
for a in inst['arcs']: out[a['u']].append(a)
def td(start,t0,tgt):
    dist={start:t0};prev={};pq=[(t0,start)]
    while pq:
        t,u=heapq.heappop(pq)
        if t!=dist[u]:continue
        if u==tgt:break
        for a in out[u]:
            if a['a'] is not None and a['a']<=t<a['b']:continue
            nt=t+nodes[u]['service_min']/60+a['time_min']/60
            if nt<dist.get(a['v'],1e18):dist[a['v']]=nt;prev[a['v']]=(u,a);heapq.heappush(pq,(nt,a['v']))
    seq=[tgt]
    while tgt!=start: tgt=prev[tgt][0];seq.append(tgt)
    return list(reversed(seq)),dist
fig2,axes=plt.subplots(1,2,figsize=(20,9),dpi=150)
for ax,t0,title in zip(axes,[8.0,12.0],['出发 08:00（早高峰：定鼎大道/中州中路部分路段封闭）','出发 12:00（平峰：限行解除）']):
    for wy in ways:
        xs=[];ys=[]
        for g in wy['geometry']:
            x,y=XY(g['lat'],g['lon']);xs.append(x);ys.append(y)
        ax.plot(xs,ys,color='#eef0f2',lw=0.7,zorder=1)
    for e in edges:
        x1,y1=pos[e['u']];x2,y2=pos[e['v']]
        c,ls,lw=estyle(e)
        ax.plot([x1,x2],[y1,y2],color='#d6dbde',lw=1.3,zorder=2)
    path,dd=td(0,t0,18)
    pe=set(zip(path,path[1:]))|set(zip(path[1:],path))
    for e in edges:
        if (e['u'],e['v']) in pe:
            x1,y1=pos[e['u']];x2,y2=pos[e['v']]
            ax.plot([x1,x2],[y1,y2],color='#d62728',lw=4,zorder=5)
    for n in nodes:
        x,y=pos[n['id']];on = n['id'] in path
        if n['kind']=='depot':
            ax.scatter([x],[y],s=300,marker='s',fc='#d62728',ec='black',zorder=7)
        elif n['kind']=='delivery':
            ax.scatter([x],[y],s=170,marker='^',fc='#ff7f0e' if not on else '#2ca02c',ec='black',zorder=7)
        else:
            ax.scatter([x],[y],s=90,marker='o',fc='#ffffff' if not on else '#ffd2d2',ec='#888',zorder=6)
        ax.text(x,y,f"{n['id']}",ha='center',va='center',fontsize=7,zorder=8)
    ax.set_aspect('equal');ax.axis('off')
    ax.set_xlim(xmin-mg,xmax+mg);ax.set_ylim(ymin-mg,ymax+mg)
    ax.set_title(f"{title}\n车场→节点18(大张超市) 最短路：{path}  到达{dd[18]:.2f}时",fontsize=11)
fig2.suptitle('时间依赖演示：同一对起终点，不同出发时刻最短路不同（静态APSP约简失效）',fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(D,'fig2_timedep_demo.png'),bbox_inches='tight')
print("saved fig2_timedep_demo.png")
