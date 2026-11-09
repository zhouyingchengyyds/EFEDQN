# -*- coding: utf-8 -*-
"""
Construct a REAL road-network instance for STSP-TVRAR.
City: Luoyang (Henan), Xigong central district.
Topology / coordinates / road lengths come from OpenStreetMap (real).
Restriction time windows follow Luoyang public freight-control policy.
"""
import sys, math, json
from collections import defaultdict, deque
sys.path.insert(0, '/tmp')
from osm import overpass

S, W, N, E = 34.640, 112.410, 34.695, 112.460
import os
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_osm_cache.json')
if os.path.exists(CACHE):
    WAYS = json.load(open(CACHE))
else:
    q = f'[out:json][timeout:120];(way["highway"]["name"]({S},{W},{N},{E}););out geom;'
    WAYS = overpass(q)['elements']
    json.dump(WAYS, open(CACHE, 'w'), ensure_ascii=False)

def hav(a, b):
    R = 6371000.0
    lat1, lon1 = map(math.radians, a); lat2, lon2 = map(math.radians, b)
    dlat = lat2-lat1; dlon = lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

import heapq
ROADS = defaultdict(list); NAMES_AT = defaultdict(set)
# global graph over ALL ways (vertices keyed by rounded coord), with names per vertex
GADJ = defaultdict(dict); GCOORD = {}; ONROAD = defaultdict(set)
def rk(p): return (round(p[0], 7), round(p[1], 7))
for wy in WAYS:
    nm = wy['tags']['name']
    gs = [(g['lat'], g['lon']) for g in wy['geometry']]
    ROADS[nm].append(gs)
    for p in gs: NAMES_AT[p].add(nm)
    for a, b in zip(gs, gs[1:]):
        ka, kb = rk(a), rk(b)
        GCOORD[ka] = a; GCOORD[kb] = b
        d = hav(a, b)
        GADJ[ka][kb] = min(GADJ[ka].get(kb, 1e18), d)
        GADJ[kb][ka] = min(GADJ[kb].get(ka, 1e18), d)
        ONROAD[ka].add(nm); ONROAD[kb].add(nm)

def cross(*road_lists):
    """coordinate of the at-grade intersection where one road from each list shares a node"""
    for p, ms in NAMES_AT.items():
        if all(any(r in ms for r in rl) for rl in road_lists):
            return p
    return None

def corridor_len(names, pA, pB, snap=40.0):
    """shortest along-road length staying on vertices belonging to one of `names`;
    multi-source Dijkstra from any vertex near pA to any near pB."""
    names = set(names)
    def near_set(p):
        out = []
        for k, c in GCOORD.items():
            if names & ONROAD[k] and hav(p, c) <= snap:
                out.append((hav(p, c), k))
        return [k for _, k in sorted(out)]
    SA, SB = near_set(pA), near_set(pB)
    if not SA or not SB: return None
    sb = set(SB)
    dist = {k: 0.0 for k in SA}; pq = [(0.0, k) for k in SA]
    while pq:
        d, x = heapq.heappop(pq)
        if d != dist.get(x): continue
        if x in sb: return d
        for y, w in GADJ[x].items():
            if not (names & ONROAD[y]): continue
            nd = d + w
            if nd < dist.get(y, 1e18):
                dist[y] = nd; heapq.heappush(pq, (nd, y))
    return None

def seg_len(nm, pA, pB):
    return corridor_len([nm], pA, pB)

def _local(p, lat0, lon0):
    return ((p[1]-lon0)*111320.0*math.cos(math.radians(lat0)), (p[0]-lat0)*111320.0)
def _pt_seg_dist(px,py,ax,ay,bx,by):
    dx,dy=bx-ax,by-ay
    L2=dx*dx+dy*dy or 1e-9
    t=max(0,min(1,((px-ax)*dx+(py-ay)*dy)/L2))
    qx,qy=ax+t*dx,ay+t*dy
    return math.hypot(px-qx,py-qy)

def tube_len(pA, pB, tube=None, snap=60.0, getpath=False):
    """real local drivable length between two intersections: shortest path that
    stays within `tube` metres of the straight segment (prevents long detours /
    cutting across other grid nodes). Tube auto-scales for genuinely bending
    arterials. Endpoint access legs (node->nearest road vertex) are counted."""
    d0 = hav(pA, pB)
    if tube is None:
        tube = max(120.0, 0.6*d0)
    lat0=(pA[0]+pB[0])/2; lon0=(pA[1]+pB[1])/2
    ax,ay=_local(pA,lat0,lon0); bx,by=_local(pB,lat0,lon0)
    def in_tube(c):
        x,y=_local(c,lat0,lon0)
        return _pt_seg_dist(x,y,ax,ay,bx,by) <= tube
    SA={k:hav(pA,c) for k,c in GCOORD.items() if hav(pA,c)<=snap and in_tube(c)}
    SB={k:hav(pB,c) for k,c in GCOORD.items() if hav(pB,c)<=snap and in_tube(c)}
    if not SA or not SB: return None
    dist=dict(SA); pq=[(d,k) for k,d in SA.items()]; prev={}
    while pq:
        d,x=heapq.heappop(pq)
        if d!=dist.get(x):continue
        if x in SB:
            total=d+SB[x]
            if getpath:
                path=[];y=x
                while y is not None:path.append(y);y=prev.get(y)
                return total,path
            return total
        for y,w in GADJ[x].items():
            if not in_tube(GCOORD[y]):continue
            nd=d+w
            if nd<dist.get(y,1e18):
                dist[y]=nd;prev[y]=x;heapq.heappush(pq,(nd,y))
    return None

# 20 nodes (real intersections / POI)
# row roads
R1w, R1e = '唐宫西路', '唐宫中路'
R2 = '中州中路'; R3w, R3e = '凯旋西路', '凯旋东路'; R4 = '九都路'; R0 = '汉屯路'
# col roads
C1 = '芳林路'; C2 = '解放路'; C3 = '八一路'; C4n, C4a, C4b = '金谷园路', '人民西路', '体育场路'; C5a, C5b = '定鼎大道', '嘉豫门大街'

raw = {
 ('R1','C1'): cross([R1w],[C1]),
 ('R1','C2'): cross([R1w],[C2]),
 ('R1','C3'): cross([R1w],[C3]),
 ('R1','C4'): cross([R1w,R1e],[C4a]),
 ('R1','C5'): cross([R1e],[C5a]),
 ('R2','C1'): cross([R2],[C1]),
 ('R2','C2'): cross([R2],[C2]),
 ('R2','C3'): cross([R2],[C3]),
 ('R2','C4'): cross([R2],[C4a,'人民东路']),
 ('R2','C5'): cross([R2],[C5a]),
 ('R3','C1'): cross([R3w],[C1]),
 ('R3','C2'): cross([R3w],[C2]),
 ('R3','C3'): cross([R3w],[C3]),
 ('R3','C4'): cross([R3w,R3e],[C4a,C4b]),
 ('R3','C5'): cross([R3e],[C5a]),
 ('R4','C1'): cross([R4],[C1]),
 ('R4','C4'): cross([R4],[C4b]),
 ('R4','C5'): cross([R4],[C5b,C5a]),
 ('R0','C2'): cross([R0],[C2]),
}
DEPOT = (34.68638, 112.45076)  # grocery wholesale market (OSM marketplace)
assert all(v is not None for v in raw.values()), {k:v for k,v in raw.items() if v is None}

# stable id ordering
order = [('R0','C2'),
         ('R1','C1'),('R1','C2'),('R1','C3'),('R1','C4'),('R1','C5'),
         ('R2','C1'),('R2','C2'),('R2','C3'),('R2','C4'),('R2','C5'),
         ('R3','C1'),('R3','C2'),('R3','C3'),('R3','C4'),('R3','C5'),
         ('R4','C1'),('R4','C4'),('R4','C5')]
ID = {}
nodes = []
nid = 0
# depot = node 0
ID['DEPOT'] = 0
nodes.append({'id':0,'key':'DEPOT','name':'副食品批发市场(车场)','kind':'depot',
              'lat':DEPOT[0],'lon':DEPOT[1]})
nid = 1
name_map = {
 ('R1','C1'):'唐宫西路×芳林路口', ('R1','C2'):'唐宫西路×解放路口', ('R1','C3'):'唐宫西路×八一路口',
 ('R1','C4'):'唐宫路×人民西路口', ('R1','C5'):'唐宫中路×定鼎大道口',
 ('R2','C1'):'中州中路×芳林路口', ('R2','C2'):'中州中路×解放路口', ('R2','C3'):'中州中路×八一路口',
 ('R2','C4'):'中州中路×人民西路口', ('R2','C5'):'中州中路×定鼎大道口',
 ('R3','C1'):'凯旋西路×芳林路口', ('R3','C2'):'凯旋西路×解放路口', ('R3','C3'):'凯旋西路×八一路口',
 ('R3','C4'):'凯旋路×体育场路口', ('R3','C5'):'凯旋东路×定鼎大道口',
 ('R4','C1'):'九都路×芳林路口(平交)', ('R4','C4'):'九都路×体育场路口(平交)', ('R4','C5'):'九都路×嘉豫门大街口(平交)',
 ('R0','C2'):'汉屯路×解放路口',
}
# delivery (real adjacent POI)
delivery = {
 ('R1','C1'):'丹尼斯超市(唐宫路店)',
 ('R2','C1'):'洛阳市中心医院',
 ('R2','C2'):'新都汇/王府井商圈',
 ('R2','C4'):'中央百货大楼',
 ('R1','C3'):'河南科技大学第二附属医院',
 ('R4','C4'):'大张超市(七一路店)',
}
for k in order:
    ID[k] = nid
    la, lo = raw[k]
    kind = 'delivery' if k in delivery else 'transit'
    nm = delivery.get(k, name_map[k])
    nodes.append({'id':nid,'key':f'{k[0]}-{k[1]}','name':nm,'intersection':name_map[k],
                  'kind':kind,'lat':la,'lon':lo})
    nid += 1
assert nid == 20

# Edges (follow real roads)
# each entry: (u_key, v_key, corridor(s) as (road,from,to) for length, road_class)
def P(k): return raw[k]
edge_defs = [
 # vertical C1 Fanglin Road
 (('R1','C1'),('R2','C1'),[(C1,P(('R1','C1')),P(('R2','C1')))],'次干路'),
 (('R2','C1'),('R3','C1'),[(C1,P(('R2','C1')),P(('R3','C1')))],'次干路'),
 (('R3','C1'),('R4','C1'),[(C1,P(('R3','C1')),P(('R4','C1')))],'次干路'),
 # vertical C2 Jiefang Road
 (('R0','C2'),('R1','C2'),[(C2,P(('R0','C2')),P(('R1','C2')))],'主干路'),
 (('R1','C2'),('R2','C2'),[(C2,P(('R1','C2')),P(('R2','C2')))],'主干路'),
 (('R2','C2'),('R3','C2'),[(C2,P(('R2','C2')),P(('R3','C2')))],'主干路'),
 # vertical C3 Bayi Road
 (('R1','C3'),('R2','C3'),[(C3,P(('R1','C3')),P(('R2','C3')))],'支路'),
 (('R2','C3'),('R3','C3'),[(C3,P(('R2','C3')),P(('R3','C3')))],'支路'),
 # vertical C4 Renmin West Road / Renmin East Road (split naming) / Tiyuchang Road
 (('R1','C4'),('R2','C4'),[('人民西路/人民东路',P(('R1','C4')),P(('R2','C4')))],'次干路'),
 (('R2','C4'),('R3','C4'),[('人民西路/人民东路',P(('R2','C4')),P(('R3','C4')))],'次干路'),
 (('R3','C4'),('R4','C4'),[(C4b,P(('R3','C4')),P(('R4','C4')))],'次干路'),
 # vertical C5 Dingding Avenue / Jiayumen Street
 (('R1','C5'),('R2','C5'),[(C5a,P(('R1','C5')),P(('R2','C5')))],'主干路'),
 (('R2','C5'),('R3','C5'),[(C5a,P(('R2','C5')),P(('R3','C5')))],'主干路'),
 (('R3','C5'),('R4','C5'),[(C5b,P(('R3','C5')),P(('R4','C5')))],'主干路'),
 # horizontal R1 Tanggong (the west/east naming switches near Renmin West Road)
 (('R1','C1'),('R1','C2'),[(R1w,P(('R1','C1')),P(('R1','C2')))],'主干路'),
 (('R1','C2'),('R1','C3'),[(R1w,P(('R1','C2')),P(('R1','C3')))],'主干路'),
 (('R1','C3'),('R1','C4'),[(R1w,P(('R1','C3')),P(('R1','C4')))],'主干路'),
 (('R1','C4'),('R1','C5'),[('唐宫西路/唐宫中路',P(('R1','C4')),P(('R1','C5')))],'主干路'),
 # horizontal R2 Zhongzhou Middle Road
 (('R2','C1'),('R2','C2'),[(R2,P(('R2','C1')),P(('R2','C2')))],'主干路'),
 (('R2','C2'),('R2','C3'),[(R2,P(('R2','C2')),P(('R2','C3')))],'主干路'),
 (('R2','C3'),('R2','C4'),[(R2,P(('R2','C3')),P(('R2','C4')))],'主干路'),
 (('R2','C4'),('R2','C5'),[(R2,P(('R2','C4')),P(('R2','C5')))],'主干路'),
 # horizontal R3 Kaixuan (west/east boundary at C4)
 (('R3','C1'),('R3','C2'),[(R3w,P(('R3','C1')),P(('R3','C2')))],'主干路'),
 (('R3','C2'),('R3','C3'),[(R3w,P(('R3','C2')),P(('R3','C3')))],'主干路'),
 (('R3','C3'),('R3','C4'),[(R3w,P(('R3','C3')),P(('R3','C4')))],'主干路'),
 (('R3','C4'),('R3','C5'),[('凯旋西路/凯旋东路',P(('R3','C4')),P(('R3','C5')))],'主干路'),
 # horizontal R4 Jiudu Road (quasi-expressway, limited at-grade access)
 (('R4','C1'),('R4','C4'),[(R4,P(('R4','C1')),P(('R4','C4')))],'快速路'),
 (('R4','C4'),('R4','C5'),[(R4,P(('R4','C4')),P(('R4','C5')))],'快速路'),
 # depot link along Dingding Avenue
 ('DEPOT',('R1','C5'),[(C5a,DEPOT,P(('R1','C5')))],'主干路'),
]

SPEED = {'快速路':40,'主干路':30,'次干路':25,'支路':20}  # km/h, urban running speed
edges = []
EDGE_PATHS = {}
for i,(uk,vk,cors,cls) in enumerate(edge_defs, start=1):
    pA, pB = cors[0][1], cors[-1][2]
    roads = '、'.join(sorted({str(n).replace('/','/') for n,_,_ in cors}))
    snap_e = 330.0 if (uk=='DEPOT' or vk=='DEPOT') else 60.0
    res = tube_len(pA, pB, snap=snap_e, getpath=True)
    assert res is not None, (uk,vk,'no tube path')
    L, path = res
    assert L and L > 30, (uk,vk,L)
    EDGE_PATHS[i] = path
    kmh = SPEED[cls]
    tmin = L/1000.0/kmh*60.0
    edges.append({'eid':i,'u':ID[uk] if uk!='DEPOT' else 0,'v':ID[vk],
                  'u_key':str(uk),'v_key':str(vk),'roads':roads,'road_class':cls,
                  'length_m':round(L,1),'time_min':round(tmin,2),'cost':round(L/100.0,2)})

json.dump({'nodes':nodes,'edges':edges}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'graph_raw.json'),'w'), ensure_ascii=False, indent=1)
print(f"nodes={len(nodes)}  undirected edges={len(edges)}  directed arcs={2*len(edges)}")
print(f"total road length = {sum(e['length_m'] for e in edges):.0f} m")

# Topological check: an edge path must not pass through another selected node
node_coord={nd['id']:(nd['lat'],nd['lon']) for nd in nodes}
for e in edges:
    interior=EDGE_PATHS[e['eid']][1:-1]
    for vid in interior:
        c=GCOORD[vid]
        for oid in (e['u'],e['v']): continue
        for nid2,oc in node_coord.items():
            if nid2 in (e['u'],e['v']): continue
            if hav(c,oc) <= 25:
                print("  [warn] edge",e['eid'],"passes near node",nid2,round(hav(c,oc),1),"m")
print("topology check done")

# Restricted time windows (Luoyang public freight-control policy)
# decimal hours; each edge carries AT MOST ONE window; departure from tail forbidden inside [a,b]
# policy basis:
#  AM peak 07:00-09:00 / PM peak 17:30-19:30 freight ban (Luoyang PSB public replies)
#  fuel urban-distribution van daytime ban 06:00-22:00 inside control area
#  named control arterials: Zhongzhou Road / Kaixuan Road / Jiefang Road / Dingding Avenue; school-zone escort window
RESTR = {
 19:(7.0,9.0,'早高峰货车禁行(中州中路商业走廊)'),
 21:(7.0,9.0,'早高峰货车禁行(中州中路)'),
 23:(17.5,19.5,'晚高峰货车禁行(凯旋西路)'),
 25:(17.5,19.5,'晚高峰货车禁行(凯旋路)'),
 5 :(7.0,9.0,'早高峰货车禁行(解放路)'),
 10:(7.5,8.5,'校园周边护学时段(人民西路·唐宫西路小学)'),
 13:(7.0,9.0,'早高峰货车禁行(定鼎大道)'),
 27:(6.0,22.0,'燃油配送车白天禁入(九都路准快速路)'),
}
def hm(x):
    h=int(x); m=int(round((x-h)*60)); return f"{h:02d}:{m:02d}"

# Directed arcs (bidirectional; identical base attributes for a symmetric visual instance)
arcs=[]
aid=0
for e in edges:
    w=RESTR.get(e['eid'])
    for (uu,vv,uk,vk) in [(e['u'],e['v'],e['u_key'],e['v_key']),(e['v'],e['u'],e['v_key'],e['u_key'])]:
        aid+=1
        arcs.append({'arc_id':aid,'eid':e['eid'],'u':uu,'v':vv,'u_key':uk,'v_key':vk,
                     'roads':e['roads'],'road_class':e['road_class'],
                     'length_m':e['length_m'],'time_min':e['time_min'],'cost':e['cost'],
                     'restricted': w is not None,
                     'a': w[0] if w else None,'b': w[1] if w else None,
                     'window': f"[{hm(w[0])},{hm(w[1])}]" if w else None,
                     'restriction_type': w[2] if w else None})

# Service time (delivery nodes only, deterministic)
SERVICE_MIN = 5.0
for nd in nodes:
    nd['service_min'] = SERVICE_MIN if nd['kind']=='delivery' else 0.0

instance = {
 'meta':{
   'problem':'STSP-TVRAR (Steiner TSP with Time-Varying Road Access Restrictions)',
   'city':'河南省洛阳市西工区中心片区(唐宫路—九都路 / 芳林路—定鼎大道)',
   'depot':'洛阳副食品批发市场(定鼎大道·玄武门大街口)',
   'crs':'WGS84 lat/lon (OpenStreetMap)',
   'n_nodes':len(nodes),'n_delivery':sum(n['kind']=='delivery' for n in nodes),
   'n_transit':sum(n['kind']=='transit' for n in nodes),
   'n_undirected_edges':len(edges),'n_directed_arcs':len(arcs),
   'n_restricted_edges':len(RESTR),'rule':'每条边至多1个限行窗; 车辆在尾节点的出发时刻落入[a,b]则该边禁止驶入; 全程不允许等待',
   'edge_cost':'cost = length_m/100 (可按目标函数整体缩放)',
   'travel_time':'按道路等级的城市运行速度: 快速路40/主干路30/次干路25/支路20 km/h',
   'service_time_min':SERVICE_MIN,
   'time_unit':'小时(24h制), 另附 HH:MM',
 },
 'sources':[
   '路网几何/路口/路段长度/POI: OpenStreetMap (© OpenStreetMap contributors), Overpass 实时导出',
   '限行时段: 洛阳市公安局公开答复——货车早高峰7:00-9:00、晚高峰17:30-19:30禁行 (洛阳网百姓呼声)',
   '燃油城市配送车每日6:00-22:00禁入限行区、22:00-次日6:00通行 (腾讯新闻/洛阳通告2021)',
   '中州路、凯旋路、人民路、定鼎大道等为洛阳具名货运管控道路 (河南省政府/洛阳日报通告)'
 ],
 'nodes':nodes,
 'arcs':arcs,
 'undirected_edges':edges,
}
OUTDIR=os.path.dirname(os.path.abspath(__file__))
json.dump(instance, open(os.path.join(OUTDIR,'luoyang_stsp_20node.json'),'w'), ensure_ascii=False, indent=1)

# CSV outputs
import csv
with open(os.path.join(OUTDIR,'nodes.csv'),'w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f); w.writerow(['id','key','kind','name','intersection','lat','lon','service_min'])
    for nd in nodes: w.writerow([nd['id'],nd['key'],nd['kind'],nd['name'],nd.get('intersection',''),nd['lat'],nd['lon'],nd['service_min']])
with open(os.path.join(OUTDIR,'arcs.csv'),'w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f); w.writerow(['arc_id','eid','u','v','roads','class','length_m','time_min','cost','restricted','a','b','window','type'])
    for a in arcs: w.writerow([a['arc_id'],a['eid'],a['u'],a['v'],a['roads'],a['road_class'],a['length_m'],a['time_min'],a['cost'],a['restricted'],a['a'],a['b'],a['window'],a['restriction_type']])

# adjacency list (out-neighbours)
adj={i:[] for i in range(len(nodes))}
for a in arcs: adj[a['u']].append({'to':a['v'],'time_min':a['time_min'],'cost':a['cost'],'window':a['window'],'eid':a['eid']})
json.dump(adj, open(os.path.join(OUTDIR,'adjacency.json'),'w'), ensure_ascii=False, indent=1)
print("restricted edges:",sorted(RESTR)," total arcs:",len(arcs))
print("WROTE luoyang_stsp_20node.json / nodes.csv / arcs.csv / adjacency.json")
for e in edges:
    tag=f"  RESTR {hm(RESTR[e['eid']][0])}-{hm(RESTR[e['eid']][1])}" if e['eid'] in RESTR else ""
    print(f"e{e['eid']:02d} {e['u']:2d}->{e['v']:2d} {e['length_m']:7.1f}m {e['time_min']:5.2f}min {e['road_class']}{tag}")
