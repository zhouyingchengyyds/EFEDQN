# -*- coding: utf-8 -*-
import json, os, heapq, itertools
D=os.path.dirname(os.path.abspath(__file__))
inst=json.load(open(os.path.join(D,'luoyang_stsp_20node.json')))
nodes=inst['nodes']; arcs=inst['arcs']
N=len(nodes)
out={i:[] for i in range(N)}
for a in arcs: out[a['u']].append(a)
name={n['id']:n['name'] for n in nodes}
deliveries=[n['id'] for n in nodes if n['kind']=='delivery']
depot=0

seen={0};stack=[0];und={i:set() for i in range(N)}
for a in arcs: und[a['u']].add(a['v'])
while stack:
    u=stack.pop()
    for v in und[u]:
        if v not in seen: seen.add(v);stack.append(v)
print("connected:",len(seen)==N, f"({len(seen)}/{N})")
print("delivery nodes:",deliveries)
for i in deliveries: print("   ",i,name[i])

def in_win(t,a,b): return (a is not None) and a<=t<b

def td_dijkstra(start,t0,target=None):
    dist={start:t0};prev={};pq=[(t0,start)]
    while pq:
        t,u=heapq.heappop(pq)
        if t!=dist[u]:continue
        if target is not None and u==target:break
        for a in out[u]:
            if in_win(t,a['a'],a['b']):continue
            nt=t+nodes[u]['service_min']/60.0+a['time_min']/60.0
            if nt<dist.get(a['v'],1e18):
                dist[a['v']]=nt;prev[a['v']]=(u,a);heapq.heappush(pq,(nt,a['v']))
    return dist,prev

def path_of(prev,s,g):
    if g!=s and g not in prev:return None
    seq=[g]
    while g!=s:
        pu,_=prev[g];seq.append(pu);g=pu
    return list(reversed(seq))

print("\n-- reachability at 4 departure times --")
for t0 in (6.5,8.0,12.0,18.0):
    dist,_=td_dijkstra(depot,t0)
    ok=all(d in dist for d in deliveries)
    print(f"depart {t0:05.2f}: all 6 reachable={ok}  arrivals=",
          {d:(round(dist[d],2) if d in dist else None) for d in deliveries})

print("\n-- depot->node18 shortest path by departure time --")
for t0 in (6.5,8.0,12.0,18.0):
    dist,prev=td_dijkstra(depot,t0)
    p=path_of(prev,depot,18);used=[prev[x][1]['eid'] for x in p[1:]]
    print(f"start {t0:05.2f}: arrive {dist[18]:.2f} nodes {p} edges {used}")

print("\n-- pairwise shortest-path differences 08:00 vs 12:00 --")
diff=0;ex=[]
for s,g in itertools.product(range(N),range(N)):
    if s>=g:continue
    d1,p1=td_dijkstra(s,8.0);d2,p2=td_dijkstra(s,12.0)
    if g in d1 and g in d2:
        q1=path_of(p1,s,g);q2=path_of(p2,s,g)
        e1=[p1[x][1]['eid'] for x in q1[1:]];e2=[p2[x][1]['eid'] for x in q2[1:]]
        if e1!=e2:
            diff+=1
            if len(ex)<6:ex.append((s,g,q1,q2))
print("pairs with changed shortest path:",diff)
for s,g,q1,q2 in ex: print(f"  {s}->{g}: 08:00 {q1} | 12:00 {q2}")
