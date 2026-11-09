# -*- coding: utf-8 -*-
"""Export the real Luoyang STSP-TVRAR instance to model-ready tensors / matrices."""
import json, os
import numpy as np
D=os.path.dirname(os.path.abspath(__file__))
inst=json.load(open(os.path.join(D,'luoyang_stsp_20node.json')))
nodes=inst['nodes']; arcs=inst['arcs']; N=len(nodes)

# ---- node features: [id, is_depot, is_delivery, is_transit, service_min] ----
node_feat=np.zeros((N,5))
for n in nodes:
    i=n['id']
    node_feat[i]=[i, 1 if n['kind']=='depot' else 0, 1 if n['kind']=='delivery' else 0,
                  1 if n['kind']=='transit' else 0, n['service_min']]
coord=np.array([[n['lat'],n['lon']] for n in nodes])

# ---- directed edge_index + edge_attr [cost, time_min, a(h), b(h), restricted] ----
# unrestricted window encoded as a=b=-1
edge_index=[];edge_attr=[]
for a in sorted(arcs,key=lambda z:z['arc_id']):
    edge_index.append([a['u'],a['v']])
    edge_attr.append([a['cost'],a['time_min'],
                      a['a'] if a['restricted'] else -1.0,
                      a['b'] if a['restricted'] else -1.0,
                      1.0 if a['restricted'] else 0.0])
edge_index=np.array(edge_index).T   # (2, E)
edge_attr=np.array(edge_attr)       # (E, 5)

# ---- dense matrices (directed): inf if no arc ----
INF=np.inf
C=np.full((N,N),INF); T=np.full((N,N),INF); WA=np.full((N,N),-1.0); WB=np.full((N,N),-1.0)
for a in arcs:
    u,v=a['u'],a['v']
    C[u,v]=a['cost']; T[u,v]=a['time_min']
    if a['restricted']: WA[u,v]=a['a']; WB[u,v]=a['b']
np.fill_diagonal(C,0);np.fill_diagonal(T,0)

np.savez(os.path.join(D,'luoyang_stsp_tensors.npz'),
         node_feat=node_feat, coord=coord, edge_index=edge_index, edge_attr=edge_attr,
         cost_matrix=C, time_matrix=T, win_a=WA, win_b=WB)

print("node_feat",node_feat.shape," edge_index",edge_index.shape," edge_attr",edge_attr.shape)
print("directed arcs:",edge_index.shape[1]," restricted arcs:",int(edge_attr[:,-1].sum()))
# compact adjacency for Q-network action space (out-neighbours per node)
adj={}
for i in range(N):
    outs=np.where(edge_index[0]==i)[0]
    adj[i]=[(int(edge_index[1,k]),) for k in outs]
print("out-degree:",{i:len(v) for i,v in adj.items()})
print("\nnode table:")
for n in nodes:
    print(f" {n['id']:2d} {n['kind']:8s} {n['name'][:22]:22s} serv={n['service_min']} ({n['lat']:.5f},{n['lon']:.5f})")
print("\nsaved luoyang_stsp_tensors.npz")
