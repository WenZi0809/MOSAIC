# -*- coding: utf-8 -*-
"""
Created on Tue Mar 09 10:06:07 2021

@author: DELL
"""
from scipy.stats.mstats import zscore
import numpy as np
from scipy.signal import convolve
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

def get_com_mat(matrix):
    matrix = matrix.copy()
    col_sum = matrix.sum(axis=0)
    matrix_correct = np.matrix(col_sum).T * np.matrix(col_sum) / col_sum.sum()
    matrix = matrix - matrix_correct
    matrix[np.isnan(matrix)] = 0
    return matrix

def get_mod(part, matrix, remv = None):
    ct_ma =  get_com_mat(matrix)
    cls = sorted(list(set(part) - set({remv})))
    mod = []
    for c in cls :
        mask = part == c
        mod.append(ct_ma[mask][:,mask].sum()*0.5 / matrix.sum())
    return mod,sum(mod)

def get_oe(matrix):
    matrix = np.array(matrix).copy()
    lens = matrix.shape[0]
    mask = np.ones(lens)
    cut_off = lens * 0.4
    matrix_num = matrix.copy()
    matrix_num[matrix_num > 0] = 1
    col_sum = matrix_num.sum(axis = 0)
    sites = np.arange(lens)
    for i in sites:
        if col_sum[i] <= cut_off:
            mask[i] = 0
    mask = mask == 1
    x_o = sites.copy()
    y_o = sites.copy()
    dis = np.zeros(lens)
    mask_o = mask.copy()
    exp_matrix = np.ones((lens, lens))
    x_e = sites.copy()
    y_e = sites.copy()
    for i in range(lens):
    ##obs_matrix
        diag_n = matrix[(x_o, y_o)]
        diag_line = diag_n[mask_o]
        if diag_line.shape[0] == 0:
            dis[i] = 0
        else :
            dis[i] = diag_line.mean()
            x_o = np.delete(x_o,-1)
            y_o = np.delete(y_o,0)
            mask_o = np.delete(mask_o,-1)
        
        ##exp_matrix
        if dis[i] == 0:
            x_e = np.delete(x_e,-1)
            y_e = np.delete(y_e,0)
            continue
        exp_matrix[(x_e, y_e)] = dis[i]
        exp_matrix[(y_e, x_e)] = dis[i]
        x_e = np.delete(x_e,-1)
        y_e = np.delete(y_e,0)

    oe = np.matrix(matrix/exp_matrix)
    return np.asarray(oe), mask
    

def rearange_shuffle(matrix, part):
    part = part.copy()
    ctmatrix = get_com_mat(matrix) 
    sites = np.arange(len(part))   
    np.random.shuffle(sites)
    classes = sorted(set(part) - {0.0})
     
    for i in sites:
        current_c = part[i]
 
     ##clustering
       
        mod_value = -1
        for c in classes:
            part_mask = part == c
            c_sites = sites[part_mask]
#            o_sites = sites[~part_mask]
            Lic_sum = ctmatrix[i,c_sites].mean() #matrix[i,c_sites].mean() / matrix[i,o_sites].mean() #
            if Lic_sum <= -1 and current_c != 0 : 
               continue
                                      
            values = Lic_sum 
            if values > mod_value :
               mod_value = values
               part[i] = c 
    return part

def get_mod_values(part, matrix, remv = None):
    ct_ma =  get_com_mat(matrix)
    cls = sorted(list(set(part) - set({remv})))
    mod = []
    part_mask = {}
    for c in cls :
        part_mask[c] = part == c
    for i,p in enumerate(part):
        value = ct_ma[i,part_mask[p]].mean()
        mod.append(value)
    
    return np.asarray(mod)

def rearange_seq(matrix, part):
    part = part.copy()
    ctmatrix = get_com_mat(matrix) 
    sites = np.arange(len(part))   
    classes = sorted(set(part) - {0.0})
    
    mod_sample = get_mod_values(part, matrix)
    mask = mod_sample <= 0
    for i in np.hstack((sites[mask],sites[~mask])):
        current_c = part[i]
     
     ##clustering
        
        mod_value = -1
        for c in classes:
            part_mask = part == c
            c_sites = sites[part_mask]

#            o_sites = sites[~part_mask]
            Lic_sum = ctmatrix[i,c_sites].mean() #matrix[i,c_sites].mean() / matrix[i,o_sites].mean() #
            if Lic_sum <= -1 and current_c != 0 : 
               continue
            
            values = Lic_sum 
            if values > mod_value :
               mod_value = values
               part[i] = c 
    return part

def sort_part(matrix, part):
    sites = np.arange(len(part))   
    sort_values = np.array(part)
    ranges = set(sort_values) - {0}
    list_values = []
    for i in ranges:
        mask = sort_values == i
        loci_site = sites[mask]
        means = matrix[loci_site, loci_site].mean()
        list_values.append([means,loci_site])

    sort_cls = sorted(list_values, key=lambda x: (x[0]))
    new_part = dict()
    for i in range(len(ranges)):
        for x in sort_cls[i][1]:
           new_part[x] = i
    
    outs = [new_part.get(j,-1) for j in sites]
         
    return np.array(outs, dtype = np.float)


def select_pc(pcs, matrix, lc=[0], remv = None):
        
    mod = 0 
    exclud = []
    exclud.extend(lc)
    sites = np.delete(np.arange(10),exclud)
    for i in sites:  
        label_mid = np.zeros(matrix.shape[0])
        pcmid = pcs[i]
        mask_u = pcmid > 0 #np.percentile(pcmid, 75)
        mask_d = pcmid <= 0 #np.percentile(pcmid, 25)
        label_mid[mask_u] = 3
        label_mid[mask_d] = 2
        c_mod = get_mod(label_mid, matrix , remv)[1]
        print i+1,c_mod
        if c_mod > mod :
            mod = c_mod
            lci = i
                
    return lci 

def similarity(mat,method = 'euclidean'):
    mat = np.asarray(mat.T)
    
    ma = np.matrix(mat)
    
    if method == 'euclidean':
       dis_mat = squareform(pdist(ma,method))
       similarity_mat = 1/np.exp(dis_mat/(2 * 1))
    elif method == 'cosine' :
       dis_mat = squareform(pdist(ma,method))
       similarity_mat = 1-dis_mat   
    elif method == 'corr' :
       similarity_mat = np.corrcoef(ma)
    

    return similarity_mat  

def Kmeans_SVD(matrix, mask, cent = 0, cluster_number=4):
    ##input O/E matrix/gap position/centromere position/cluster number
    mat = matrix[mask][:,mask]
    lens = mat.shape[0]      
    oe = mat.copy()
    ##svd
    u,s,v = np.linalg.svd(oe)                             ## first round of SVD
       
    del_s = []
    lc = select_pc(v, oe, [0], 0)                         
    svd_id = np.delete(np.arange(20),[0,lc])
    for i in svd_id:                                      ## remove centromere effect
        sites = np.ones(lens)
        p =  zscore(convolve(v[i],[1,1,1], mode = 'same'))
        sites[p<=0] = -1
        print i+1,abs(sites[:cent].mean()),abs(sites[cent:].mean())
        if (abs(sites[:cent].mean()) + abs(sites[cent:].mean())) >= 1.0:
            del_s.append(i)
        
        elif abs(sites[:cent].mean()) + abs(sites[cent:].mean()) >= 0.8:
            v[i][:cent] = v[i][:cent] - v[i][:cent].mean()
            u[:,i][:cent] = u[:,i][:cent] - u[:,i][:cent].mean()
            v[i][cent:] = v[i][cent:] - v[i][cent:].mean()
            u[:,i][cent:] = u[:,i][cent:] - u[:,i][cent:].mean()
              
    s[del_s] = 0 
    oe = np.dot((u*s),v)
    oe[oe<0] = 0
    oe2 =  oe.copy()
    u,s,v = np.linalg.svd(np.asarray(oe))                 ## second round of SVD
    lc = select_pc(v, oe, [0], 0)                         ## select EV1
    lc2 = select_pc(v, oe, [0,lc], 0)                     ## select EV2
        
    pc1 = v[lc].copy()
    pc2 = v[lc2].copy()
#    pc3 = v[lc3].copy()
    
    pct = 1
    p5 = np.percentile(pc1,pct)   
    pc1[pc1<p5] = p5  
    p95 = np.percentile(pc1,(100-pct))   
    pc1[pc1>p95] = p95  
    
    p5 = np.percentile(pc2,pct)   
    pc2[pc2<p5] = p5  
    p95 = np.percentile(pc2,(100-pct))   
    pc2[pc2>p95] = p95 
    

    X = np.asarray(zip(v[lc], v[lc2]*(s[lc2] / s[lc])))  ##weighted by eigenvalues
        
    label = KMeans(n_clusters=cluster_number, random_state=0).fit(X).labels_ + 1  ##K-means

    prd_buff = label.copy()
    mod = get_mod(label,oe2)[1]
    prd_pre = label.copy()            
    while True :                                        ##Louvain algorithm
        prd_pre = rearange_seq(oe2,prd_pre)
        
        l0 = len(prd_pre[(prd_pre - prd_buff) == 0]) / float(oe.shape[1])
        print l0,get_mod(prd_pre,oe2)[1]
        
        if  get_mod(prd_pre,oe2)[1] <= mod : 
            prd_pre = prd_buff.copy()
            break
        else :
            prd_buff = prd_pre.copy()
            mod = get_mod(prd_pre,oe2)[1]
        
    prd = sort_part(oe, prd_pre) + 1
    return prd

if __name__ == '__main__':
    oe, mask = get_oe(matrix)
    labels = Kmeans_SVD(oe, mask)

