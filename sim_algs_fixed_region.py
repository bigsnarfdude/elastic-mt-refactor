#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 23:12:47 2021

@author: tim
Simulation algorthims.
Started Jan 16. Same as sim_algs3, but attempted with fixed regions.
All other scripts remain with the same name, but will have new functions for this.
Simulation parameters are GLOBAL imported from parameters.py.
"""
import numpy as np
if np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('2.2.2'):
    np.set_printoptions(legacy='1.25') #print numbers the 'normal' way
import itertools as it
from comparison_fns import event, compare, inter_r_bdry2, dist, mt, which_region, bundle, deflect_angle, region_traj, bdl_exist, branch_geo, branch, earliest_event
from comparison_fns import sort_and_find_events, cat_rate, add_pi
from comparison_fns import nucleate, determine_branch_compass, sample_angle, periodic, ghost_pt,determine_branch_compass2, step_forward
from comparison_fns import check_fixed_region
from zippering import zip_cat
from sim_anchoring_alg_test import map_angle
from parameters import v_s, xdomain, ydomain, grid_w, grid_l, x_interval, y_interval
from parameters import del_l2, conv, deflect_on, edge_cat, vtol, tread_bool, cat_on, branch_stuff, pause_bool, bdl_on, plant, pseudo_bdry
from parameters import dr_tol, no_bdl_id, angle_dep_cat, vtol_nuc, R_meta, LDD_bool
if plant == 'arth':
    from parameters import f_gs, f_gp, f_ps, f_pg, f_sg, f_sp, r_n
else:
    from parameters import r_c, r_r, r_n, angle_part_n, angle_part, r_part, rate_dict
import random as rnd
from time import time
from plotting import plot_snap, order_param, order_hist, s2, s2_new, plot_hist_mtlist
from math import pi, sin, cos
import copy
import pickle
from multiprocessing import current_process
import sys
from inspect import currentframe #debugging

# Rule-ablation hook (SPRINT_S1). None => the real collision rule (baseline,
# bit-for-bit unchanged). A runner sets this to a fn (angle1, angle2, r) -> label
# to override ONLY the real-collision decision at site ~1023. Branch-nucleation
# geometry (sites ~1379, ~2020) deliberately does NOT pass it, so it stays on the
# real rule. Late-bound global lookup => assigning sim_algs.ABLATION takes effect.
ABLATION = None
def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno
mt_none = mt(None) #dummy mt for None placeholder in event(mt1,mt2=None,...)
# del_l2 = .1

class global_prop:
    '''
    Global simulation properties
    '''
    def __init__(self):
        self.nuc = 0 #number of nucleations
        self.free = 0 #number of freeing events
        self.cross = 0
        self.deflect = 0 #number of defection events
        self.more_anchoring = 0 #number of time gen_path is called after initial MT growth
        self.grow_mt = [] #list of actively growing 
        # for i in range(angle_part_n):
        #     self.grow_mt.append([])
        self.grow_angle = [] #corresponding list of growing mt angles
        self.shrink = [] #list of actively shrinking mts
        self.pause_mt = [] #paused plus-end mts
        self.stochastic_update = True #tells alg to recalculate stochastic events
        #statistics regarding mt lengths
        self.mtlen = [] #mt lengths prior to being non-free
        self.mtlen_t = [] #times corresponding to above
        self.init_angle = [] #initial angles for the above
        # self.mtlen_cont = [] #mt lengths of free mts that are hit
        # self.mtlen_cont_t = [] #corresponding times
        # self.init_angle_cont = [] #initial angles for the above
    def find_angle_part(self, angle):#find angle partition index for a given angle
        new_angle = map_angle(angle) #map to [0,pi/2]
        idx = -1
        for i in range(angle_part_n):
            if new_angle >= angle_part[i] and new_angle < angle_part[i+1]:
                idx = i
                break
        assert idx != -1
        return(idx)
    def tread_replace(self, ext_idx, mt_idx): #tread causes re-naming of root mt
        # growing = False #whether this mt is growing
        # for j, grow_list in enumerate(self.grow_mt): #check growing sublists
        #     if mt_idx in grow_list:
        #         i = self.grow_mt[j].index(mt_idx)
        #         self.grow_mt[j][i] = ext_idx
        #         growing = True
        #         break
        if mt_idx in self.grow_mt:
            i = self.grow_mt.index(mt_idx)
            self.grow_mt[i] = ext_idx
        elif mt_idx in self.pause_mt:
            i = self.pause_mt.index(mt_idx)
            self.pause_mt[i] = ext_idx
        else:
            i = self.shrink.index(mt_idx)
            self.shrink[i] = ext_idx
    def shrunk(self, mt, mt_list, mt_idx, event): #newly shrinking mt, must update pop.
        original_mt = mt_idx #to find root MT
        mt_found = mt
        # angle = mt.angle[-1] #in case we need this angle
        while mt_found.prev_mt is not None:
            original_mt = mt_found.prev_mt
            mt_found = mt_list[original_mt]
        if event.prev != 'pause_to_shrink':
            assert (not mt_found.grow) or mt_found.hit_bdry
            # grow_i = self.find_angle_part(angle)
            del_idx = self.grow_mt.index(original_mt) #delete from growing mt list
            del self.grow_mt[del_idx]
            del self.grow_angle[del_idx]
            self.shrink.append(original_mt) #add to shrinking list
            self.stochastic_update = True #must recalculate stochastics
        else:
            del_idx = self.pause_mt.index(original_mt)
            del self.pause_mt[del_idx]
            self.shrink.append(original_mt)
            self.stochastic_update = True
    def grow(self, mt, mt_list, mt_idx, event): #rescued mt
        angle = map_angle(mt.angle[-1]) #TODO put this in loop?
        # grow_i = self.find_angle_part(angle)
        original_mt = mt_idx #to find root MT
        mt_found = mt
        while mt_found.prev_mt is not None:
            original_mt = mt_found.prev_mt
            mt_found = mt_list[original_mt]
        if event.prev != 'pause_to_grow':
            del_idx = self.shrink.index(original_mt) #delete from shrinking mt list
            del self.shrink[del_idx]
            self.grow_mt.append(original_mt) #add newly rescued mt
            self.grow_angle.append(angle) #and corresponding angle
            self.stochastic_update = True #must recalculate stochastics
        else:
            del_idx = self.pause_mt.index(original_mt) #delete from shrinking mt list
            del self.pause_mt[del_idx]
            self.grow_mt.append(original_mt)
            self.grow_angle.append(angle)
            self.stochastic_update = True
    def pause(self, mt, mt_list, mt_idx, event):
        # angle = map_angle(mt.angle[-1]) #TODO put this in loop?
        original_mt = mt_idx #to find root MT
        mt_found = mt
        while mt_found.prev_mt is not None:
            original_mt = mt_found.prev_mt
            mt_found = mt_list[original_mt]
        if event.prev == 'grow_to_pause':
            # grow_i = self.find_angle_part(angle)
            del_idx = self.grow_mt.index(original_mt) #delete from growing mt list
            del self.grow_mt[del_idx]
            del self.grow_angle[del_idx]
            self.pause_mt.append(original_mt)
            self.stochastic_update = True
        else:
            del_idx = self.shrink.index(original_mt) #delete from shrinking mt list
            del self.shrink[del_idx]
            self.pause_mt.append(original_mt)
            self.stochastic_update = True
    def nucleate(self, mt, mt_idx): #nucleation event
        self.nuc += 1
        angle = map_angle(mt.angle[-1])
        # grow_i = self.find_angle_part(angle)
        self.grow_mt.append(mt.number)
        self.grow_angle.append(angle)
        self.stochastic_update = True #must recalculate stochastics
    def update_grow_mt(self, mt, mt_list, mt_idx, angle_prev,t): #check if MT is in new angle partition
        #TODO check if this makes sense with new method that doesn't use partitions
        # angle_new = map_angle(mt.angle[-1])
        # n_prev, n_new = self.find_angle_part(angle_prev), self.find_angle_part(angle_new)
        # if n_prev != n_new: #if it enters a new partition, change
        #     # assert n_prev < n_new
        #     original_mt = mt_idx #to find root MT
        #     mt_found = mt
        #     while mt_found.prev_mt is not None:
        #         original_mt = mt_found.prev_mt
        #         mt_found = mt_list[original_mt]
        #     del_idx = self.grow_mt[n_prev].index(original_mt) #delete from growing mt list
        #     del self.grow_mt[n_prev][del_idx]
        #     self.grow_mt[n_new].append(original_mt)
        # t_real = t*conv/60/60
        if angle_dep_cat: #only need to update in this case
            self.stochastic_update = True
    def record_len(self, mt1, t, cont = False, mt_list = None, bdl_list = None): #record mt length due to non-freeing event
        if mt1.free:
            phi = mt1.angle[-1]
            if abs(phi - pi/2)>vtol and abs(phi - 3*pi/2)>vtol:
                mt_len = mt1.total_len + np.sum(mt1.seg_dist)
                if mt1.grow:
                    mt_len += t-mt1.update_t[-1]
                self.mtlen.append(mt_len)
                self.mtlen_t.append(t)
                self.init_angle.append(mt1.init_angle)
                if mt1.init_angle == None:
                    print(mt1.number, mt1.init_angle)
                assert mt1.init_angle != None
                    
def stochastic(t,global_prop):
    '''
    Next nucleation time using rate k, from Gillespie

    Parameters
    ----------
    t : Current time
    Returns
    -------
    Next nucleation time and nucleation point

    '''
    N_grow = len(global_prop.grow_mt) #no. of growing MTs
    N_shrink = len(global_prop.shrink) #no. of shrinking MTs

    # L= 40
    lx = xdomain[1]-xdomain[0]
    ly = ydomain[1]-ydomain[0]
    k = r_n*lx*ly
    # k = lx*ly*L**3*0.1/8#nucleation rate
    # k = 2

    # r_c = 4.5 #catastrophe rate
    # r_r = 1#rescue rate
    if pause_bool:
        N_pause = len(global_prop.pause_mt)
        a1, a2, a3, a4, a5, a6, a7 = k, f_gp*N_grow, f_gs*N_grow, f_sp*N_shrink, f_sg*N_shrink, f_pg*N_pause, f_ps*N_pause #propensities
        R = a1+a2+a3+a4+a5+a6+a7 #total transition rate
        r = rnd.uniform(0,1) #for generating time
        dt = -np.log(r)/R #new time
        T = t+dt #next time
        r2 = rnd.uniform(0,R) #for choice of event
        pt = None #if nucleate, this is the pt
        stoch_policy = None
        mt_no = None #if catastrophe or rescue, which one we choose
        if r2 <= a1: #nucleation
            x, y = rnd.uniform(xdomain[0],xdomain[1]), rnd.uniform(ydomain[0],ydomain[1])
            pt = [x,y]
            stoch_policy = 'nucleate'
        elif r2 <= a2+a1 and r2> a1: #catastrophe
            mt_idx = rnd.randint(0,N_grow-1) #choose mt
            mt_no = global_prop.grow_mt[mt_idx] #chosen mt no.
            stoch_policy = 'grow_to_pause'
        elif r2 <= a1+a2+a3 and r2 > a2+a1:
            mt_idx = rnd.randint(0, N_grow-1)
            mt_no = global_prop.grow_mt[mt_idx] #chosen mt no.
            stoch_policy = 'sp_catastrophe' #spontaneous, not induced
        elif r2 <= a1+a2+a3+a4 and r2 > a2+a1+a3:
            mt_idx = rnd.randint(0, N_shrink-1)
            mt_no = global_prop.shrink[mt_idx] #chosen mt no.
            stoch_policy = 'shrink_to_pause'
        elif r2 <= a1+a2+a3+a4+a5 and r2 > a2+a1+a3+a4:
            mt_idx = rnd.randint(0, N_shrink-1)
            mt_no = global_prop.shrink[mt_idx] #chosen mt no.
            stoch_policy = 'rescue'
        elif r2 <= a1+a2+a3+a4+a5+a6 and r2 > a2+a1+a3+a4+a5:
            mt_idx = rnd.randint(0, N_pause-1)
            mt_no = global_prop.pause_mt[mt_idx] #chosen mt no.
            stoch_policy = 'pause_to_shrink'
        else: #rescue
            mt_idx = rnd.randint(0, N_pause-1)
            mt_no = global_prop.pause_mt[mt_idx] #chosen mt no.
            stoch_policy = 'pause_to_grow'
        return(stoch_policy, T, pt, mt_no)
    else:
        prop = [0] #cumulative propensities, including zero
        for rate in rate_dict:
            r = rate_dict[rate]
            if rate == 'r_n':
                r = r*lx*ly
                prop.append(r)
            elif rate == 'r_r':
                r = r*N_shrink
                prop.append(r)
            elif rate == 'r_c':
                # p = angle_part_n
                # for i in range(p):
                #     r = r_part[i]*len(global_prop.grow_mt[i])
                #     prop.append(r)
                for angle in global_prop.grow_angle:
                    r = cat_rate(angle,t)
                    prop.append(r)
        prop_interval = np.cumsum(prop)
        R = prop_interval[-1] #total rate
        r = rnd.uniform(0,1) #for generating time
        dt = -np.log(r)/R #new time
        T = t+dt #next time
        r2 = rnd.uniform(0,R) #for choice of event
        pt = None #if nucleate, this is the pt
        stoch_policy = None
        mt_no = None #if catastrophe or rescue, which one we choose
        for i in range(len(prop_interval)-1):
            if r2 <= prop_interval[i+1] and r2 >= prop_interval[i]:
                if i < 2: #non-cat events, no subsampling needed
                    if i == 0:
                        x, y = rnd.uniform(xdomain[0],xdomain[1]), rnd.uniform(ydomain[0],ydomain[1])
                        pt = [x,y]
                        stoch_policy = 'nucleate'
                    else:
                        mt_idx = rnd.randint(0,N_shrink-1) #choose mt
                        mt_no = global_prop.shrink[mt_idx] #chosen mt no.
                        stoch_policy = 'rescue'
                    break
                else: #need to sample within cat events
                    j = i - 2 #index of growing sublist
                    assert j <= len(global_prop.grow_angle) and j >=0
                    # N_grow = len(global_prop.grow_mt[j])
                    # mt_idx = rnd.randint(0,N_grow-1) #choose mt
                    mt_no = global_prop.grow_mt[j] #chosen mt no.
                    stoch_policy = 'sp_catastrophe' #spontaneous, not induced
                    break

        # a1, a2, a3 = k, r_c*N_grow, r_r*N_shrink #propensities
        # R = a1+a2+a3 #total transition rate
        # r = rnd.uniform(0,1) #for generating time
        # dt = -np.log(r)/R #new time
        # T = t+dt #next time
        # r2 = rnd.uniform(0,R) #for choice of event
        # pt = None #if nucleate, this is the pt
        # stoch_policy = None
        # mt_no = None #if catastrophe or rescue, which one we choose
        # if r2 <= a1: #nucleation
        #     x, y = rnd.uniform(xdomain[0],xdomain[1]), rnd.uniform(ydomain[0],ydomain[1])
        #     pt = [x,y]
        #     stoch_policy = 'nucleate'
        # elif r2 <= a2+a1 and r2> a1: #catastrophe
        #     mt_idx = rnd.randint(0,N_grow-1) #choose mt
        #     mt_no = global_prop.grow_mt[mt_idx] #chosen mt no.
        #     stoch_policy = 'sp_catastrophe' #spontaneous, not induced
        # else: #rescue
        #     mt_idx = rnd.randint(0,N_shrink-1) #choose mt
        #     mt_no = global_prop.shrink[mt_idx] #chosen mt no.
        #     stoch_policy = 'rescue'
        return(stoch_policy, T, pt, mt_no)

def update_event_list(MT_list,mt_sublist,event_list,bdl_list,branch_list,region_list,t,global_prop,last_result = None, result2 = None):
    '''
    Deletes invalid and calculate new events: edits and reorders events within event_list only.

    Parameters
    ----------
    MT_list : List of MT objects
    pair_list : List of tuples of pairwise MT indices. Each tuple represents MTs to be compared
    t: current time
    pevet_list: list of previous events which occured
    Returns
    -------
    Collision policy
    BE CAREFUL: THE DISTANCE RETURNED IN BDRY COLLISION IS DISTANCE OF PRESENT TIP TO BDRY, NOT
    SEGMENT LENGTH!!!
    '''
    r = None
    n = None
    mt_index = None# [mt.number for mt in MT_list if mt.exist] #indices of active mts
    l = None #declare
    sort_regions = []
    # print(t)
    if t==0:
        if len(MT_list) != 0:
            mt_index = [mt.number for mt in MT_list if mt.exist]
            pair_list = list(it.combinations(mt_index,2)) #list of combinations of indices of MTs
            pair_list[:] = [pair for pair in pair_list if check_fixed_region(pair[0],pair[1],MT_list,t,r)]
            l = len(pair_list)
            n = len(mt_index)
            for j in range(n): #check for bdry collisions
                if MT_list[j].grow == True: #no bdry collision
                    MT = MT_list[j]
                    R = MT.region #XXX
                    if R not in sort_regions:
                        sort_regions.append(R)
                    bdry_res = inter_r_bdry2(MT, MT_list, bdl_list,region_list, free = True) #find intersection info
                    next_time = bdry_res[0] + MT.update_t[-1] #time of collision
                    event_list[R].append(event(MT,mt_none,next_time,bdry_res[1],bdry_res[2])) #XXX
    else:
        pevent = event_list[result2[0]][0] #for last event info
        policy = event_list[result2[0]][0].policy #get policy
        l_idx = last_result
        pt = pevent.pt
        t = pevent.t
        if policy in ['top','bottom', 'left','right', 'reuse_bdry']:
            MT = MT_list[last_result] #this is the extension
            old_idx = MT.prev_mt #index of previous MT
            MT_prev = MT_list[old_idx]
            new_idx = last_result#index of new MT
            event_list_R1 = event_list[MT.region] #list in new region
            event_list_R2 = event_list[MT_prev.region] #list in old region
            sort_regions = [MT.region,MT_prev.region]
            # bdl_prev = bdl_list[MT_prev.bdl]
            bdl = bdl_list[MT.bdl]
            #mts to compare w/ extension mt
            MT_list1 = mt_sublist[MT.region]
            branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
            mt_index= [mt.number for mt in MT_list1 if mt.exist and (mt.number != new_idx) \
            and (MT.region==mt.region) and (mt.number not in (bdl.mts)) \
            and (mt.bdl not in bdl.cross_bdl+branch_bdl) and (MT.free or mt.free)]
            pair_list = list(it.product(mt_index,[new_idx])) #generate new pairs
            #no need to delete events involving original mt, these should not exist by collision calculations
            # event_list.pop(0)
            event_list_R2[:] = [x for x in event_list_R2 if not (x.mt1_n == old_idx and x.policy == policy)]
            l = len(pair_list)
            #check for newly branched mt's intersection w/ crossovers, branches, overtakes, deflect on its bdl
            bdl.cross_bdl_event(MT, event_list_R1)
            # if len(bdl_prev.mts) > 1: #regardless if it is recalculated, find wall intersection from current traj
            #     bdl_prev.new_bdl_deflect(event_list,MT_list,bdl_list,region_list,pt,t,mt_not=MT_prev,bdry=True)
            if len(bdl.mts) > 1 or not deflect_on or policy == 'reuse_bdry': #only care about traffic if there are 0 or >1 mts
                bdl.branch_bdl_event(MT, event_list_R1, branch_list)
                bdl.deflect_bdl_event_v(event_list_R1,MT_list,bdl_list,region_list,t,MT,MT.seg[-1]) #use seg[-1] instead of pt in case of periodic bdry pt
                if not bdl.pseudo_bdry: #don't care about bdl traffic if it's a bdry
                    bdl.overtake_bdl_event(MT, MT.seg[-1], t, MT_list, event_list_R1)
            else:
                # bdry events
                bdry_res = inter_r_bdry2(MT, MT_list, bdl_list, region_list, free=True) #find intersection info
                next_time = bdry_res[0] + MT.update_t[-1] #time of collision
                event_list_R1.append(event(MT,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t=t))
            if MT_prev.tread: #if tread, will disap
                mt_l = np.sum(MT_prev.seg_dist)#total length of mt
                disap_t = mt_l/(MT_prev.vt)+MT_prev.tread_t #collision distances from 1 to 2, growing can only collide w/ shrinking
                event_list_R2.append(event(MT_prev,mt_none,disap_t,None,'disap_tread',calc_t=t))
        elif policy in ['deflect', 'follow_bdl']: #deflection
            MT = MT_list[l_idx]
            sort_regions = [MT.region]
            event_list_R1 = event_list[MT.region]
            new_idx = [last_result] #index of updated MTs
            #traffic events not the current one are still valid
            event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n!=last_result and x.mt2_n!=last_result) or \
                              (x.policy in ['nucleate','sp_catastrophe','rescue','1catch2','1catch2_m','cross_br','uncross_m','cross_bdl'])] #valid_events(x,last_result)]
            mt_index = None #declare
            MT_list1 = mt_sublist[MT.region]
            bdl1 = bdl_list[MT.bdl] #need to exclude mts in the bdl
            branch_bdl = [branch_list[brn].branch_bdln for group in bdl1.branchn for brn in group if brn != 0]
            mt_index= [mt.number for mt in MT_list1 if (mt.exist and (MT.free or mt.free) and (mt.number != last_result) \
            and (MT.region==mt.region) and mt.number not in (bdl1.mts)) and mt.bdl not in bdl1.cross_bdl + branch_bdl]
            if policy == 'deflect': #'deflect' creates a new pt on the bdl
                bdl1.new_bdl_deflect(event_list_R1,MT_list,bdl_list,region_list,pt,t,mt_not=MT)
                #bdry events, only happens for leading/free mts
                bdry_res = inter_r_bdry2(MT,MT_list, bdl_list, region_list, free = True) #find intersection info
                next_time = bdry_res[0] + MT.update_t[-1] #time of collision
                event_list_R1.append(event(MT,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t=t))
            else: #follow bdl does not calculate a new pt, just search for new pt
                bdl1.deflect_bdl_event_v(event_list_R1,MT_list,bdl_list,region_list,t,MT,pt)
            n = len(mt_index)
            pair_list = list(it.product(mt_index,new_idx)) #generate new pairs
            l = len(pair_list)
            global_prop.update_grow_mt(MT, MT_list, l_idx, MT.angle[-2],t)
            #ANGULAR CAT
            #uses its own previous angle
        elif policy == 'disap':
            event_list_R1 = event_list[MT_list[l_idx].region] #events in current region
            if MT_list[l_idx].from_bdry and not MT_list[l_idx].tread: #and it's an extension
                prev_idx = MT_list[l_idx].prev_mt #index of newly shrinking MT
                MTp = MT_list[prev_idx]
                sort_regions = [MT_list[l_idx].region,MTp.region]
                event_list_R2 = event_list[MTp.region] #events in previous mt's region
                event_list_R2[:] = [x for x in event_list_R2 if (x.mt1_n != prev_idx and x.mt2_n != prev_idx)\
                                  or (x.policy in ['nucleate','sp_catastrophe','rescue','1catch2_m', 'uncross_m'])]#discard newly shrinking one
                event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != last_result and x.mt2_n != last_result)\
                                  or (x.policy in ['nucleate','sp_catastrophe','rescue'])]#discard other older ones
                
                MT_list1 = mt_sublist[MTp.region]
                bdl = bdl_list[MTp.bdl]
                bdl.uncross_bdl_event(MTp, pevent, event_list_R2, bdl_list) #might uncross on bdl
                bdl.overtake_bdl_event(MTp, MTp.seg[-1], t, MT_list, event_list_R2) #newly shrinking for traffic, pt is last pt of MTp
                branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
                mt_index = [mt.number for mt in MT_list1 if (mt.exist and (mt.number != prev_idx) \
                and (mt.grow) and (MTp.region==mt.region)) and (MTp.free or mt.free)  \
                and (mt.number not in bdl.mts) and (mt.bdl not in bdl.cross_bdl+branch_bdl)] #usual exclusions + exclude mts on bdl
                pair_list = list(it.product(mt_index,[prev_idx])) #new pairs
                l = len(pair_list)
                assert MT_list[l_idx].exist==False
                #bdry events
                mt_l = np.sum(MT_list[prev_idx].seg_dist)#total length of shrinking mt, cannot got lower than this
                if MTp.tread:
                    mt_l -= MTp.vt*(t-MTp.tread_t) #left over len from tread
                    disap_t = mt_l/(v_s+MTp.vt)+t #collision distances from 1 to 2, growing can only collide w/ shrinking
                    event_list_R2.append(event(MTp,mt_none,disap_t,None,'disap',calc_t=t))
                else:
                    disap_t = mt_l/v_s+t #collision distances from 1 to 2, growing can only collide w/ shrinking
                    event_list_R2.append(event(MTp,mt_none,disap_t,None,'disap',calc_t=t))
                # disap_t = mt_l/v_s+t #collision distances from 1 to 2, growing can only collide w/ shrinking
                # event_list.append(event(MTp,mt_none,disap_t,None,'disap',calc_t=t))
            else: #if last one disappeared, no continuation
                event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != last_result and x.mt2_n != last_result)]#discard all invalid events
                l = 0 #no new comparisons to be made with non-existant MT
                assert MT_list[l_idx].exist==False
                if not MT_list[l_idx].hit_bdry:
                    del_idx = global_prop.shrink.index(MT_list[l_idx].number) #delete from shrhinking mt list
                    del global_prop.shrink[del_idx]
                else:
                    del_idx = global_prop.pause_mt.index(MT_list[l_idx].number) #delete from shrhinking mt list
                    del global_prop.pause_mt[del_idx]
                global_prop.stochastic_update = True #must recalculate stochastics
        elif policy == 'disap_tread':
            event_list_R1 = event_list[MT_list[l_idx].region]
            if MT_list[l_idx].hit_bdry: #and it's an extension
                ext_idx = MT_list[l_idx].ext_mt #index of newly shrinking MT
                MText = MT_list[ext_idx]
                sort_regions = [MT_list[l_idx].region,MText.region]
                event_list_R2 = event_list[MText.region] #ext mt's region
                event_list_R2[:] = [x for x in event_list_R2 if (x.mt1_n != ext_idx and x.mt2_n != ext_idx)\
                                  or (x.policy in ['nucleate','sp_catastrophe','rescue'])
                                  or (x.policy in ['top','bottom', 'left','right', 'reuse_bdry','1catch2','1catch2_m','cross_br','cross_bdl','deflect','follow_bdl','uncross'])]#discard newly shrinking one
                event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != last_result and x.mt2_n != last_result)\
                                  or (x.policy in ['nucleate','sp_catastrophe','rescue'])]#discard other older ones
                MT_list1 = mt_sublist[MText.region]
                bdl = bdl_list[MText.bdl]
                bdl.uncross_bdl_event(MText, pevent, event_list_R2, bdl_list) #might uncross on bdl
                bdl.overtake_bdl_event(MText, MText.seg[0], t, MT_list, event_list_R2) #newly shrinking for traffic, pt is last pt of MTp
                branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
                mt_index = [mt.number for mt in MT_list1 if (mt.exist and (mt.number != ext_idx) \
                and (MText.region==mt.region)) and (MText.free or mt.free)  \
                and (mt.number not in bdl.mts) and (mt.bdl not in bdl.cross_bdl+branch_bdl)] #usual exclusions + exclude mts on bdl
                pair_list = list(it.product(mt_index,[ext_idx])) #new pairs
                l = len(pair_list)
                assert MT_list[l_idx].exist==False
                #bdry events
                mt_l = np.sum(MText.seg_dist)#total length of shrinking mt, cannot got lower than this
                if not MText.hit_bdry and not MText.grow:
                    mt_l -= v_s*(t-MText.update_t[-1]) #left over len from cat
                    disap_t = mt_l/(v_s+MText.vt)+t #collision distances from 1 to 2, growing can only collide w/ shrinking
                    event_list_R2.append(event(MText,mt_none,disap_t,None,'disap',calc_t=t))
                elif MText.hit_bdry and MText.ext_mt != None:
                    disap_t = mt_l/MText.vt+t
                    event_list_R2.append(event(MText,mt_none,disap_t,None,'disap_tread',calc_t=t))
                elif MText.hit_bdry:
                    disap_t = mt_l/MText.vt+t
                    event_list_R2.append(event(MText,mt_none,disap_t,None,'disap',calc_t=t))
                # disap_t = mt_l/MText.vt+t #collision distances from 1 to 2, growing can only collide w/ shrinking
                # event_list.append(event(MText,mt_none,disap_t,None,'disap',calc_t=t))
                global_prop.tread_replace(ext_idx,l_idx)
            else: #if last one disappeared, no continuation
                sort_regions = [MT_list[l_idx].region]
                event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != last_result and x.mt2_n != last_result)]#discard all invalid events
                l = 0 #no new comparisons to be made with non-existant MT
                assert MT_list[l_idx].exist==False
                del_idx = global_prop.shrink.index(MT_list[l_idx].number) #delete from shrhinking mt list
                del global_prop.shrink[del_idx]
                global_prop.stochastic_update = True #must recalculate stochastics
        elif policy in ['1hit2','2hit1','cross','zipper','catas', 'follow_br','freed_br','freed_rescue', 'entrain_spaced', 'entrain_other_region']:
            mt1, mt2 = MT_list[pevent.mt1_n], None #mt2 might not exist for catas
            if pevent.mt2_n is not None: #if mt2 does exist
                mt2 = MT_list[pevent.mt2_n]
            if mt1.number != last_result and policy != 'freed_rescue': #need to switch if '2hit1', freed rescue is excempt
                mt1, mt2 = MT_list[pevent.mt2_n], MT_list[pevent.mt1_n]
            bdl1, bdl2 = None, None #bdls might not exist for catas
            if policy not in ['catas','follow_br','freed_br','freed_rescue']: #bdl2 is None for these
                bdl1, bdl2 = bdl_list[mt1.bdl], bdl_list[mt2.bdl]
            MT = MT_list[l_idx] #TODO BE CAREFUL, MT != mt1?
            MT_list1 = mt_sublist[MT.region]
            # elif MT.grow is False and not MT.hit_bdry: #if updated MT is due to cat. and it didn't disap
            event_list_R1 = event_list[MT.region]
            sort_regions = [MT.region]
            if policy == 'catas': #TM
                #delete events
                event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != last_result and x.mt2_n != last_result)\
                                  or (x.policy in ['nucleate','sp_catastrophe','rescue','uncross_m'])\
                                  or (x.policy=='1catch2_m' and x.mt2_n == last_result)]#discard all invalid events
                #find changed mts
                bdl1 = bdl_list[mt1.bdl]
                bdl1.uncross_bdl_event(mt1, pevent, event_list_R1, bdl_list) #add uncross events
                bdl1.overtake_bdl_event(mt1, pt, t, MT_list, event_list_R1) #newly shrinking for traffic
                branch_bdl = [branch_list[brn].branch_bdln for group in bdl1.branchn for brn in group if brn != 0.5]
                mt_index = [mt.number for mt in MT_list1 if (MT.free or mt.free) and\
                (mt.number not in bdl1.mts) and (mt.bdl not in bdl1.cross_bdl+branch_bdl) and mt.exist and \
                (mt.number != last_result) and (mt.grow) and (MT.region==mt.region)] #exclude all mts involved w/ bdl1
                #generate new pairs
                n = len(mt_index)
                new_idx = [last_result] #index of updated MTs
                pair_list = list(it.product(mt_index,new_idx))
                l = len(pair_list)
                #bdry collisions
                mt_l = np.sum(MT_list[l_idx].seg_dist)#total length of shrinking mt, cannot got lower than this
                if MT.tread:
                    mt_l -= MT.vt*(t-MT.tread_t) #left over len from tread
                    disap_t = mt_l/(v_s+MT.vt)+t #collision distances from 1 to 2, growing can only collide w/ shrinking
                    event_list_R1.append(event(MT,mt_none,disap_t,None,'disap',calc_t=t))
                else:
                    disap_t = mt_l/v_s+t #collision distances from 1 to 2, growing can only collide w/ shrinking
                    event_list_R1.append(event(MT,mt_none,disap_t,None,'disap',calc_t=t))
                global_prop.shrunk(MT, MT_list, last_result, pevent) #update shrinking pop
            elif policy == 'cross' and cat_on and bdl_on: #TM
                #list of mts to compare
                branch_bdl = [branch_list[brn].branch_bdln for group in bdl1.branchn for brn in group if brn != 0.5]
                mt_index= [mt.number for mt in MT_list1 if mt.exist and (mt.number != last_result) \
                and (MT.region==mt.region) and (mt.bdl != bdl2.number) and (mt.bdl != bdl1.number) \
                and (mt.bdl not in bdl1.cross_bdl+branch_bdl) and (MT.free or mt.free)]
                #generate pairs to compare
                n = len(mt_index)
                new_idx = [last_result] #index of updated MTs
                pair_list = list(it.product(mt_index,new_idx)) #generate new pairs
                l = len(pair_list)
                #delete invalid events
                event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != last_result and x.mt2_n != last_result)\
                                  or (x.policy in ['nucleate','sp_catastrophe','rescue','cross_br','cross_bdl','uncross_m'])\
                                  or (x.mt2_n == mt1.number and x.policy=='1catch2_m')]#discard all involving updated mt
                                  #for no-deflect case: cross_br events still valid after new cross events; never happens in deflect case
                event_list_R1[:] = [x for x in event_list_R1 if (not ((x.mt1_n in bdl1.mts+bdl2.mts)\
                        and (x.mt2_n in bdl1.mts+bdl2.mts)) and x.pt != pt) or x.policy in ['1catch2','1catch2_m']] #discard all events between mts on both bdls
                # event_list[:] = [x for x in event_list if not ((x.bdl1==bdl1.number and x.bdl2==bdl2.number)\
                #         or (x.bdl1==bdl2.number and x.bdl2==bdl1.number)) and x.pt != pt] #discard all events between mts on both bdls
                #calculate possible cross/uncross events
                if not no_bdl_id:
                    bdl2.new_cross_events(event_list_R1, MT_list, pt, t, bdl_list) #add events along bdl2 #TM
                    bdl1.new_cross_events(event_list_R1, MT_list, pt, t, bdl_list, mt_not = mt1) #add events along bdl1, exclude mt1 #TM TODO: is the mt_not calc redundant?
                #bdry events
                bdl1.deflect_bdl_event(event_list_R1,MT_list,bdl_list,region_list,t,MT,pt,bypass=True)
                # bdry_res = inter_r_bdry2(MT,MT_list, bdl_list, region_list, free = True) #find intersection info
                # next_time = bdry_res[0] + MT.update_t[-1] #time of collision
                # event_list.append(event(MT,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t=t))
            elif policy == 'cross' and (not cat_on or not bdl_on):
                event_list_R1[:] = [x for x in event_list_R1 if x.pt != pt]
            elif policy in ['zipper','follow_br','freed_br','freed_rescue']:
                assert MT.hit_bdry and not MT.grow
                # bdl = bdl_list[MT.bdl]
                old_idx = last_result #index of previous MT
                new_idx = MT.ext_mt #index of new MT
                mt_br = MT_list[new_idx] #new mt = mt2
                bdl = bdl_list[mt_br.bdl] #bdl of branched mt
                bdl3 = bdl2 #confusing but I rename bdl2 to be something else below, still need this variable
                bdl2 = bdl_list[MT.bdl] #bdl of original mt BE CAREFUL, SAME AS bdl1 if zipper
                event_list_R1 = event_list[MT.region]
                #mts to compare w/ branched mt
                branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
                mt_index1= [mt.number for mt in MT_list1 if (mt.exist and (mt.number != new_idx) \
                and (mt_br.region==mt.region) and mt.number != last_result) and (mt_br.free or mt.free) \
                and (mt.number not in bdl.mts) and (mt.bdl not in bdl.cross_bdl+branch_bdl)]
                pair_list = list(it.product(mt_index1,[new_idx])) #generate new pairs
                #mts to compare w/ original mt
                branch_bdl2 = [branch_list[brn].branch_bdln for group in bdl2.branchn for brn in group if brn != 0]
                mt_index2= [mt.number for mt in MT_list1 if (mt.exist and (mt.number != last_result) \
                and (MT.region==mt.region) and mt.number != new_idx) and (MT.free or mt.free)\
                and (mt.number not in bdl2.mts) and (mt.bdl not in bdl2.cross_bdl+branch_bdl2)] #idxs for original mt
                pair_list2 = list(it.product(mt_index2,[last_result])) #generate new pairs
                #combined pair list
                pair_list = pair_list+pair_list2
                l = len(pair_list)
                if policy == 'zipper': #TODO more precise way to calculate zippering from the same branch - avoid small numerical errors?
                    mt_not = bdl1.mts + bdl3.mts #discard future zipper events between these two, will be calculated from cross_bdl
                    event_list_R1[:] = [x for x in event_list_R1 if (not (x.mt1_n in mt_not and x.mt2_n in mt_not) and (x.mt1_n != old_idx and x.mt2_n != old_idx))\
                                      or (x.policy in ['nucleate','sp_catastrophe','rescue'])
                                      or (x.mt1_n == mt1.number and x.policy=='uncross_m')\
                                      or (x.mt1_n != mt1.number and x.policy in ['1catch2','1catch2_m'])]#discard all invalid events
                else:
                    event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != old_idx and x.mt2_n != old_idx)\
                                      or (x.policy in ['nucleate','sp_catastrophe','rescue'])
                                      or (x.mt1_n == old_idx and x.policy=='uncross_m')\
                                      or (x.mt1_n != old_idx and x.policy in ['1catch2_m'])]#exclude 1catch2?
                    if policy == 'follow_br' and mt_br.bdl != pevent.bdl2 and pevent.bdl2 != 0:
                        #in the case of natural branch being replaced at same pt, must edit events to involve this new branch
                        #otherwise, purge list might delete event
                        for n in range(len(event_list_R1)): #give the events the new branch bdl
                            ev = event_list_R1[n]
                            if ev.pt==pt and ev.policy == 'follow_br' and ev.bdl1 == bdl2.number:
                                ev.bdl2 = mt_br.bdl
                if MT.tread: #if tread, will disap
                    mt_l = np.sum(MT.seg_dist)#total length of mt
                    disap_t = mt_l/(MT.vt)+MT.tread_t #collision distances from 1 to 2, growing can only collide w/ shrinking
                    event_list_R1.append(event(MT,mt_none,disap_t,None,'disap_tread',calc_t=t))
                #I grouped events weirdly below, makes sense though
                if policy in ['zipper','freed_br','freed_rescue']:
                    #check for NEW branch crossovers
                    bdl2.new_branch_events(event_list_R1, MT_list, pt, t, bdl_list, branch_list) #new branch on bdl2, all other mts need to know
                    if policy == 'zipper': #other bdl has other branches potentially
                        bdl.new_branch_events(event_list_R1, MT_list, pt, t, bdl_list, branch_list, mt_not = mt_br) #new branch on bdl
                        bdl.cross_bdl_event(mt_br, event_list_R1) #check for newly branched mt's intersection w/ crossovers on bdl #TM
                if not bdl.pseudo_bdry: #don't care about bdl traffic if bdl is bdry
                    bdl.overtake_bdl_event(mt_br, pt, t, MT_list, event_list_R1) #TM
                if policy in ['zipper','follow_br']: #follow existing deflections
                    bdl.deflect_bdl_event(event_list_R1,MT_list,bdl_list,region_list,t,mt_br,pt) #check for mt_br to follow vertex
                    if policy == 'follow_br':
                        bdl.branch_bdl_event(mt_br,event_list_R1, branch_list) #check for mt_br hitting branch pt
                        bdl.cross_bdl_event(mt_br, event_list_R1) #check for mt_br hitting cross pt
                        # global_prop.free += 1 #keep track of these
                else: #free_br
                    #bdry events
                    bdry_res = inter_r_bdry2(MT_list[new_idx], MT_list, bdl_list, region_list, free = True) #find intersection info
                    next_time = bdry_res[0] + MT.update_t[-1] #time of collision
                    event_list_R1.append(event(MT_list[new_idx],mt_none,next_time,bdry_res[1],bdry_res[2],calc_t=t))
                if policy == 'freed_rescue' or pevent.prev in ['rescue', 'pause_to_grow']: #edit stochastic stuff
                    global_prop.grow(mt_br, MT_list, new_idx, pevent) #update growing pop
                    #no need to move within growing list since it's not switching partitions
                    # global_prop.free += 1 #keep track of these
                else:
                    global_prop.update_grow_mt(mt_br, MT_list, new_idx, MT.angle[-1],t)
                #ANGULAR CAT
                #uses angle from previous mt
            elif policy in ['entrain_spaced', 'entrain_other_region']:
                assert no_bdl_id
                # assert MT.hit_bdry and not MT.grow
                # bdl = bdl_list[MT.bdl]
                if policy == 'entrain_other_region': #mts of interest aren't actually the same mts as in the other case
                    event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != last_result and x.mt2_n != last_result)\
                                      or (x.policy in ['nucleate','sp_catastrophe','rescue'])]#discard events for old mt prior to renaming stuff
                    MT = MT_list[MT.prev_mt]
                    mt2 = MT_list[MT.ext_mt]
                    assert mt2.number != last_result
                    last_result = MT.number
                    bdl1 = bdl_list[MT.bdl]
                    bdl2 = bdl_list[mt2.bdl]
                    assert sort_regions[0] != MT.region
                    sort_regions.append(MT.region)
                old_idx = last_result #index of previous MT
                new_idx = MT.ext_mt #index of new MT
                mt_br = MT_list[new_idx] #new mt = mt2
                bdl = bdl_list[mt_br.bdl] #bdl of branched mt
                bdl3 = bdl2 #confusing but I rename bdl2 to be something else below, still need this variable
                bdl2 = bdl_list[MT.bdl] #bdl of original mt BE CAREFUL, SAME AS bdl1 if zipper
                event_list_R1 = event_list[MT.region]
                #mts to compare w/ branched mt
                branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
                mt_index1= [mt.number for mt in MT_list1 if (mt.exist and (mt.number != new_idx) \
                and (mt_br.region==mt.region) and mt.number != last_result) and (mt_br.free or mt.free) \
                and (mt.number not in bdl.mts) and (mt.bdl not in bdl.cross_bdl+branch_bdl)]
                pair_list = list(it.product(mt_index1,[new_idx])) #generate new pairs
                #mts to compare w/ original mt
                branch_bdl2 = [branch_list[brn].branch_bdln for group in bdl2.branchn for brn in group if brn != 0]
                mt_index2= [mt.number for mt in MT_list1 if (mt.exist and (mt.number != last_result) \
                and (MT.region==mt.region) and mt.number != new_idx) and (MT.free or mt.free)\
                and (mt.number not in bdl2.mts) and (mt.bdl not in bdl2.cross_bdl+branch_bdl2)] #idxs for original mt
                pair_list2 = list(it.product(mt_index2,[last_result])) #generate new pairs
                #combined pair list
                pair_list = pair_list+pair_list2
                l = len(pair_list)
                event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != old_idx and x.mt2_n != old_idx)\
                                  or (x.policy in ['nucleate','sp_catastrophe','rescue'])
                                  or (x.mt1_n == old_idx and x.policy=='uncross_m')\
                                  or (x.mt1_n != old_idx and x.policy in ['1catch2_m'])]#exclude 1catch2?
                if MT.tread: #if tread, will disap
                    mt_l = np.sum(MT.seg_dist)#total length of mt
                    disap_t = mt_l/(MT.vt)+MT.tread_t #collision distances from 1 to 2, growing can only collide w/ shrinking
                    event_list_R1.append(event(MT,mt_none,disap_t,None,'disap_tread',calc_t=t))
                #I grouped events weirdly below, makes sense though
                # bdl2.new_branch_events(event_list_R1, MT_list, pt, t, bdl_list, branch_list) #new branch on bdl2, all other mts need to know
                if not bdl.pseudo_bdry: #don't care about bdl traffic if bdl is bdry
                    bdl.overtake_bdl_event(mt_br, pt, t, MT_list, event_list_R1) #TM
                bdry_res = inter_r_bdry2(MT_list[new_idx], MT_list, bdl_list, region_list, free = True) #find intersection info
                next_time = bdry_res[0] + MT.update_t[-1] #time of collision
                event_list_R1.append(event(MT_list[new_idx],mt_none,next_time,bdry_res[1],bdry_res[2],calc_t=t))
                global_prop.update_grow_mt(mt_br, MT_list, new_idx, MT.angle[-1],t)
        elif policy in ['grow_to_pause', 'shrink_to_pause']:
            MT = MT_list[l_idx]
            sort_regions = [MT.region]
            event_list_R1 = event_list[MT.region]
            MT_list1 = mt_sublist[MT.region]
            assert MT.hit_bdry and not MT.grow
            bdl = bdl_list[MT.bdl]
            #mts to compare w/ original mt
            branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
            mt_index= [mt.number for mt in MT_list1 if (mt.exist and (mt.number != last_result) \
            and (MT.region==mt.region)) and (MT.free or mt.free)\
            and (mt.number not in bdl.mts) and (mt.bdl not in bdl.cross_bdl+branch_bdl)] #idxs for original mt
            pair_list = list(it.product(mt_index,[last_result])) #generate new pairs
            l = len(pair_list)
            event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != l_idx and x.mt2_n !=l_idx)\
                               or (x.mt1_n == MT.number and x.policy=='uncross_m')\
                               # or (x.mt2_n != MT.number and x.policy == '1catch2')
                               or (x.mt1_n != MT.number and x.policy == '1catch2_m')]#discard all invalid events
            if MT.tread: #if tread, will disap
                mt_l = np.sum(MT.seg_dist)#total length of mt
                disap_t = mt_l/(MT.vt)+MT.tread_t
                event_list_R1.append(event(MT,mt_none,disap_t,None,'disap',calc_t=t))
            bdl.overtake_bdl_event(MT, pt, t, MT_list, event_list_R1)
            global_prop.pause(MT, MT_list, last_result, pevent)
        elif policy in ['cross_bdl','cross_br','1catch2','1catch2_m','freed_not_free']:
            MT = MT_list[last_result]
            sort_regions = [MT.region]
            event_list_R1 = event_list[MT.region]
            #TODO need to find
            del event_list_R1[0] #only delete last event, no new events to be found
            if policy == 'freed_not_free': #need to make potentially new collision calculations NOT bdl calculations
                event_list_R1[:] = [x for x in event_list_R1 if not ((x.mt1_n == last_result or x.mt2_n == last_result) and x.policy in ['1hit2','2hit1'])]#all mtn sim mt intersections will be recalculated
                MT_list1 = mt_sublist[MT.region]
                bdl = bdl_list[MT.bdl]
                rr = [brn for group in bdl.branchn for brn in group if brn != 0]
                assert None not in rr 
                branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
                mt_index = [mt.number for mt in MT_list1 if (mt.exist and (mt.number != last_result) \
                and (MT.region==mt.region)) and (MT.free or mt.free) \
                and (mt.number not in bdl.mts) and (mt.bdl not in bdl.cross_bdl+branch_bdl)] #in addition to the usual exclusions
                n = len(mt_index)
                pair_list = list(it.product(mt_index,[last_result])) #generate new pairs
                l = len(pair_list)
        elif policy in ['uncross', 'uncross_m','uncross_all']:
            #do uncrossing here
            MT = MT_list[last_result]
            sort_regions = [MT.region]
            event_list_R1 = event_list[MT.region]
            del event_list_R1[0] #get rid of last event
            if policy =='uncross_all': #no longer crossed
                event_list_R1[:] = [x for x in event_list_R1 if x.pt!= pevent.pt] #get rid of all events including this pt
                #need to recalculate intersections of mts on bdl1 w/ bdl2, no longer connected
                bdl1, bdl2 = bdl_list[pevent.bdl1], bdl_list[pevent.bdl2]
                mts1 = [x for x in bdl1.mts if x != last_result] #mts on bdl1 exluding newest
                pair_list = list(it.product(bdl2.mts,mts1)) #pair list
                l = len(pair_list)
            #no mts physically changed, so no recalculations needed
        elif policy == 'unbound_nucleation':#nucleation occurs
            new_idx = [last_result] #index of updated MTs
            MT = MT_list[last_result]
            sort_regions = [MT.region]
            MT_list1 = mt_sublist[MT.region]
            mt_index= [mt.number for mt in MT_list1 if (MT.region==mt.region) and mt.number!=last_result and (MT.free or mt.free)]
            n = len(mt_index)
            pair_list = list(it.product(mt_index,new_idx)) #generate new pairs
            l = len(pair_list)
            #bdry events
            MT = MT_list[l_idx]
            bdry_res = inter_r_bdry2(MT, MT_list, bdl_list, region_list, free = True) #find intersection info
            next_time = bdry_res[0] + MT.update_t[-1] #time of collision
            event_list[MT.region].append(event(MT,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t=t))
            global_prop.nucleate(MT,last_result) #update pop info
        elif policy == 'no_nucleation':
            #nucleation complex dissociated
            global_prop.stochastic_update = True #this should deal w/ deleting this event from list
            # sys.exit()
        elif policy == 'parallel_nucleation':
            #basically a "grow" and possibly "tread_disap" thing
            MT = MT_list[last_result]
            sort_regions = [MT.region]
            event_list_R1 = event_list[MT.region]
            event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != last_result and x.mt2_n != last_result) or x.policy in ['1catch2_m','uncross_m']]#discard all invalid events
            MT_list1 = mt_sublist[MT.region]
            bdl = bdl_list[MT.bdl]
            bdl.branch_bdl_event(MT, event_list_R1, branch_list)
            bdl.cross_bdl_event(MT, event_list_R1) #might cross bdl crossings
            bdl.overtake_bdl_event(MT, pt, t, MT_list, event_list_R1, recalc_grow = True) #edit overtake events
            if MT.tread:
                bdl.overtake_bdl_event(MT, pt, t, MT_list, event_list_R1) #re-call to calc for minus end stuff
                bdl.uncross_bdl_event(MT, pevent, event_list_R1, bdl_list) #might uncross on bdl
            bdl.deflect_bdl_event(event_list_R1,MT_list,bdl_list,region_list,t,MT,pt) #follow existing stuff
            # bdl.uncross_bdl_event(MT, pevent, event_list, bdl_list)
            branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
            mt_index = [mt.number for mt in MT_list1 if (mt.exist and (mt.number != last_result) \
            and (MT.region==mt.region)) and (MT.free or mt.free) \
            and (mt.number not in bdl.mts) and (mt.bdl not in bdl.cross_bdl+branch_bdl)] #in addition to the usual exclusions
            n = len(mt_index)
            pair_list = list(it.product(mt_index,[last_result])) #generate new pairs
            l = len(pair_list)
            global_prop.nucleate(MT,last_result) #update pop info
        elif policy == "branched_nucleation":
            new_idx = last_result
            MT = MT_list[l_idx] #TODO BE CAREFUL, MT != mt1?
            MT_list1 = mt_sublist[MT.region]
            # elif MT.grow is False and not MT.hit_bdry: #if updated MT is due to cat. and it didn't disap
            event_list_R1 = event_list[MT.region]
            sort_regions = [MT.region]
            mt_br = MT
            bdl = bdl_list[pevent.bdl2] #bdl of branched mt
            bdl2 = bdl_list[pevent.bdl1] #bdl of original mt BE CAREFUL, SAME AS bdl1 if zipper
            event_list_R1 = event_list[MT.region]
            #mts to compare w/ branched mt
            branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
            mt_index1= [mt.number for mt in MT_list1 if (mt.exist and (mt.number != l_idx) \
            and (mt_br.region==mt.region)) and (mt_br.free or mt.free) and (mt.bdl not in branch_bdl)]
            pair_list = list(it.product(mt_index1,[new_idx])) #generate new pairs
            #mts to compare w/ original mt
            # branch_bdl2 = [branch_list[brn].branch_bdln for group in bdl2.branchn for brn in group if brn != 0]
            # mt_index2= [mt.number for mt in MT_list1 if (mt.exist and (mt.number != last_result) \
            # and (MT.region==mt.region) and mt.number != new_idx) and (MT.free or mt.free)\
            # and (mt.number not in bdl2.mts) and (mt.bdl not in bdl2.cross_bdl+branch_bdl2)] #idxs for original mt
            # pair_list2 = list(it.product(mt_index2,[last_result])) #generate new pairs
            #combined pair list
            pair_list = pair_list #+pair_list2
            l = len(pair_list)
            #I grouped events weirdly below, makes sense though
            bdl2.new_branch_events(event_list_R1, MT_list, pt, t, bdl_list, branch_list) #new branch on bdl2, all other mts need to know
            bdry_res = inter_r_bdry2(MT, MT_list, bdl_list, region_list, free = True) #find intersection info
            next_time = bdry_res[0] + MT.update_t[-1] #time of collision
            event_list_R1.append(event(MT,mt_none,next_time,bdry_res[1],bdry_res[2],calc_t=t))
            global_prop.nucleate(MT,last_result) #update pop info
        elif policy in ['sp_catastrophe','edge_cat', 'pause_to_shrink']:
            MT = MT_list[last_result]
            MT_list1 = mt_sublist[MT.region]
            sort_regions = [MT.region]
            event_list_R1 = event_list[MT.region]
            #should be equivalent to induced cat
            event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != last_result and x.mt2_n != last_result)\
                             or (x.policy=='1catch2_m' and x.mt2_n == last_result) or x.policy=='uncross_m']#discard all invalid events
            bdl = bdl_list[MT.bdl]
            bdl.uncross_bdl_event(MT, pevent, event_list_R1, bdl_list) #uncross event
            bdl.overtake_bdl_event(MT, pt, t, MT_list, event_list_R1) #edit overtake events
            branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
            mt_index = [mt.number for mt in MT_list1 if (mt.exist and (mt.number != last_result) \
            and (mt.grow) and (MT.region==mt.region)) and (MT.free or mt.free)\
            and (mt.number not in bdl.mts) and (mt.bdl not in bdl.cross_bdl+branch_bdl)] #in addition to the usual exclusions
            n = len(mt_index)
            pair_list = list(it.product(mt_index,[last_result])) #generate new pairs
            l = len(pair_list)
            #disap event
            mt_l = np.sum(MT.seg_dist)#total length of shrinking mt, cannot got lower than this
            if MT.tread:
                mt_l -= MT.vt*(t-MT.tread_t) #left over len from tread
                disap_t = mt_l/(v_s+MT.vt)+t #collision distances from 1 to 2, growing can only collide w/ shrinking
                event_list_R1.append(event(MT,mt_none,disap_t,None,'disap', calc_t=t))
            else:
                disap_t = mt_l/v_s+t #collision distances from 1 to 2, growing can only collide w/ shrinking
                event_list_R1.append(event(MT,mt_none,disap_t,None,'disap', calc_t=t))
            global_prop.shrunk(MT, MT_list, last_result, pevent) #update shrink pop
        elif policy in ['rescue', 'pause_to_grow']:
            MT = MT_list[last_result]
            sort_regions = [MT.region]
            event_list_R1 = event_list[MT.region]
            event_list_R1[:] = [x for x in event_list_R1 if (x.mt1_n != last_result and x.mt2_n != last_result) or x.policy in ['1catch2_m','uncross_m']]#discard all invalid events
            MT_list1 = mt_sublist[MT.region]
            bdl = bdl_list[MT.bdl]
            bdl.branch_bdl_event(MT, event_list_R1, branch_list)
            bdl.cross_bdl_event(MT, event_list_R1) #might cross bdl crossings
            bdl.overtake_bdl_event(MT, pt, t, MT_list, event_list_R1) #edit overtake events
            bdl.deflect_bdl_event(event_list_R1,MT_list,bdl_list,region_list,t,MT,pt) #follow existing stuff
            # bdl.uncross_bdl_event(MT, pevent, event_list, bdl_list)
            branch_bdl = [branch_list[brn].branch_bdln for group in bdl.branchn for brn in group if brn != 0]
            mt_index = [mt.number for mt in MT_list1 if (mt.exist and (mt.number != last_result) \
            and (MT.region==mt.region)) and (MT.free or mt.free) \
            and (mt.number not in bdl.mts) and (mt.bdl not in bdl.cross_bdl+branch_bdl)] #in addition to the usual exclusions
            n = len(mt_index)
            pair_list = list(it.product(mt_index,[last_result])) #generate new pairs
            l = len(pair_list)
            global_prop.grow(MT, MT_list, last_result, pevent) #update growing pop
            #ANGULAR CAT
            #uses angle from itself
    if global_prop.stochastic_update: #system change requires recalculation of stochastic event
        sort_regions.append(-1) #will need to resort stochastic list which has index -1
        event_list_S = event_list[-1]
        event_list_S[:] = [x for x in event_list_S if x.policy not in ['nucleate','sp_catastrophe','rescue','freed_rescue', 'pause_to_shrink',\
                                                                   'pause_to_grow', 'grow_to_pause', 'shrink_to_pause',\
                                                                       'cross_br', 'follow_br', 'freed_br', 'freed_not_free',\
                                                                    'parallel_nucleation', 'no_nucleation', 'branched_nucleation', 'unbound_nucleation']]
        #need to take into account that some events will have their policies changed to non-stochastic ones
        stoch_res = stochastic(t,global_prop) #generate
        stoch_policy, T, pt, mt_no = stoch_res[0], stoch_res[1], stoch_res[2], stoch_res[3] #results
        if mt_no is None: #if nucleation, no mt object for fn
            event_list_S.append(event(mt_none,mt_none,T,pt,stoch_policy,calc_t = t))
        else:
            event_list_S.append(event(MT_list[mt_no],mt_none,T,pt,stoch_policy,calc_t = t))
        global_prop.stochastic_update = False
    if l is not None: #not crossover, things change
        for i in range(l): #compare all pairs
            mt1_i = pair_list[i][0] #numbers of mts to compare
            mt2_i = pair_list[i][1]
            l_mt1, l_mt2 = mt1_i,mt2_i#l_mt[0],l_mt[1]
            result = compare(MT_list[l_mt1], MT_list[l_mt2],t,region_list) #retrieve output object
            check = True #to check for parallel collisions
            if (result.policy not in ['no_collision']) and check: #if there is collision
                assert MT_list[mt1_i].region == MT_list[mt2_i].region
                assert result.dist >=0 and MT_list[l_mt1].exist and MT_list[l_mt2].exist
                if result.point[0] < xdomain[1] and result.point[0] > xdomain[0] \
                and result.point[1] < ydomain[1] and result.point[1] > ydomain[0]: #must be inside domain
                    R = MT_list[mt1_i].region #XXX
                    next_time = result.dist+t
                    event_list[R].append(event(MT_list[mt1_i],MT_list[mt2_i],next_time,result.point,result.policy,result.idx,calc_t = t)) #XXX
    #TODO need to figure out how many lists I need to resort
    # event_list.sort(key=lambda x: x.t) #sort according to time
    res = sort_and_find_events(event_list,sort_regions)
    return(res)
    # return(event_list[0])


def update(mt_list,mt_sublist,bdl_list,branch_list,region_list,event_list,event,global_prop):
    '''
    Updates MT based on next event: edits mt,bdl,region_taj classes within each list.

    Parameters
    ----------
    mt_list : list of MTs
    collided : bool of whether MTs collided (True) or it was a bdry event
    policy : what type of collision event
    dists : event time
    pt : point of intersection
    idx : tuple of idices if two MTs collide, single index if bdry event
    t : current time

    Returns
    -------
    Index of updated MT

    '''
    if edge_cat:
        if event.policy in ['top','bottom']: #for edge catastrophes, convert these events to cat
            if event.pt[1] in ydomain:
                event.policy = 'edge_cat'
    #unpack event info
    idxs = None
    if event.mt2_n is None: #if there are one or multiple mts involved
        idxs = event.mt1_n
    else:
        idxs = [event.mt1_n,event.mt2_n]
    policy = event.policy
    dists = event.t
    pt = event.pt
    col_idx = event.mt_idx
    #for rescue, mts might be at branch pt -- need to change event to follow_br in this case
    rescue_branch = False #special bool for this case
    if policy in ['pause_to_grow']: #TODO need bdl2 number?
        temp_mt = mt_list[idxs] #to find growing tip mt
        new_no = idxs
        while temp_mt.ext_mt is not None:
            new_no = temp_mt.ext_mt
            temp_mt = mt_list[new_no]
        bdl1 = bdl_list[temp_mt.bdl]
        if policy == 'pause_to_grow':
            event.pt = mt_list[new_no].seg[-1]
            pt = event.pt
        if pt in bdl1.branch_pt:
            rescue_branch = True
            event.mt1_n = new_no
            event.bdl1 = bdl1.number
            event.policy = 'cross_br'
            policy = 'cross_br'
            idxs = new_no
    elif policy in ['1catch2','1catch2_m']:
        mt_idx1, mt_idx2 = idxs[0],idxs[1] #overlaps with cross_br event in non-geodesic case
        mt1 = mt_list[mt_idx1] #get mts
        bdl1 = bdl_list[event.bdl1]
        if pt in bdl1.branch_pt:
            event.policy = 'cross_br'
            policy = 'cross_br'
            idxs = mt_idx1
    # print(event.policy)
    # if collided is True: #if dynamic ends collided
    if policy =='1hit2' or policy=='2hit1':
        mt_idx1, mt_idx2 = idxs[0],idxs[1]
        if policy == '2hit1':
            mt_idx1,mt_idx2 = idxs[1],idxs[0]
        mt1 = mt_list[mt_idx1] #get mts
        mt2 = mt_list[mt_idx2]
        assert mt1.grow
        seg_idx = col_idx # which_seg(mt1,mt2,t)
        r = 1
        if cat_on:
            r = rnd.randint(0,1) #can also use for bundle collision
        # angle_traj = mt2.traj[seg_idx]
        # angle2 = region_list[mt2.region].angle[angle_traj]
        #mt2.angle[seg_idx]
        zip_res = zip_cat(mt1.angle[-1],mt2.angle[seg_idx],pt,mt1.seg[-1],r, decision_fn=ABLATION) #determine collision geometry (ABLATION hook, S1)
        #TODO use traj angle for increased accuracy?
        new_pt,new_angle = zip_res[1], zip_res[0]
        resolve = zip_res[2]
        event.policy = resolve #specify the event result
        # else: #if it's not a bdry mt, record length of mt that got hit
        #     global_prop.record_len(mt2, dists, True, mt_list, bdl_list)
        if not cat_on: #option for turning off catastrophe
            if resolve == 'catas':
                resolve = 'cross'
                event.policy = 'cross'
        if not bdl_on:
            if resolve in ['zipper+','zipper-']:
                resolve = 'cross'
                event.policy = 'cross'
        if resolve == 'cross' and cat_on and bdl_on:#crossover
            # mt1.add_vertex(new_pt, new_angle, dists, event.policy) #add vertex - causes numerical instability
            if not no_bdl_id: #TODO double check this
                bdl1 = None #declare first bdl
                if mt1.bdl is None: #bdl does not exist ERASE
                    bdl_list.append(bundle(mt1, bdl_list, Policy = policy)) #create new bdl w/ mt1
                    bdl1 = bdl_list[-1]
                else:
                    bdl1 = bdl_list[mt1.bdl]
                bdl2 = None #declare
                if mt2.bdl is not None: #mt2 is already a bundle
                    bdl2 = bdl_list[mt2.bdl] #declare bdl2
                else: #need to create new bdl ERASE
                    bdl_list.append(bundle(mt2, bdl_list, Policy = policy)) #create new bdl w/ mt1
                    bdl2 = bdl_list[-1]
                bdl1.add_cross_bdl(new_pt, bdl2) #add crossover info for both bundles
                bdl2_mts = bdl2.mt_overlap(mt_list,new_pt,dists,mt2_n = mt_idx2)
                # bdl2_mts.append(mt2.number)
                bdl1.cross_mts.append(bdl2_mts) #add overlaping mts
                #possibly many overlaps mt1
                bdl2.add_cross_bdl(new_pt, bdl1)
                bdl2.cross_mts.append([mt1.number]) #only mt1 crossed mt2
            global_prop.cross += 1
            global_prop.record_len(mt1, dists)
        elif resolve =='catas': #catastrophe
            mt1.add_vertex(new_pt, new_angle, dists, event.policy, grow=False) #add vertex, no need to add angle
            mt1.grow = False
            mt1.free = False
            global_prop.record_len(mt1, dists)
        elif resolve in ['zipper+','zipper-']:
            if no_bdl_id: #physical bundling
                bdl1 = bdl_list[mt1.bdl]
                if (new_pt[1] > ydomain[1] or new_pt[1] < ydomain[0]) and edge_cat:
                    event.policy = 'catas'
                    mt1.add_vertex(new_pt, mt1.angle[-1], dists, event.policy, grow=False) #add vertex, no need to add angle
                    mt1.grow = False
                    mt1.free = False
                    global_prop.record_len(mt1, dists)
                else:
                    #for x_max, x_min case, the "not region" case handles it
                    pt_region = which_region(new_pt)
                    mt_from_bdry = False #whether it's from bdry
                    if mt1.from_bdry_2:
                        if mt1.prev_mt != None:
                            if mt_list[mt1.prev_mt].exist:
                                mt_from_bdry = True
                    if pt_region == mt1.region or (pt_region != mt1.region and not mt_from_bdry):
                        stuck = mt1.check_if_stuck(new_pt) #check if it is in between a fork position
                        if stuck:
                            event.policy = 'catas'
                            mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy, grow=False) #add vertex, no need to add angle
                            mt1.grow = False
                            mt1.free = False
                            global_prop.record_len(mt1, dists)
                        else:
                            inverted = bdl1.check_step_back(mt1, dists, new_pt) #check if MT gets inverted
                            if inverted:
                                step_distance = dist(new_pt,pt) #find if it's because it's close to parallel or not
                                if step_distance > dr_tol: #if it's really long, might as well entrain anyway
                                    new_pt = bdl1.step_back(mt1, dists, new_pt, pt, inverted=True) #get new entrainment pt
                                    #copy and pasted from freed_br, but with new point and angle
                                    event.policy = 'entrain_spaced'
                                    event.pt = new_pt
                                    mt1.add_vertex(new_pt, mt1.angle[-1], dists, event.policy, double_check = True) #add vertex
                                    mt1.grow = False #no longer considered growing
                                    mt1.hit_bdry = True #hit bdry
                                    mt1.free = False
                                    #for potential calculations
                                    mt1.prev_t = mt1.update_t[-1]
                                    global_prop.free += 1 #keep track of these
                                    #new mt
                                    mt_idx = mt_list[-1].number+1 #new mt number
                                    mt1.ext_mt = mt_idx #let it know which one is its continuation
                                    mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                                    mt_br = mt_list[-1] #branched MT
                                    mt_br.init_angle = new_angle
                                    angle = mt_br.next_path(new_angle, del_l2, global_prop) #if vertical, returns same angle w/ inf tip length
                                    traj_n = region_list[mt1.region].add_traj(new_pt, angle, mt_br,bdl_list)
                                    mt_br.entrain(mt1, new_pt, angle, traj_n)
                                    mt_br.free = True
                                    # mt_br.init_angle = mt1.angle[-1]
                                    bdl_list.append(bundle(mt_br, bdl_list, Policy = event.policy)) #create new bdl w/ mt2 TODO IT'S FREE!
                                    bdl2 = bdl_list[-1] #cannot use add_branch to bdl1 because branch already 'exists', must edit manually
                                    mt_sublist[mt_br.region].append(mt_br)
                                else:
                                    event.policy = 'catas'
                                    mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy, grow=False) #add vertex, no need to add angle
                                    mt1.grow = False
                                    mt1.free = False
                                    global_prop.record_len(mt1, dists)
                            else:
                                new_pt = bdl1.step_back(mt1, dists, new_pt, pt, inverted=False)
                                #copy and pasted from freed_br, but with new point and angle
                                event.policy = 'entrain_spaced'
                                event.pt = new_pt
                                mt1.add_vertex(new_pt, mt1.angle[-1], dists, event.policy, double_check = rescue_branch) #add vertex
                                mt1.grow = False #no longer considered growing
                                mt1.hit_bdry = True #hit bdry
                                mt1.free = False
                                #for potential calculations
                                mt1.prev_t = mt1.update_t[-1]
                                #new mt
                                mt_idx = mt_list[-1].number+1 #new mt number
                                mt1.ext_mt = mt_idx #let it know which one is its continuation
                                mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                                mt_br = mt_list[-1] #branched MT
                                global_prop.free += 1 #keep track of these
                                mt_br.init_angle = new_angle
                                angle = mt_br.next_path(new_angle, del_l2, global_prop) #if vertical, returns same angle w/ inf tip length
                                traj_n = region_list[mt1.region].add_traj(new_pt, angle, mt_br,bdl_list)
                                mt_br.entrain(mt1, new_pt, angle, traj_n)
                                mt_br.free = True
                                # mt_br.init_angle = mt1.angle[-1]
                                bdl_list.append(bundle(mt_br, bdl_list, Policy = event.policy)) #create new bdl w/ mt2 TODO IT'S FREE!
                                mt_sublist[mt_br.region].append(mt_br)
                    else:
                        #steps back into other region. Can go even further into another region, but we stop here for simplicity
                        #be careful! mt_prev is of interest here, not mt1! Different regions!!!
                        assert mt1.from_bdry_2
                        mt_prev = mt_list[mt1.prev_mt]
                        bdl_prev = bdl_list[mt_prev.bdl]
                        new_pt2 = mt1.check_pt_topology(new_pt)
                        stuck = mt_prev.check_if_stuck(new_pt2, other_region = True) #check if it is in between a fork position
                        if stuck:
                            event.policy = 'catas'
                            mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy, grow=False) #add vertex, no need to add angle
                            mt1.grow = False
                            mt1.free = False
                            global_prop.record_len(mt1, dists)
                        else:
                            
                            inverted = bdl_prev.check_step_back(mt_prev, dists, new_pt)
                            # if dists == 3.772562961644761:
                            #     print(inverted, bdl_prev.seg, new_pt, new_pt2)
                            #     sys.exit()
                            if inverted:
                                step_distance = dist(new_pt,pt) #find if it's because it's close to parallel or not
                                if step_distance > dr_tol: #if it's really long, might as well entrain anyway
                                    #now we use the bdry pt instead of the actual collision pt, since we want it to be inside the other region!
                                    new_pt = bdl_prev.step_back(mt_prev, dists, new_pt2, mt_prev.seg[-1], inverted=True) #get new entrainment pt
                                    event.policy = 'entrain_other_region'
                                    event.pt = new_pt
                                    # mt_prev.add_vertex(new_pt, mt1.angle[-1], dists, event.policy, double_check = rescue_branch) #add vertex
                                    mt_prev.step_back_seg(new_pt, dists)
                                    mt_prev.grow = False #no longer considered growing
                                    mt_prev.hit_bdry = True #hit bdry
                                    mt_prev.free = False
                                    #for potential calculations
                                    mt_prev.prev_t = mt1.update_t[-1]
                                    #new mt
                                    mt_idx = mt_list[-1].number+1 #new mt number
                                    mt_prev.ext_mt = mt_idx #let it know which one is its continuation
                                    mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                                    mt_br = mt_list[-1] #branched MT
                                    global_prop.free += 1 #keep track of these
                                    mt_br.init_angle = new_angle
                                    angle = mt_br.next_path(new_angle, del_l2, global_prop) #if vertical, returns same angle w/ inf tip length
                                    traj_n = region_list[mt_prev.region].add_traj(new_pt, angle, mt_br,bdl_list)
                                    mt_br.entrain(mt_prev, new_pt, angle, traj_n)
                                    mt_br.free = True
                                    # mt_br.init_angle = mt1.angle[-1]
                                    mt_sublist[mt_br.region].append(mt_br)
                                    bdl_list.append(bundle(mt_br, bdl_list, Policy = event.policy)) #create new bdl w/ mt2 TODO IT'S FREE!
                                    #delete old MT and its old crossovers
                                    for i, cross_pt in enumerate(bdl1.cross_pt):
                                        bdl_cross = bdl_list[bdl1.cross_bdl[i]]
                                        idx = bdl_cross.cross_bdl.index(bdl1.number)
                                        if mt1.number in bdl_cross.cross_mts[idx]: #check it mt1 is in any of these cross pts
                                            bdl_cross.del_cross(mt1,cross_pt) #if so, delete
                                    mt1.grow = False
                                    mt1.exist = False
                                    bdl1.del_mt(mt1)
                                else:
                                    event.policy = 'catas'
                                    mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy, grow=False) #add vertex, no need to add angle
                                    mt1.grow = False
                                    mt1.free = False
                                    global_prop.record_len(mt1, dists)
                            else:
                                new_pt = bdl_prev.step_back(mt_prev, dists, new_pt2, mt_prev.seg[-1], inverted=False) #get new entrainment pt
                                event.policy = 'entrain_other_region'
                                event.pt = new_pt
                                # sys.exit()
                                # mt_prev.add_vertex(new_pt, mt1.angle[-1], dists, event.policy, double_check = rescue_branch) #add vertex
                                mt_prev.step_back_seg(new_pt, dists)
                                mt_prev.grow = False #no longer considered growing
                                mt_prev.hit_bdry = True #hit bdry
                                mt_prev.free = False
                                #for potential calculations
                                mt_prev.prev_t = mt1.update_t[-1]
                                #new mt
                                mt_idx = mt_list[-1].number+1 #new mt number
                                mt_prev.ext_mt = mt_idx #let it know which one is its continuation
                                mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                                mt_br = mt_list[-1] #branched MT
                                global_prop.free += 1 #keep track of these
                                mt_br.init_angle = new_angle
                                angle = mt_br.next_path(new_angle, del_l2, global_prop) #if vertical, returns same angle w/ inf tip length
                                traj_n = region_list[mt_prev.region].add_traj(new_pt, angle, mt_br,bdl_list)
                                mt_br.entrain(mt_prev, new_pt, angle, traj_n)
                                mt_br.free = True
                                # mt_br.init_angle = mt1.angle[-1]
                                mt_sublist[mt_br.region].append(mt_br)
                                bdl_list.append(bundle(mt_br, bdl_list, Policy = event.policy)) #create new bdl w/ mt2 TODO IT'S FREE!
                                #delete old MT and its old crossovers
                                for i, cross_pt in enumerate(bdl1.cross_pt):
                                    bdl_cross = bdl_list[bdl1.cross_bdl[i]]
                                    idx = bdl_cross.cross_bdl.index(bdl1.number)
                                    if mt1.number in bdl_cross.cross_mts[idx]: #check it mt1 is in any of these cross pts
                                        bdl_cross.del_cross(mt1,cross_pt) #if so, delete
                                mt1.grow = False
                                mt1.exist = False
                                bdl1.del_mt(mt1)
            else:
                event.policy = 'zipper'
                mt1.add_vertex(new_pt, mt1.angle[-1], dists, event.policy) #add vertex
                mt1.grow = False #no longer considered growing
                mt1.hit_bdry = True #hit bdry
                mt1.free = False
                global_prop.record_len(mt1, dists)
                #for potential calculations
                mt1.prev_t = mt1.update_t[-1]
                #Branched nucleation
                mt_idx = mt_list[-1].number+1 #new border mt number
                mt1.ext_mt = mt_idx #let it know which one is its continuation
                mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                mt_br = mt_list[-1] #branched MT
                mt_br.entrain(mt1, new_pt, new_angle, mt2.traj[seg_idx]) #add entrainment info
                if not deflect_on:
                    mt_br.carry_path(mt1)
                mt_sublist[mt_br.region].append(mt_br)
                #bundle business
                bdl2 = bdl_list[mt2.bdl] #TODO determine branch geometry and add branch structure
                bdl2.add_mt(mt_br) #add branched mt to mt2 bdl
                bdl1 = bdl_list[mt1.bdl]
                bdl1.add_branch_bdl(pt, bdl2, mt1, mt_br, mt_br.angle[-1], False, branch_list, mt_list, region_list,bdl_list, dists, natural = False, origin = True) #bdl1 has branch outward
                bdl2.add_branch_bdl(pt, bdl1, mt_br, mt1, mt1.angle[-1], True, branch_list, mt_list, region_list,bdl_list, dists, natural = False, origin = False) #bdl2 has inward branch
                bdl2.mt_sides.append(branch_list[-1].level) #above method does not add mt sidedness
                assert len(bdl2.mt_sides) == len(bdl2.mts)
                assert len(bdl1.branch_pos) == len(bdl1.branch_pt) and len(bdl2.branch_pos) == len(bdl2.branch_pt)
        return(mt_idx1)
    elif policy == 'cross_bdl': #crossing within a bdl
        bdl1 = bdl_list[event.bdl1]
        mt1 = mt_list[event.mt1_n]
        cross_idx = bdl1.cross_pt.index(pt) #get crossover index
        bdl2 = bdl_list[bdl1.cross_bdl[cross_idx]] #barrier bdl id
        cross_mt = bdl1.cross_mts[cross_idx] #mts on on barrier bdl in the way
        r = 1 #declare
        if len(cross_mt) != 0 and cat_on: #if there's mt in the way
            r = rnd.randint(0,1) #use for bundle collision
        # assert len(cross_mt) > 0 #should be at least one, event does not exist if there are none
        resolve = 'catas'
        if r > 0 :
            resolve = 'cross_bdl'
        event.policy = resolve #specify event result
        event.bdl2 = bdl2.number #add bdl2 to event info, not used for catas but needed for fn arg
        if resolve == 'cross_bdl':
            # mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy) #add vertex - causes numerical instability
            #possibly many overlaps mt1
            # if not no_bdl_id: #TODO double check this
            cross_idx2 = bdl2.cross_pt.index(pt)
            bdl2.cross_mts[cross_idx2].append(mt1.number) #add mt1 crossover bdl2
        elif resolve =='catas': #catastrophe
            mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy, grow=False) #add vertex, no need to add angle
            mt1.grow = False
        return(idxs)
    elif policy in ['uncross','uncross_m']:
        bdl1, bdl2 = bdl_list[event.bdl1], bdl_list[event.bdl2]
        mt1 = mt_list[idxs]
        bdl_gone = bdl2.del_cross(mt1, pt) #delete mt1 from crossing bdl2, see if bdl1 still crosses
        # if bdl_gone: #if bdl2 no longer crossing bdl1
        #     bdl1.del_cross_bdl(pt)
        #     event.policy = 'uncross_all' #indicate that there is no longer bdl2 crossing this
        return(idxs)
    elif policy == 'cross_br': #branch crossing
        mt1 = mt_list[idxs]
        bdl1 = bdl_list[event.bdl1]
        assert len(bdl1.branch_pos) == len(bdl1.branch_pt)
        pt_i = bdl1.branch_pt.index(pt) #branch index
        #find the appropriate branch bdl
        bdl2n = None
        bdl2 = None
        for i in range(len(bdl1.branchn[pt_i])): #first look for the entrained (not natural) branch
            branchn = bdl1.branchn[pt_i][i]
            if not branch_list[branchn].natural:
                bdl2n = branch_list[branchn].branch_bdln
                bdl2 = bdl_list[bdl2n]
                break #other branch, if exists, is part of the same bdl
        br_mts= []
        incident_mts = []
        if bdl2n != None:
            for i in range(len(bdl1.branchn[pt_i])): #branch mts extending from bdl1
                br_mts += branch_list[bdl1.branchn[pt_i][i]].mts
            incident_mts = bdl2.mt_overlap(mt_list,pt,dists,mt2_n = None, mt_none=br_mts) #other mts in the way
        #useful info
        follow = False
        br_focus = None
        br_nat = None
        new_branch = False #depending on deflect_on, different conditions
        reuse = False #if vertical, don't deflect
        if len(incident_mts) > 0 and not rescue_branch: #bdl2 in the way, must entrain
            deep_angle = False #check whether these are nucleated branches that are at a deep angle
            for brn in bdl1.branchn[pt_i]:
                br_test = branch_list[brn]
                if not br_test.shallow:
                    deep_angle = True
            if deep_angle: #as if it's a deep collision
                X = rnd.randint(0,1)
                if X == 1:
                    event.policy = 'catas'
                    mt1.add_vertex(event.pt, mt1.angle[-1], dists, event.policy, grow=False) #add vertex, no need to add angle
                    mt1.grow = False
                    mt1.free = False
                    global_prop.record_len(mt1, dists)
                else:
                    #TODO this is physically like a crossover pt but I've coded it as a branch pt
                    #if I want to consider severing, need to modify this
                    event.policy = 'cross_br' #cross over like nothing happend
            else: #TODO nucleated branch case requires more careful consideration, especial when it doesn't have mts that act as branches on the other bdl
                event.policy = 'follow_br'
                #edit old and create new MT
                mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy, double_check = rescue_branch) #add vertex
                mt1.grow = False #no longer considered growing
                mt1.hit_bdry = True #hit bdry
                mt1.free = False
                #for potential calculations
                mt1.prev_t = mt1.update_t[-1]
                #new mt
                mt_idx = mt_list[-1].number+1 #new mt number
                mt1.ext_mt = mt_idx #let it know which one is its continuation
                mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                mt_br = mt_list[-1] #branched MT
                #branch business
                follow = True
                branches_zip = [br for br in bdl1.branchn[pt_i] if not branch_list[br].natural] #TODO will this be non-zero len?
                assert len(branches_zip) <= 2 and len(branches_zip) >=1
                assert bdl2 != None #should have found it above
                reg = mt1.region 
                region = region_list[reg]
                angle1 = region.angle[branch_list[branches_zip[0]].traj] #angle of branch, orientation doesn't matter here
                angle2 = region.angle[mt1.traj[-1]] #oriented angle of bdl seg mt1 is on
                zip_res = zip_cat(mt1.angle[-1],angle1,pt,mt1.seg[-1],r=0) #get branch angle
                compass = branch_geo(angle2, zip_res[0], branch_in = False) #get branch geometry; sidedness
                #assumes this branch is only due to zippering, not always the case?
                # print(compass[1],dists, angle1, angle2, get_linenumber())
                branches = [br for br in bdl1.branchn[pt_i] if not branch_list[br].natural and branch_list[br].side == compass[1]]
                if len(branches) != 0: #add mt to corresponding branch
                #TODO: can there be no mts on the branch? Doesn't matter in this case; incident mts are guaranteed
                    assert len(branches) == 1
                    br_focus = branch_list[branches[0]]
                    overlap_check = br_focus.add_mt_mid(mt1, mt_br, bdl1, branch_in=False) #check if it's due to overlap or branch encounter
                    #XXX remember that the branch methods add sidedness but mt.entrain adds other info
                    if br_focus.nucleated: #nucleation case special, there may not be a branch already in br_focus while the twin has a mt
                        if overlap_check: #branch mt added by overlap
                            branch_list[br_focus.twin].add_mt_mid(mt_br, mt1, bdl2, branch_in = True, overlap=incident_mts)
                        else:
                            cross_collision = branch_list[br_focus.twin].add_mt_br(mt_br, mt1, bdl2, bdl1, branch_list, mt_list, branch_in = True)
                            if cross_collision: #this is the case where the branch doesn't have a mt, overlap_check returns false here
                                branch_list[br_focus.twin].add_mt_mid(mt_br, mt1, bdl2, branch_in = True, overlap=incident_mts)
                    else:
                        if overlap_check: #branch mt added by overlap
                            branch_list[br_focus.twin].add_mt_mid(mt_br, mt1, bdl2, branch_in = True, overlap=incident_mts)
                        else: #branch mt added by branch encounter
                            branch_list[br_focus.twin].add_mt_br(mt_br, mt1, bdl2, bdl1, branch_list, mt_list, branch_in = True) #includes sidedness info added
                    mt_br.entrain(mt1, pt, zip_res[0], branch_list[branches[0]].traj) #add entrainment info
                    bdl2.add_mt(mt_br) #add branch mt to branch bdl
                    assert len(bdl2.mt_sides) == len(bdl2.mts)
                    assert len(bdl1.branch_pos) == len(bdl1.branch_pt) and len(bdl2.branch_pos) == len(bdl2.branch_pt)
                else: #need to create new branches
                    #TODO: will a branch disappear before this event is called? Ans: depends on when I delete branch objects. In the present case, this should never be called
                    mt_br.entrain(mt1, pt, zip_res[0], branch_list[branches_zip[0]].traj) #add entrainment info
                    bdl1.add_branch_bdl(pt, bdl2, mt1, mt_br, mt_br.angle[-1], False, branch_list, mt_list, region_list,bdl_list, dists, natural = False, origin = True) #bdl1 has branch outward
                    bdl2.add_branch_bdl(pt, bdl1, mt_br, mt1, mt1.angle[-1], True, branch_list, mt_list, region_list,bdl_list, dists, natural = False, origin = False) #bdl2 has inward branch
                    bdl2.mt_sides.append(branch_list[-1].level)
                    bdl2.add_mt(mt_br) #add branch mt to branch bdl
                    assert len(bdl2.mt_sides) == len(bdl2.mts)
                    assert len(bdl1.branch_pos) == len(bdl1.branch_pt) and len(bdl2.branch_pos) == len(bdl2.branch_pt)
                mt_sublist[mt_br.region].append(mt_br)
        else: #bdl2 not intersecting, follow branch or continue onwards
            assert br_focus == None
            if bdl1.branch_pos[pt_i] != 0:
                mt_i = bdl1.mts.index(mt1.number)
                mt_pol = bdl1.rel_polarity[mt_i]
                mt_side = bdl1.mt_sides[mt_i]
                mt_ns = 'south'
                br_nat = None
                if mt_pol == bdl1.pol:
                    mt_ns = 'north'
                assert br_focus == None
                for br_tempn in bdl1.branchn[pt_i]: #figure out if branch can entrain
                    br_temp = branch_list[br_tempn]
                    assert br_focus == None #should not have been assigned already
                    if not br_temp.natural:
                        if branch_stuff: #only care are about branch sideness when branch_stuff
                            if mt_ns == 'north':
                                if (br_temp.branch_pol == 'south' and br_temp.branch_in) or (br_temp.branch_pol == 'north' and not br_temp.branch_in):
                                    br_focus = br_temp
                                    break
                            else:
                                if (br_temp.branch_pol == 'south' and not br_temp.branch_in) or (br_temp.branch_pol == 'north' and br_temp.branch_in):
                                    br_focus = br_temp
                                    break
                    else: #even when not branch_stuff, always care about natural branches
                        #This assumes there exists only one natural branch, is this true? YES! If natural branch exists, must follow it anyways
                        assert br_nat == None
                        br_nat = br_temp
                if br_focus != None:
                    #check if there are branch mts to follow, and whether they are in the way
                    if len(br_focus.mts) > 0: #there exists branch to entrain
                        if (br_focus.side == 'east' and mt_side > br_focus.level) or (br_focus.side == 'west' and mt_side < br_focus.level):
                            follow = True
                elif br_nat != None: #check if it should follow natural branch otherwise
                    follow = True #never vertical by assumption of naturalness
                    br_focus = br_nat
                    assert len(bdl1.branchn[pt_i]) == 1
                    br1n = bdl1.branchn[pt_i][0]
                    br1 = branch_list[br1n]
                    if len(br1.mts) == 0: #empty
                        new_branch = True
                        #replace the old branch below
            elif bdl1.branch_pos[pt_i] == 0:
                follow = True
                if abs(mt1.angle[-1] - pi/2)<vtol or abs(mt1.angle[-1] - 3*pi/2)<vtol: #vertical case, may reuse bdl info
                    if 0 in bdl1.branchn[pt_i] : #newly branched
                        new_branch = True
                        reuse = True #TODO delete, not needed?
                    else:
                        if bdl_exist(bdl1,bdl_list): #why did I use bdl1 even though this is always true? Because I CAN resuse it! 
                            br_focus = branch_list[bdl1.branchn[0][0]]
                            reuse = True
                        else:
                            new_branch = True
                else: #checks if base of mt has extension already
                    if (0 in bdl1.branchn[pt_i]): #not occupied
                        new_branch = True
                    else: #occupied previously but may be empty
                        assert len(bdl1.branchn[pt_i]) == 1
                        br1n = bdl1.branchn[pt_i][0]
                        br1 = branch_list[br1n]
                        br_focus = br1
                        if len(br1.mts) == 0: #empty
                            new_branch = True
            if follow or new_branch:
                event.policy = 'follow_br'
                mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy, double_check = rescue_branch) #add vertex
                mt1.grow = False #no longer considered growing
                mt1.hit_bdry = True #hit bdry
                mt1.free = False
                #for potential calculations
                mt1.prev_t = mt1.update_t[-1]
                #new mt
                mt_idx = mt_list[-1].number+1 #new mt number
                mt1.ext_mt = mt_idx #let it know which one is its continuation
                mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                mt_br = mt_list[-1] #branched MT
                if new_branch: #only happens at root of mt in geodesic case
                    global_prop.free += 1 #keep track of these
                    #TODO nearly vertical can reuse in theory, but bdry collision is not calculated backwards in current alg
                    #as a result, we re-calculate using new angle instead of original - could be more efficient in theory
                    if br_nat != None: 
                        #this occurs when a non-root point has a new natural branch, need to delete previous
                        assert vtol < 2*pi
                        bdl1.branchn[pt_i] = []
                        # del bdl1.branch_pos[pt_i]
                        #TODO why did I delete this but not branch_pt??????
                        #TODO need to replace old event info involving prev branch
                    angle = mt_br.next_path(mt1.angle[-1], del_l2, global_prop) #if vertical, returns same angle w/ inf tip length
                    traj_n = region_list[mt1.region].add_traj(pt, angle, mt_br,bdl_list)
                    mt_br.entrain(mt1, pt, angle, traj_n)
                    mt_br.free = True
                    mt_br.init_angle = mt1.angle[-1]
                    bdl_list.append(bundle(mt_br, bdl_list, Policy = policy)) #create new bdl w/ mt2 TODO IT'S FREE!
                    bdl2 = bdl_list[-1] #cannot use add_branch to bdl1 because branch already 'exists', must edit manually
                    assert len(bdl1.branch_pos) == len(bdl1.branch_pt) and len(bdl2.branch_pos) == len(bdl2.branch_pt)
                    bdl1.add_branch_bdl(pt, bdl2, mt1, mt_br, mt_br.angle[-1], False, branch_list, mt_list, region_list,bdl_list, dists, natural = True, origin = True) #bdl1 has branch outward
                    bdl2.add_branch_bdl(pt, bdl1, mt_br, mt1, mt1.angle[-1], True, branch_list, mt_list, region_list,bdl_list, dists, natural = True, origin = False) #bdl2 has inward branch
                    #figure out the branch polarity stuff
                    mt1_idx = bdl1.mts.index(mt1.number)
                    mt1_pol = bdl1.rel_polarity[mt1_idx]
                    if mt1_pol == bdl1.pol:
                        bdl2.mt_sides.append(bdl1.get_side(mt1.number))
                    else:
                        bdl2.mt_sides.append(-bdl1.get_side(mt1.number))
                    #NOTE: DO NOT NEED TO ADD SIDEDNESS, ALREADY DONE IN CREATION OF BDL <- not needed?
                    assert len(bdl2.mt_sides) == len(bdl2.mts)
                    assert len(bdl1.branch_pos) == len(bdl1.branch_pt) and len(bdl2.branch_pos) == len(bdl2.branch_pt)
                elif follow: #branch already exists
                    if br_focus == None:
                        assert br_focus != None
                    # assert bdl2.number == br_focus.branch_bdln
                    n = bdl1.mts.index(idxs) #find pol of mt1
                    mt_pol = bdl1.rel_polarity[n]
                    pol = 'south'
                    if mt_pol == bdl1.pol:
                        pol = 'north'
                    bdl2 = bdl_list[br_focus.branch_bdln]
                    
                    overlap_check = br_focus.add_mt_mid(mt1, mt_br, bdl1, branch_in = False)
                    if not br_focus.natural:
                        assert not overlap_check #should hit branch only, not any incident mts
                    br_twin = branch_list[br_focus.twin]
                    fake_br = False
                    if bdl1.branch_pos[pt_i] == 0 or br_nat != None:
                        fake_br = True
                    br_twin.add_mt_br(mt_br, mt1, bdl2, bdl1, branch_list, mt_list, branch_in = True, root=True, fake = fake_br) #includes sidedness info added
                    br_in= br_focus.branch_in
                    br_pol = br_focus.branch_pol
                    angle = br_focus.angle
                    traj_n = br_focus.traj
                    if br_pol == pol:
                        assert not br_in
                    else:
                        assert br_in
                        if angle > pi: #angle reverses
                            angle -= pi
                        else:
                            angle += pi
                    mt_br.entrain(mt1, pt, angle, traj_n) #add entrainment info
                    bdl2.add_mt(mt_br)
                    if reuse: #only difference here is that the below will be used to calculate new intersections
                        mt_br.free = True
                        mt_br.erase_path()
                    assert len(bdl2.mts) == len(bdl2.mt_sides)
                mt_sublist[mt_br.region].append(mt_br)
            else: #doesn't follow branch or create branch on existing branch location
                #if it's a nucleated branch, and there's nothing on the branch to follow, simply continue onward
                #it will reach the ghost point and do the usual stuff w/ natural branches, rather than do that here
                cross_bool = False #determine whether to cross right away
                if br_focus == None: #just no branches
                    cross_bool = True
                else:
                    if br_focus.nucleated and br_focus.branch_in: #ref bdl is from branch nucleation
                        cross_bool = True
                if cross_bool: 
                    #TODO this if/else statement seems unnessary, but keeping to for readability
                    mts = bdl1.mt_overlap(mt_list, pt, dists, mt2_n = None, mt_none = [mt1.number],branch_cross = True)
                    if len(mts) == 0:
                        if abs(mt1.angle[-1] - pi/2)<vtol or abs(mt1.angle[-1] - 3*pi/2)<vtol:
                            event.policy = 'freed_not_free' #free but not really
                            mt1.free = True
                            mt1.erase_path() #forget old deflection info if any
                else:
                    mts = bdl1.mt_overlap(mt_list, pt, dists, mt2_n = None, mt_none = [mt1.number],branch_cross = True)
                    if len(mts) == 0:
                        if abs(mt1.angle[-1] - pi/2)<vtol or abs(mt1.angle[-1] - 3*pi/2)<vtol:
                            event.policy = 'freed_not_free' #free but not really
                            mt1.free = True
                            mt1.erase_path() #forget old deflection info if any
                        else: #new branch just a bit beyond the current branch, same as 'new_branch' w/ new pt
                            res = earliest_event(mt1, dists, mt_list, bdl_list, event_list)
                            event.pt, event.t = res[0], res[1]
                            pt, dists = res[0], res[1]
                            if pt in bdl1.branch_pt: #in weird case (such as very close to bdry), branch pt may already exist
                                event.policy = 'follow_br'
                                mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy, double_check = rescue_branch) #add vertex
                                mt1.grow = False #no longer considered growing
                                mt1.hit_bdry = True #hit bdry
                                mt1.free = False
                                #for potential calculations
                                mt1.prev_t = mt1.update_t[-1]
                                #new mt
                                mt_idx = mt_list[-1].number+1 #new mt number
                                mt1.ext_mt = mt_idx #let it know which one is its continuation
                                mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                                mt_br = mt_list[-1] #branched MT
                                pt_i = bdl1.branch_pt.index(pt)
                                #basically copy-paste from above parts to figure out which branch to use if there are two
                                mt_i = bdl1.mts.index(mt1.number)
                                mt_pol = bdl1.rel_polarity[mt_i]
                                mt_ns = 'south'
                                if mt_pol == bdl1.pol:
                                    mt_ns = 'north'
                                br_focus = None
                                for br_tempn in bdl1.branchn[pt_i]: #figure out if branch can entrain
                                    br_temp = branch_list[br_tempn]
                                    if not br_temp.natural:
                                        if branch_stuff: #only care are about branch sideness when branch_stuff
                                            if mt_ns == 'north':
                                                if (br_temp.branch_pol == 'south' and br_temp.branch_in) or (br_temp.branch_pol == 'north' and not br_temp.branch_in):
                                                    br_focus = br_temp
                                                    break
                                            else:
                                                if (br_temp.branch_pol == 'south' and not br_temp.branch_in) or (br_temp.branch_pol == 'north' and br_temp.branch_in):
                                                    br_focus = br_temp
                                                    break
                                    else: #even when not branch_stuff, always care about natural branches
                                        #This assumes there exists only one natural branch, is this true? YES! If natural branch exists, must follow it anyways
                                        assert br_focus == None
                                        br_focus = br_temp
                                assert br_focus != None
                                bdl2 = bdl_list[br_focus.branch_bdln]
                                overlap_check = br_focus.add_mt_mid(mt1, mt_br, bdl1, branch_in = False)
                                if not br_focus.natural:
                                    assert not overlap_check #should hit branch only, not any incident mts
                                br_twin = branch_list[br_focus.twin]
                                fake_br = False
                                if br_focus.natural:
                                    fake_br = True
                                br_twin.add_mt_br(mt_br, mt1, bdl2, bdl1, branch_list, mt_list, branch_in = True, root=True, fake = fake_br) #includes sidedness info added
                                if bdl2.branch_pt.index(pt) == 0:
                                    #in this case, the branch number was reverted to the 0 placeholder, need to update that
                                    if bdl2.branchn[0][0] == 0:
                                        bdl2.branchn[0][0] = br_twin.number
                                br_in= br_focus.branch_in
                                br_pol = br_focus.branch_pol
                                angle = br_focus.angle
                                traj_n = br_focus.traj
                                if br_pol == mt_ns:
                                    assert not br_in
                                else:
                                    assert br_in
                                    if angle > pi: #angle reverses
                                        angle -= pi
                                    else:
                                        angle += pi
                                mt_br.entrain(mt1, pt, angle, traj_n) #add entrainment info
                                bdl2.add_mt(mt_br)
                                if reuse: #only difference here is that the below will be used to calculate new intersections
                                    mt_br.free = True
                                    mt_br.erase_path()
                                mt_sublist[mt_br.region].append(mt_br)
                            else:
                                event.policy = 'freed_br'
                                mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy, double_check = rescue_branch) #add vertex
                                mt1.grow = False #no longer considered growing
                                mt1.hit_bdry = True #hit bdry
                                mt1.free = False
                                #for potential calculations
                                mt1.prev_t = mt1.update_t[-1]
                                #new mt
                                mt_idx = mt_list[-1].number+1 #new mt number
                                mt1.ext_mt = mt_idx #let it know which one is its continuation
                                mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                                mt_br = mt_list[-1] #branched MT
                                global_prop.free += 1 #keep track of these
                                mt_br.init_angle = mt1.angle[-1]
                                #TODO nearly vertical can reuse in theory, but bdry collision is not calculated backwards in current alg
                                #as a result, we re-calculate using new angle instead of original - could be more efficient in theory
                                angle = mt_br.next_path(mt1.angle[-1], del_l2, global_prop) #if vertical, returns same angle w/ inf tip length
                                traj_n = region_list[mt1.region].add_traj(pt, angle, mt_br,bdl_list)
                                mt_br.entrain(mt1, pt, angle, traj_n)
                                mt_br.free = True
                                mt_br.init_angle = mt1.angle[-1] #TODO delete?
                                bdl_list.append(bundle(mt_br, bdl_list, Policy = policy)) #create new bdl w/ mt2 TODO IT'S FREE!
                                bdl2 = bdl_list[-1] #cannot use add_branch to bdl1 because branch already 'exists', must edit manually
                                bdl1.add_branch_bdl(pt, bdl2, mt1, mt_br, mt_br.angle[-1], False, branch_list, mt_list, region_list,bdl_list, dists, natural = True, origin = True) #bdl1 has branch outward
                                bdl2.add_branch_bdl(pt, bdl1, mt_br, mt1, mt1.angle[-1], True, branch_list, mt_list, region_list,bdl_list, dists, natural = True, origin = False) #bdl2 has inward branch
                                #figure out the branch polarity stuff
                                mt1_idx = bdl1.mts.index(mt1.number)
                                mt1_pol = bdl1.rel_polarity[mt1_idx]
                                if mt1_pol == bdl1.pol:
                                    bdl2.mt_sides.append(bdl1.get_side(mt1.number))
                                else:
                                    bdl2.mt_sides.append(-bdl1.get_side(mt1.number))
                                #NOTE: DO NOT NEED TO ADD SIDEDNESS, ALREADY DONE IN CREATION OF BDL <- not needed?
                                assert len(bdl2.mt_sides) == len(bdl2.mts)
                                assert len(bdl1.branch_pos) == len(bdl1.branch_pt) and len(bdl2.branch_pos) == len(bdl2.branch_pt)
                                mt_sublist[mt_br.region].append(mt_br)
        return(idxs)
    elif policy in ['1catch2','1catch2_m']: #traffic overtake
        mt_idx1, mt_idx2 = idxs[0],idxs[1] #TODO this can overlap with cross_br event in non-geodesic case
        mt1 = mt_list[mt_idx1] #get mts
        mt2 = mt_list[mt_idx2]
        bdl1 = bdl_list[event.bdl1]
        mts = bdl1.mt_overlap(mt_list, pt, dists, mt2_n = None, mt_none = [mt1.number,mt2.number]) #mts that mt1 lies on after catchup
        if len(mts) == 0: #deflects
            if abs(mt1.angle[-1] - pi/2)>vtol and abs(mt1.angle[-1] - 3*pi/2)>vtol:
                event.policy = 'freed_br'
                if which_region(pt) != mt1.region: #this was for an issue caused by the wrong tubulin length unit, should be fine now?
                    print("Edits made here!", get_linenumber())
                    pt = step_forward(mt1,pt)
                    event.pt = pt
                global_prop.free += 1 #keep track of these
                mt1.add_vertex(pt, mt1.angle[-1], dists, event.policy) #add vertex
                mt1.grow = False #no longer considered growing
                mt1.hit_bdry = True #hit bdry
                # mt1.free = False
                #for potential calculations
                mt1.prev_t = mt1.update_t[-1]
                #Branched nucleation
                mt_idx = mt_list[-1].number+1 #new border mt number
                mt1.ext_mt = mt_idx #let it know which one is its continuation
                mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                mt_br = mt_list[-1] #branched MT
                mt_br.init_angle = mt1.angle[-1]
                # frac = 10+rnd.uniform(0, 1)
                # angle = deflect_angle(mt1,frac)
                angle = mt_br.next_path(mt1.angle[-1], del_l2, global_prop)
                traj_n = region_list[mt1.region].add_traj(pt,angle,mt_br,bdl_list)
                mt_br.entrain(mt1, pt, angle, traj_n) #add entrainment info
                mt_br.free = True
                mt_br.init_angle = mt1.angle[-1]
                mt_sublist[mt_br.region].append(mt_br)
                #bundle business
                bdl_list.append(bundle(mt_br, bdl_list, Policy = event.policy)) #create new bdl w/ mt2
                bdl2 = bdl_list[-1]
                bdl1 = bdl_list[mt1.bdl]
                bdl1.add_branch_bdl(pt, bdl2, mt1, mt_br, mt_br.angle[-1], False, branch_list, mt_list, region_list,bdl_list, dists, natural = True, origin = True) #bdl1 has branch outward
                bdl2.add_branch_bdl(pt, bdl1, mt_br, mt1, mt1.angle[-1], True, branch_list, mt_list, region_list,bdl_list, dists, natural = True, origin = False) #bdl2 has inward branch
                #figure out the branch polarity stuff
                mt1_idx = bdl1.mts.index(mt1.number)
                mt1_pol = bdl1.rel_polarity[mt1_idx]
                if mt1_pol == bdl1.pol:
                    bdl2.mt_sides.append(bdl1.get_side(mt1.number))
                else:
                    bdl2.mt_sides.append(-bdl1.get_side(mt1.number))
                assert len(bdl2.mt_sides) == len(bdl2.mts)
                assert len(bdl1.branch_pos) == len(bdl1.branch_pt) and len(bdl2.branch_pos) == len(bdl2.branch_pt)
            else:
                event.policy = 'freed_not_free' #free but not really
                mt1.free = True
                mt1.erase_path() #forget old deflection info if any
        return(mt_idx1)
    elif policy =='disap': #mt has disappeared
            mt_idx1 = idxs
            mt1 = mt_list[mt_idx1]
            mt1.exist = False #has dissapeared
            if mt1.bdl is not None: #if mt1 was part of bdl, erase from bdl
                bdl = bdl_list[mt1.bdl]
                if mt1.from_bdry and not mt1.tread: #if it's an extension, must update previous MT
                    prev_idx = mt1.prev_mt
                    mt_prev = mt_list[prev_idx]
                    mt_prev.grow = False #it must shrink now
                    mt_prev.hit_bdry = False #no longer hitting bdry
                    mt_prev.update_t.append(dists) #update time of new bdry point now shrinking
                    mt_prev.ext_mt = None
                    prev_bdl =  bdl_list[mt_list[prev_idx].bdl] #should have a bdl
                    pt_prev = mt_prev.seg[-1] #point of disappearance
                    event.pt = pt_prev
                    if pt_prev in prev_bdl.branch_pt: #it was a branch pt
                        #remove current mt from branch of prev bdl
                        #if there isn't delfect, cannot just revert the branch_bdl
                        pt_i = prev_bdl.branch_pt.index(pt_prev) #find index to remove branch mt
                        br = None
                        for br_n in prev_bdl.branchn[pt_i]:
                            br_temp = branch_list[br_n]
                            if mt_idx1 in br_temp.mts:
                                br_temp.remove_mt(mt1, mt_prev, prev_bdl, mt_list)
                                br = br_temp
                        #be careful! below reverts the branch number if at the minus end
                        if deflect_on and prev_bdl.branch_pos[pt_i] == 0 and len(br.mts) == 0: #revert to original pseudo branch
                            prev_angle = prev_bdl.angle[0]
                            if abs(prev_angle - pi/2)>vtol and abs(prev_angle-3*pi/2)>vtol: #for sure next time mt will deflect
                                br_idx = prev_bdl.branchn[pt_i].index(br.number)
                                prev_bdl.branchn[pt_i][br_idx] = 0
                        #remove prev mt from branch of current bdl
                        pt_i2 = bdl.branch_pt.index(pt_prev) #index on current branch
                        br2 = branch_list[br.twin]
                        br2.remove_mt(mt_prev, mt1, bdl, mt_list) #requires mt1 to still be recorded on bdl
                        if deflect_on and bdl.branch_pos[pt_i2] == 0 and len(br2.mts) == 0: #revert to original pseudo branch
                            og_angle = bdl.angle[0]
                            if abs(og_angle - pi/2)>vtol and abs(og_angle-3*pi/2)>vtol: #for sure next mt will deflect
                                br_idx = bdl.branchn[pt_i2].index(br2.number)
                                bdl.branchn[pt_i2][br_idx] = 0 #if it's also root of current branch, make 0.5 for consistency
                            #else prev_bdl obviously exists                
                bdl.del_mt(mt1) #do this last because we need the sidedness info above                       
            return(mt_idx1)
    elif policy =='disap_tread': #mt has disappeared, tread
            mt_idx1 = idxs
            mt1 = mt_list[mt_idx1]
            mt1.exist = False #has dissapeared
            if mt1.bdl is not None: #if mt1 was part of bdl, erase from bdl
                bdl = bdl_list[mt1.bdl]
                if mt1.hit_bdry: #if it has extension, must update extension MT
                    ext_idx = mt1.ext_mt
                    mt_ext = mt_list[ext_idx]
                    mt_ext.tread = True #it must shrink now
                    mt_ext.tread_t = dists
                    # mt_ext.hit_bdry = False #no longer hitting bdry
                    # mt_ext.update_t.append(dists) #update time of new bdry point now shrinking
                    mt_ext.prev_mt = None
                    ext_bdl =  bdl_list[mt_list[ext_idx].bdl] #should have a bdl
                    pt_ext = mt_ext.seg[0] #point of disappearance
                    event.pt = pt_ext
                    if pt_ext in ext_bdl.branch_pt: #it was a branch pt
                        #remove current mt from branch of prev bdl
                        #if there isn't delfect, cannot just revert the branch_bdl
                        pt_i = ext_bdl.branch_pt.index(pt_ext) #find index to remove branch mt
                        br = None
                        for br_n in ext_bdl.branchn[pt_i]:
                            br_temp = branch_list[br_n]
                            if mt_idx1 in br_temp.mts:
                                br_temp.remove_mt(mt1, mt_ext, ext_bdl, mt_list)
                                br = br_temp
                                break
                        if deflect_on and ext_bdl.branch_pos[pt_i] == 0 and len(br.mts) == 0: #revert to original pseudo branch
                            ext_angle = ext_bdl.angle[0]
                            if abs(ext_angle - pi/2)>vtol and abs(ext_angle-3*pi/2)>vtol: #for sure next time mt will deflect
                                br_idx = ext_bdl.branchn[pt_i].index(br.number)
                                ext_bdl.branchn[pt_i][br_idx] = 0
                        #remove prev mt from branch of current bdl
                        pt_i2 = bdl.branch_pt.index(pt_ext) #index on current branch
                        br2 = branch_list[br.twin]
                        br2.remove_mt(mt_ext, mt1, bdl, mt_list)
                        if deflect_on and bdl.branch_pos[pt_i2] == 0 and len(br2.mts) == 0: #revert to original pseudo branch
                            og_angle = bdl.angle[0]
                            if abs(og_angle - pi/2)>vtol and abs(og_angle-3*pi/2)>vtol: #for sure next mt will deflect
                                br_idx = bdl.branchn[pt_i2].index(br2.number)
                                bdl.branchn[pt_i2][br_idx] = 0 #if it's also root of current branch, make 0.5 for consistency
                            #else prev_bdl obviously exists
                bdl.del_mt(mt1)
            return(mt_idx1)
    elif policy in ['nucleate', 'sp_catastrophe', 'rescue', 'edge_cat', 'grow_to_pause', 'shrink_to_pause', 'pause_to_grow', 'pause_to_shrink']:
        # if policy == 'nucleate':
        #     mt_idx = None
        #     if len(mt_list)==0:
        #         mt_idx=0
        #     else:
        #         mt_idx = mt_list[-1].number + 1 #new mt number
        #     mt_list.append(mt(mt_idx,free=True)) #introduce new MT
        #     mt1 = mt_list[-1]
        #     mt1.tread = tread_bool #turn on tread
        #     mt1.tread_t = dists
        #     mt1.update_t[-1] = dists
        #     mt1.region = which_region(pt)
        #     mt1.seg = [pt] #assign points and angles
        #     th = rnd.uniform(0,2*pi)
        #     if False:
        #         if th <= pi/2:
        #             th += 2*pi/180
        #         elif th <= pi:
        #             th -= 2*pi/180
        #         elif th <= 3*pi/2:
        #             th += 2*pi/180
        #         else:
        #             th -= 2*pi/180
        #     angle = mt1.next_path(th, del_l2, global_prop)
        #     mt1.angle = [angle]
        #     mt1.init_angle = th
        #     traj_n = region_list[mt1.region].add_traj(pt,angle,mt1,bdl_list)
        #     mt1.traj = [traj_n]
        #     bdl_list.append(bundle(mt1, bdl_list, Policy = policy)) #by default, mts are their own bdl
        #     mt_sublist[mt1.region].append(mt1)
        #     return(mt_idx)
        if policy == 'nucleate':
            if not LDD_bool: #isotropic nucleation
                event.policy = 'unbound_nucleation'
                mt_idx = None
                if len(mt_list)==0:
                    mt_idx=0
                else:
                    mt_idx = mt_list[-1].number + 1 #new mt number
                mt_list.append(mt(mt_idx,free=True)) #introduce new MT
                mt1 = mt_list[-1]
                mt1.tread = tread_bool #turn on tread
                mt1.tread_t = dists
                mt1.update_t[-1] = dists
                mt1.region = which_region(pt)
                mt1.seg = [pt] #assign points and angles
                th = rnd.uniform(0,2*pi)
                # if False:
                #     if th <= pi/2:
                #         th += 2*pi/180
                #     elif th <= pi:
                #         th -= 2*pi/180
                #     elif th <= 3*pi/2:
                #         th += 2*pi/180
                #     else:
                #         th -= 2*pi/180
                angle = mt1.next_path(th, del_l2, global_prop)
                mt1.angle = [angle]
                mt1.init_angle = th
                traj_n = region_list[mt1.region].add_traj(pt,angle,mt1,bdl_list)
                mt1.traj = [traj_n]
                bdl_list.append(bundle(mt1, bdl_list, Policy = policy)) #by default, mts are their own bdl
                mt_sublist[mt1.region].append(mt1)
                return(mt_idx)
            else: #LDD nucleation
                # print('NUCLEATION')
                # pt = [0,0] #get points
                # pt = [0.6,0.5]
                # print(pt)
                nuc_res = nucleate(region_list, mt_list, pt, dists)
                pt, dissociates, unbd_nuc, mt2_n, traj_n, meta_th = nuc_res[0], nuc_res[1], nuc_res[2], nuc_res[3], nuc_res[4], nuc_res[5]
                event.pt = pt
                # print(pt)
                # print(pt, dissociates, unbd_nuc, mt2_n, traj_n, meta_th)
                r = which_region(pt)
                # check_dist = dist(event.pt,event.prev_pt)
                # if check_dist >= R_meta:
                #     if abs(r-which_region(event.prev_pt)) < grid_w-1:
                #         assert check_dist-R_meta <= 1e-9
                if dissociates:
                    event.policy = 'no_nucleation'
                elif unbd_nuc: #same as previous nucleation stuff
                    assert pt != event.prev_pt
                    event.policy = 'unbound_nucleation'
                    mt_idx = None
                    if len(mt_list)==0:
                        mt_idx=0
                    else:
                        mt_idx = mt_list[-1].number + 1 #new mt number
                    mt_list.append(mt(mt_idx,free=True)) #introduce new MT
                    mt1 = mt_list[-1]
                    mt1.tread = tread_bool #turn on tread
                    mt1.tread_t = dists
                    mt1.update_t[-1] = dists
                    mt1.region = which_region(pt)
                    mt1.seg = [pt] #assign points and angles
                    th = rnd.uniform(0,2*pi)
                    # if False: #for testing an angle offset
                    #     if th <= pi/2:
                    #         th += 2*pi/180
                    #     elif th <= pi:
                    #         th -= 2*pi/180
                    #     elif th <= 3*pi/2:
                    #         th += 2*pi/180
                    #     else:
                    #         th -= 2*pi/180
                    angle = mt1.next_path(th, del_l2, global_prop)
                    mt1.angle = [angle]
                    mt1.init_angle = th
                    traj_n = region_list[mt1.region].add_traj(pt,angle,mt1,bdl_list)
                    mt1.traj = [traj_n]
                    bdl_list.append(bundle(mt1, bdl_list, Policy = policy)) #by default, mts are their own bdl
                    mt_sublist[mt1.region].append(mt1)
                    return(mt_idx)
                else:
                    traj_r = region_list[r]
                    ref_th = traj_r.angle[traj_n]
                    br_th = sample_angle() #get sample angles
                    # print("nucleation angle needs to be reverted!!!", get_linenumber())
                    # br_th = pi #TODO get rid of this!!
                    bdl1n = mt_list[mt2_n].bdl 
                    bdl1 = bdl_list[bdl1n] #sorry for inconsistency w/ indents :(
                    if br_th in [0,pi]:
                        event.policy = 'parallel_nucleation'
                        # print('TIME', t, dists)
                        nuc_info = bdl1.mt_overlap_parallel_nuc(mt_list, pt, dists, ref_th, meta_th, br_th, mt2_n, parallel = True)
                        mt_angle, nuc_side = nuc_info[0], nuc_info[1]
                        new_angle = add_pi(mt_angle,br_th)
                        # new_angle = periodic(new_angle) #make sure it's within 2pi
                        # print(new_angle*180/pi, nuc_side, mt2_n)
                        # sys.exit()
                        mt_idx = mt_list[-1].number+1
                        mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                        mt_br = mt_list[-1] #branched MT
                        mt_br.entrain_no_mt(dists, pt, new_angle, traj_n) #add entrainment info
                        mt_br.tread = tread_bool
                        mt_br.tread_t = dists
                        mt_sublist[mt_br.region].append(mt_br)
                        #bundle business
                        bdl1.add_mt(mt_br) #add branched mt to mt2 bdl
                        #bdl1 = bdl_list[mt1.bdl]
                        #bdl1.add_branch_bdl(pt, bdl2, mt1, mt_br, mt_br.angle[-1], False, branch_list, mt_list, region_list,bdl_list, dists, natural = False, origin = True) #bdl1 has branch outward
                        #bdl2.add_branch_bdl(pt, bdl1, mt_br, mt1, mt1.angle[-1], True, branch_list, mt_list, region_list,bdl_list, dists, natural = False, origin = False) #bdl2 has inward branch
                        bdl1.mt_sides.append(nuc_side) #above method does not add mt sidedness
                        assert len(bdl1.mt_sides) == len(bdl1.mts)
                        return(mt_idx)
                    else:
                        event.policy = 'branched_nucleation'
                        nuc_info = bdl1.mt_overlap_parallel_nuc(mt_list, pt, dists, ref_th, meta_th, br_th, mt2_n, parallel = False)
                        mt_angle, nuc_side = nuc_info[0], nuc_info[1] #mt_angle refers to the mt from which nucleation will occur off of
                        mt_angle += br_th #new angle of branched mt
                        mt_angle = periodic(mt_angle)
                        mt_idx = mt_list[-1].number+1
                        mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                        mt_br = mt_list[-1] #branched MT
                        global_prop.free += 1 #keep track of these
                        mt_br.init_angle = mt_angle
                        #TODO nearly vertical can reuse in theory, but bdry collision is not calculated backwards in current alg
                        #as a result, we re-calculate using new angle instead of original - could be more efficient in theory
                        angle = mt_br.next_path(mt_angle, del_l2, global_prop) #if vertical, returns same angle w/ inf tip length
                        spooky_ghost = ghost_pt(pt, angle) #add ghost pts to prevent weird stuff happening at bdl root, use this as traj
                        traj_new = region_list[r].add_traj(spooky_ghost[0],angle,mt_br,bdl_list)
                        mt_br.entrain_no_mt(dists, pt, angle, traj_new)
                        mt_br.free = True
                        mt_br.tread = tread_bool
                        mt_br.tread_t = dists
                        dir_in = determine_branch_compass2(periodic(angle-nuc_info[0]), nuc_info[0], ref_th) #determine compass info of new branch MUST BE CONSISTENT WITH ENTRAINMENT DIRECTIONS
                        # if tread_bool:
                        #     mt_br.tread_t = dists
                        mt_sublist[mt_br.region].append(mt_br)
                        bdl_list.append(bundle(mt_br, bdl_list, Policy = policy)) #create new bdl w/ mt2 TODO IT'S FREE!
                        bdl2 = bdl_list[-1] #cannot use add_branch to bdl1 because branch already 'exists', must edit manually
                        bdl2.seg[0] = spooky_ghost[0]
                        bdl2.start_pos[0] = spooky_ghost[1]*bdl2.pol #ghost pt is the start pt
                        bdl2.branch_pt[0] = spooky_ghost[0]
                        #add_branch_bdl normally uses two mts: branched and previous.
                        #here, there is no "previous"
                        #additionally, need to edit side + branch polarity info manually
                        # print('HUH?')
                        zip_res = zip_cat(angle, ref_th, pt, pt, 0) #determine whether branch is shallow or not
                        shallow = True
                        if zip_res[2] == 'catas':
                            shallow = False
                        bdl1.add_branch_bdl(pt, bdl2, None, mt_br, mt_br.angle[-1], False, branch_list, mt_list, region_list,bdl_list, dists, branching = True, origin = True) #bdl1 has branch outward
                        br1 = branch_list[-1]
                        bdl1_geo = branch_geo(ref_th, angle, False)
                        br1.br_in = False
                        br1.angle = angle
                        br1.level = nuc_side
                        br1.side = bdl1_geo[1]
                        br1.branch_pol = bdl1_geo[0]
                        br1.traj = mt_br.traj[-1]
                        br1.shallow = shallow
                        bdl2.add_branch_bdl(pt, bdl1, mt_br, None, None, dir_in, branch_list, mt_list, region_list,bdl_list, dists, branching = True, origin = False) #bdl2 has inward branch
                        # bdl2.branch_pos.append(spooky_ghost[1]*bdl2.pol) #manually add branch position to avoid recalc
                        assert len(bdl2.branch_pos) == len(bdl2.branch_pt)
                        br2 = branch_list[-1]
                        bdl2_geo = branch_geo(angle, ref_th, dir_in)
                        br2.br_in = dir_in
                        br2.angle = ref_th #TODO does this matter?
                        # br2.mts.append(mt_br.number) no previous mt
                        br2.level = 0
                        br2.side = bdl2_geo[1]
                        br2.branch_pol = bdl2_geo[0]
                        br2.traj = traj_n
                        br2.shallow = shallow
                        if not mt_br.tread: #if not tread, it stays as a branch
                            br1.mts.append(mt_br.number)
                        else:
                            br2.reset_level()
                            br1.reset_level()
                        # bdl2.mt_sides.append(0)
                        #NOTE: DO NOT NEED TO ADD SIDEDNESS, ALREADY DONE IN CREATION OF BDL <- not needed?
                        assert len(bdl2.mt_sides) == len(bdl2.mts)
                        assert len(bdl1.branch_pos) == len(bdl1.branch_pt) and len(bdl2.branch_pos) == len(bdl2.branch_pt)
                        mt_sublist[mt_br.region].append(mt_br)
                        event.bdl1 = bdl1.number #needed for event calculations
                        event.bdl2 = bdl2.number
                        return(mt_idx)
        elif policy in ['sp_catastrophe', 'edge_cat', 'pause_to_shrink']:
            temp_mt = mt_list[idxs] #to find growing tip mt
            new_no = idxs
            if policy in ['sp_catastrophe', 'pause_to_shrink']: #find 'root' mt
                while temp_mt.ext_mt is not None:
                    new_no = temp_mt.ext_mt
                    temp_mt = mt_list[new_no]
            mt1 = mt_list[new_no]
            assert mt1.grow or (not mt1.grow and mt1.hit_bdry)
            new_pt = None
            if policy =='sp_catastrophe': #calculate pt of cat
                new_dist = dists - mt1.update_t[-1] #calculate
                th = mt1.angle[-1]
                old_pt = mt1.seg[-1]
                x, y = old_pt[0] + new_dist*cos(th), old_pt[1] + new_dist*sin(th)
                new_pt = [x,y]
                mt1.add_vertex(new_pt, None, dists, policy, grow=False) #add vertex, last angle does not matter
                global_prop.record_len(mt1, dists)
            elif policy =='edge_cat': #already calculated by bdry collision
                new_pt = event.pt
                mt1.add_vertex(new_pt, None, dists, policy, grow=False) #add vertex, last angle does not matter
                global_prop.record_len(mt1, dists)
            else:
                mt1.update_t.append(dists)
                mt1.hit_bdry=False
            #no need to mt pts pt if pause_to_shrink
            mt1.grow = False
            mt1.free = False
            event.pt = mt1.seg[-1] #add pt to event info
            assert not mt1.hit_bdry
            return(new_no)
        elif policy in ['grow_to_pause', 'shrink_to_pause']:
            temp_mt = mt_list[idxs] #to find growing tip mt
            new_no = idxs
            while temp_mt.ext_mt is not None:
                new_no = temp_mt.ext_mt
                temp_mt = mt_list[new_no]
            mt1 = mt_list[new_no]
            new_pt = None #get paused pt, edit mt1 accordingly
            if policy =='grow_to_pause':
                assert mt1.grow
                new_dist = dists - mt1.update_t[-1] #calculate
                th = mt1.angle[-1]
                old_pt = mt1.seg[-1]
                x, y = old_pt[0] + new_dist*cos(th), old_pt[1] + new_dist*sin(th)
                new_pt = [x,y]
                mt1.add_vertex(new_pt, mt1.angle[-1], dists, event.policy)
                global_prop.record_len(mt1, dists)
            else:
                assert not mt1.grow and not mt1.hit_bdry
                mt1.rescue_seg(dists) #note: it's fine to use this method since it's not actually exclusive to rescue events
            mt1.grow = False
            mt1.hit_bdry = True
            event.pt = mt1.seg[-1]
            assert len(mt1.seg) == len(mt1.traj)
            return(new_no)
        else: #rescue
            assert policy in ['rescue', 'pause_to_grow'] #TODO growing from pause could be a branch point!
            temp_mt = mt_list[idxs] #to find growing tip mt
            new_no = idxs
            while temp_mt.ext_mt is not None:
                new_no = temp_mt.ext_mt
                temp_mt = mt_list[new_no]
            mt1 = mt_list[new_no]
            assert not mt1.grow
            if mt1.hit_bdry:
                assert mt1.ext_mt == None
                mt1.update_t.append(dists)
                mt1.hit_bdry=False
            else:
                mt1.rescue_seg(dists) #edit segments from rescue
            event.pt = mt1.seg[-1] #add pt to event info
            pt = event.pt #redef pt
            mt1.grow = True
            assert(len(mt1.seg)-1 == len(mt1.seg_dist) and len(mt1.seg)==len(mt1.angle))
            free = False #if newly growing tip is free, always False if no deflection
            bdl1 = None
            assert not mt1.hit_bdry
            assert len(mt1.seg) == len(mt1.traj)
            if deflect_on: #w/ deflection, MT might be free
                bdl1 = bdl_list[mt1.bdl]
                mts = bdl1.mt_overlap(mt_list, event.pt, dists, mt2_n = None, mt_none = [mt1.number]) #mts that mt1 lies on after catchup
                if len(mts) == 0: #no mts in the way
                    if abs(mt1.angle[-1] - pi/2)<vtol or abs(mt1.angle[-1] - 3*pi/2)<vtol: #nearly vertical, no deflection
                    #TODO what if parent mt is not entirely vertical, does this mt still follow its vertices?
                        mt1.free = True #for comparisons
                        mt1.erase_path() #forget old deflection info if any
                    else:
                        free = True
            if free: #no mt for alignment
                global_prop.free += 1 #keep track of these
                event.policy = 'freed_rescue'
                # mt1.add_vertex(new_pt, mt1.angle[-1], dists, event.policy) #add vertex
                mt1.grow = False #no longer considered growing
                mt1.hit_bdry = True #hit bdry
                #for potential calculations
                mt1.prev_t = mt1.update_t[-1]
                #Branched nucleation
                mt_idx = mt_list[-1].number+1 #new border mt number
                mt1.ext_mt = mt_idx #let it know which one is its continuation
                mt_list.append(mt(mt_idx)) #add new mt to rep continuation
                mt_br = mt_list[-1] #branched MT
                angle = mt_br.next_path(mt1.angle[-1], del_l2, global_prop)
                traj_n = region_list[mt1.region].add_traj(pt,angle,mt_br,bdl_list)
                mt_br.entrain(mt1, pt, angle, traj_n) #add entrainment info
                mt_br.init_angle = mt1.angle[-1]
                mt_br.free = True
                mt_sublist[mt_br.region].append(mt_br)
                #bundle business
                bdl_list.append(bundle(mt_br, bdl_list, Policy = event.policy)) #create new bdl w/ mt2
                bdl2 = bdl_list[-1]
                bdl1.add_branch_bdl(pt, bdl2, mt1, mt_br, mt_br.angle[-1], False, branch_list, mt_list, region_list,bdl_list, dists, natural = True, origin = True) #bdl1 has branch outward
                bdl2.add_branch_bdl(pt, bdl1, mt_br, mt1, mt1.angle[-1], True, branch_list, mt_list, region_list,bdl_list, dists, natural = True, origin = False) #bdl2 has inward branch
                #figure out the branch polarity stuff
                mt1_idx = bdl1.mts.index(mt1.number)
                mt1_pol = bdl1.rel_polarity[mt1_idx]
                if mt1_pol == bdl1.pol:
                    bdl2.mt_sides.append(bdl1.get_side(mt1.number))
                else:
                    bdl2.mt_sides.append(-bdl1.get_side(mt1.number))
                assert len(bdl1.branch_pos) == len(bdl1.branch_pt)
                assert len(bdl2.branch_pos) == len(bdl2.branch_pt)
                # bdl1.add_branch_bdl(pt, bdl2, mt_br, mt1.angle[-1], False) #bdl1 has branch outward
                # bdl2.add_branch_bdl(pt, bdl1, mt1, mt_br.angle[-1], True) #bdl2 has inward branch
            return(new_no)
    elif policy in ['deflect', 'follow_bdl']:
        l_idx = idxs #mt_to_l(mt_list,idxs)#convert to list index
        mt1 = mt_list[l_idx] #mt to be updated
        th_new = None
        traj_n = None
        if policy == 'deflect': #need to create new angle here
            th_new = mt1.next_path(mt1.angle[-1], del_l2, global_prop)
            traj_n = region_list[mt1.region].add_traj(pt,th_new,mt1,bdl_list)
            global_prop.deflect += 1
        else: #use existing angle
            bdl1 = bdl_list[mt1.bdl]
            pt_i = bdl1.seg.index(pt)
            th_new = None
            mt_pol = 1 #find polarity to decide which dir of angle
            if mt1.angle[-1] > pi/2 and mt1.angle[-1] < 3*pi/2:
                mt_pol = -1
            if bdl1.pol != mt_pol: #reverse direction
                th_new = bdl1.angle[pt_i-1] #angle is actually that of the previous seg
                traj_n = bdl1.traj[pt_i-1]
                if th_new > pi:
                    th_new -= pi
                else:
                    th_new += pi
            else:
                th_new = bdl1.angle[pt_i]
                traj_n = bdl1.traj[pt_i]
        mt1.add_vertex(pt, th_new, dists, policy, traj_id = traj_n) #add vertex
        if mt1.bdl is not None and policy == 'deflect': #it's part of a bundle, need to add new seg
            bdl1 = bdl_list[mt1.bdl] #only leading mt undergoes 'deflect', bundled mts undergo 'deflect_follow'
            bdl1.add_seg(pt, th_new, traj_n)
            # bdl1.seg.append(pt) #thus bundle also gets updated
            # bdl1.angle.append(th_new)
        return(mt1.number)
    else: #hit bdry, nucleates new MT
        l_idx = idxs#mt_to_l(mt_list, idxs) #convert to list index
        mt1 = mt_list[l_idx] #mt to be updated
        not_new = False #whether we are reusing bdl info
        if mt1.bdl is not None: #for the special case where the bdry pt updated after it was calculated
            bdll = bdl_list[mt1.bdl]
            if mt1.angle[-1] > pi/2 and mt1.angle[-1] < 3*pi/2:
                if bdll.prev_pt is not None:
                    pt = bdll.prev_pt
                    if mt1.free:
                        not_new = True
            else:
                if bdll.ext_pt is not None:
                    pt = bdll.ext_pt
                    if mt1.free:
                        not_new = True
        # if not_new: #only happens in the special case
        #     assert abs(mt1.angle[-1]-pi/2) < vtol or abs(mt1.angle[-1]-3*pi/2) < vtol
        #TODO: there is the weird case where MTs are very close to the tolerance, and the MT follows the nonexistent points
        px, py = pt[0], pt[1]
        x1, x2 = xdomain[0], xdomain[1]
        y1, y2 = ydomain[0], ydomain[1]

        if (mt1.from_bdry and len(mt1.seg) > 1) or (not mt1.from_bdry): #from bdry and it has changed at once since existence
            mt1.prev_t = mt1.update_t[-1]
        else: #otherwise we must use the other update time
            assert mt1.prev_t is not None

        th_og = mt1.angle[-1]
        mt1.add_vertex(pt, th_og, dists, policy) #add vertex

        mt1.grow = False #no longer considered growing
        mt1.hit_bdry = True #hit bdry
        mt1.free = False
        # mt1.prev_t = mt1.update_t[-1] #last physical update

        mt_idx = mt_list[-1].number+1 #new border mt number

        mt1.ext_mt = mt_idx #let it know which one is its continuation
        mt_list.append(mt(mt_idx, free = not_new)) #add new mt to rep continuation

        mt2 = mt_list[-1]
        mt2.number = mt_idx

        if len(mt1.seg) > 1: #last physical update time
            mt2.prev_t = mt1.update_t[-2] #same as update time before bdry
        else:
            mt2.prev_t = mt1.prev_t #from bdry, update time from mt1's prev mt
        mt2.from_bdry = True
        mt2.from_bdry_2 = True
        mt2.prev_mt = idxs
        mt2.update_t[-1] = dists #give update time
        mt2.angle.append(th_og) #update angle
        #periodic boundaries
        if px == x1:
            px = x2
            mt2.seg.append([px,py])
        elif px == x2:
            px = x1
            mt2.seg.append([px,py])
        elif py == y1:
            py = y2
            mt2.seg.append([px,py])
        elif py == y2:
            py = y1
            mt2.seg.append([px,py])
        else:
            mt2.seg.append([px,py])
        #assign new region
        if policy == 'top':
            mt2.region = (mt1.region + grid_w)%(grid_w*grid_l)
        elif policy == 'bottom':
            mt2.region = (mt1.region - grid_w)%(grid_w*grid_l)
        elif policy =='left':
            mod = np.floor(mt1.region/grid_w)
            horiz = mt1.region - grid_w*mod
            horiz = (horiz-1)%grid_w
            mt2.region = int(grid_w*mod + horiz)
        elif policy=='right':
            mod = np.floor(mt1.region/grid_w)
            horiz = mt1.region - grid_w*mod
            horiz = (horiz+1)%grid_w
            mt2.region = int(grid_w*mod + horiz)
        mt_sublist[mt2.region].append(mt2)
        #if in bdl, must adjust bdl info
        bdl1 = None #declare bdl for mt1
        if mt1.bdl is None: #turn into a bdl w/ ext or prev
            bdl_list.append(bundle(mt1, bdl_list, event, Policy = policy)) #mt1 already includes bdry pt on its seg! BUT ext/prev_bdl is None
            bdl1 = bdl_list[-1]
        else: #mt1 already in bdl
            bdl1 = bdl_list[mt1.bdl]
            #TODO use of ext and prev not consistent w/ intrinsic MT orientation; need to change if want to be fully generalized
            if mt1.angle[-1] < pi/2 or mt1.angle[-1] > 3*pi/2: #if already bdl but bdry pt not added yet
                if bdl1.ext_bdl is None:
                    bdl1.add_seg(pt,mt1.angle[-1],mt1.traj[-1])
            else:
                if bdl1.prev_bdl is None:
                    bdl1.add_seg(pt,mt1.angle[-1],mt1.traj[-1])
        if mt1.angle[-1] < pi/2 or mt1.angle[-1] > 3*pi/2: #makes extension
            if bdl1.ext_bdl is not None: #already exists
                bdl2 = bdl_list[bdl1.ext_bdl]
                bdl2.add_mt(mt2) #add mt2
                bdl2.mt_sides.append(bdl1.get_side(mt1.number)) #sidedeness same as prev mt
                assert len(bdl2.mt_sides)==len(bdl2.mts)
                mt2.traj.append(bdl2.prev_traj)
            else: #create one
                traj_n = region_list[mt2.region].add_traj(mt2.seg[-1],mt2.angle[-1],mt2,bdl_list,from_wall = policy)
                mt2.traj.append(traj_n)
                mt2.free = True
                bdl_list.append(bundle(mt2, bdl_list, event)) #create new bdl w/ mt2
                bdl2 = bdl_list[-1]
                bdl2.prev_bdl = bdl1.number #prev is not bdl1
                bdl2.prev_traj = traj_n
                bdl2.mt_sides.append(bdl1.get_side(mt1.number)) #sidedeness same as prev mt
                bdl1.ext_bdl = bdl2.number
                bdl1.ext_pt = pt
                bdl1.ext_traj = mt1.traj[-1]
        else: #makes prev bdl
            if bdl1.prev_bdl is not None: #already exists
                bdl2 = bdl_list[bdl1.prev_bdl]
                bdl2.add_mt(mt2) #add mt2
                bdl2.mt_sides.append(bdl1.get_side(mt1.number)) #sidedeness same as prev mt
                assert len(bdl2.mt_sides)==len(bdl2.mts)
                mt2.traj.append(bdl2.ext_traj)
            else: #create one
                traj_n = region_list[mt2.region].add_traj(mt2.seg[-1],mt2.angle[-1],mt2,bdl_list, from_wall = policy)
                mt2.traj.append(traj_n)
                mt2.free = True
                bdl_list.append(bundle(mt2, bdl_list, event, Policy = policy)) #create new bdl w/ mt2
                bdl2 = bdl_list[-1]
                bdl2.ext_bdl = bdl1.number #prev is not bdl1
                bdl2.ext_traj = traj_n
                bdl2.mt_sides.append(bdl1.get_side(mt1.number)) #sidedeness same as prev mt
                bdl1.prev_bdl = bdl2.number
                bdl1.prev_pt = pt
                bdl1.prev_traj = mt1.traj[-1]
        # if abs(mt1.angle[-1]-pi/2) < vtol or abs(mt1.angle[-1]-3*pi/2) < vtol:
        #     mt1.tip_l = np.inf #this info is carried over to mt2 for its bdry collisions
        mt2.carry_path(mt1)
        mt2.carry_len(mt1)
        mt2.init_angle = mt1.init_angle
        if not_new:
            event.policy = 'reuse_bdry'
        return(mt_idx)
    
def purge_list2(mt_list1,bdl_list1,branch_list1,event_list1,last_result1,region_list1,global_p1):
    '''
    Get rid of non-existing MTs and re-number them. Modify event list accordingly.
    Be careful: funny business when called after certain events due to loss of info.

    Parameters
    ----------
    mt_list1 : List of mt objects
        MT list.
    event_list1 : List of event arrays
        Event list.
    last_result1 : Int
        MT last updated.

    Returns
    -------
    Modified input

    '''
    'FOR MTS'
    # global event_map, id_map, bdl_map, region_map, branch_map, event_listn, bdl_listo, mt_listo, bdl_listn, mt_listn, region_listn #for troubleshooting
    bdl_listo = copy.deepcopy(bdl_list1) #for reference
    # mt_listo = copy.deepcopy(mt_list1)
    M = len(mt_list1) #length of original list
    id_map= [None]*M # mapping from ith old id --> id_map[i] (mts only)
    new_mt_list = [mt for mt in mt_list1 if mt.exist] #list of existing mts
    l = len(new_mt_list) #length of list
    #fill mapping
    for i in range(l):
        old_id = new_mt_list[i].number
        id_map[old_id] = i #i is the new id
        if pseudo_bdry:
            if i < 2*grid_w: #TODO only true in the case of pseudo mts
                assert i == old_id
    #before moving on, edit stochastic events in case root mt has treaded away
    for i in range(len(event_list1)):
        for j in range(len(event_list1[i])):
            event = event_list1[i][j]
            if event.policy in ['rescue','sp_catastrophe']:
                if id_map[event.mt1_n] == None:
                    temp_no = event.mt1_n
                    while not mt_list1[temp_no].exist: #find existing root mt
                        mt_temp = mt_list1[temp_no]
                        temp_no = mt_temp.ext_mt
                    event.mt1_n = temp_no
                    event.bdl1 = mt_list1[temp_no].bdl
    'FOR BUNDLES'
    B = len(bdl_list1)
    bdl_map = [None]*B
    dead_bdl_list = [] #can't simply use list comp here, need to search for ext/prev bdls
    for b in range(B): #find dead bdls
        bdl = bdl_list1[b]
        if bdl.number not in dead_bdl_list: #could be already added
            temp_add = [] #list of dead bdls to add
            remain = False #whether there are mts remaning on ANY ext/prev bdl
            if len(bdl.mts) != 0: #not dead
                remain = True
            else: #no mts on bdl anymore, can erase if ext/prev also don't have any
                temp_bdl = bdl
                ext_id = bdl.ext_bdl
                prev_id = bdl.prev_bdl
                temp_add.append(temp_bdl.number)
                while (ext_id is not None) and (not remain):
                    temp_bdl = bdl_list1[ext_id] #examine extension
                    temp_add.append(ext_id)
                    ext_id = temp_bdl.ext_bdl
                    if len(temp_bdl.mts) != 0:
                        remain = True
                temp_bdl = bdl #reassign for searching prev bdls
                while (prev_id is not None) and (not remain):
                    temp_bdl = bdl_list1[prev_id] #examine extension
                    temp_add.append(prev_id)
                    prev_id = temp_bdl.prev_bdl
                    if len(temp_bdl.mts) != 0:
                        remain = True
            if not remain: #if search through bundle returns no mts remaining on them
                for i in range(len(temp_add)): #append found bdl ids
                    dead_bdl_list.append(temp_add[i])
    new_bdl_list = [bdl for bdl in bdl_list1 if bdl.number not in dead_bdl_list]
    b = len(new_bdl_list)
    for i in range(b):
        old_bdl_id = new_bdl_list[i].number
        bdl_map[old_bdl_id] = i #i is the new id
    'FOR BRANCHES'
    Bn = len(branch_list1)
    branch_map = [None]*Bn
    new_branch_list = []
    ii = 0
    for i in range(Bn):
        br = branch_list1[i]
        if br.number == 0:
            new_branch_list.append(br)
            branch_map[0] = 0
        else:
            if bdl_map[br.ref_bdln]!=None and bdl_map[br.branch_bdln]!=None:
                new_branch_list.append(br)
                ii+=1
                branch_map[br.number]=ii
    #edit trajectories
    region_map = []
    new_region_list = []
    for i in range(len(region_list1)): #fill in map and corresponding new traj info
        new_traj = region_traj([],i) #new object to append things to
        region = region_list1[i] #old region_traj object
        L = len(region.angle)
        region_map.append([None]*L) #create empty column in map
        k = 0
        for j in range(L): #determine sub-map in each region
            bdl_n = region.bdl_no[j]
            keep = False
            assert bdl_n != None #all mts should be a bdl
            bdl_n2 = bdl_map[bdl_n]
            if bdl_n2 != None:
                keep = True
            if keep:
                region_map[i][j] = k #determine mapping
                new_traj.add_all(bdl_n2,region.pt[j],region.angle[j],region.bdry_pt[j],region.bdry_wall[j],region.from_wall[j])
                k += 1
        #fill in traj intersections
        old_int = region.intersections
        new_int = []
        for r in range(k): #empty array of intersections
            new_int.append([False]*(r+1))
        count = 0 #count how many existant traj pairs were found
        for r1 in range(L): #fill array
            for r2 in range(r1+1):
                i1, i2 = region_map[i][r1], region_map[i][r2] #convert to new labels
                if i1 != None and i2 != None:
                    new_int[i1][i2] = old_int[r1][r2]
                    count+=1
        assert count == k*(k+1)/2 #should be this many pairs
        new_traj.intersections = new_int
        new_region_list.append(new_traj)
    #now replace ids in new mt list
    for n in range(l):
        new_mt_list[n].number = n # don't need to use mapping here since it's just n
        # new_mt_list[n].free = True
        if new_mt_list[n].ext_mt != None: #change ids w/ map
            new_mt_list[n].ext_mt = id_map[new_mt_list[n].ext_mt]
        if new_mt_list[n].prev_mt != None:
            new_mt_list[n].prev_mt = id_map[new_mt_list[n].prev_mt]
        if new_mt_list[n].bdl != None:
            new_mt_list[n].bdl = bdl_map[new_mt_list[n].bdl]
    #edit growing mt list
    for i in range(len(global_p1.grow_mt)):
        global_p1.grow_mt[i] = id_map[global_p1.grow_mt[i]]
    #edit shrinking mt list
    for s in range(len(global_p1.shrink)):
        global_p1.shrink[s] = id_map[global_p1.shrink[s]]
    #edit paused mt list
    for h in range(len(global_p1.pause_mt)):
        global_p1.pause_mt[h] = id_map[global_p1.pause_mt[h]]
    #do the same for bdls
    for m in range(b):
        new_bdl_list[m].number = m
        for i in range(len(new_bdl_list[m].mts)): #reassign mt nos
            new_bdl_list[m].mts[i] = id_map[new_bdl_list[m].mts[i]]
        # for i in range(len(new_bdl_list[m].branch)): #TODO not needed anymore
        #     new_bdl_list[m].branch[i] = id_map[new_bdl_list[m].branch[i]]
        #crossover info must be deleted for dead bdls; should be empty arrays for mts
        s = range(len(new_bdl_list[m].cross_bdl[:])) # to be looped through
        new_bdl_list[m].cross_pos[:] = [new_bdl_list[m].cross_pos[i] for i in s \
                                        if bdl_map[new_bdl_list[m].cross_bdl[i]] != None]
        new_bdl_list[m].cross_mts[:] = [new_bdl_list[m].cross_mts[i] for i in s \
                                        if bdl_map[new_bdl_list[m].cross_bdl[i]] != None]
        new_bdl_list[m].cross_pt[:] = [new_bdl_list[m].cross_pt[i] for i in s \
                                        if bdl_map[new_bdl_list[m].cross_bdl[i]] != None]
        new_bdl_list[m].cross_bdl[:] = [new_bdl_list[m].cross_bdl[i] for i in s \
                                        if bdl_map[new_bdl_list[m].cross_bdl[i]] != None] #change this last since original is used as reference above
        #do the same for branch info
        for i in range(len(new_bdl_list[m].branch_pos)):
            br_N = len(new_bdl_list[m].branchn[i])
            if new_bdl_list[m].branch_pos[i] == 0: #special case where there's branch at start
                assert len(new_bdl_list[m].branchn[i]) == 1 #TODO is this true for geodesics?
                if new_bdl_list[m].branchn[i][0] != 0:
                    if branch_map[new_bdl_list[m].branchn[i][0]] == None:
                        new_bdl_list[m].branchn[i][0] = 0
            else:
                br_N = range(br_N)
                new_bdl_list[m].branchn[i][:] = [new_bdl_list[m].branchn[i][brn] for brn in br_N \
                                                 if branch_map[new_bdl_list[m].branchn[i][brn]] != None]
        q = range(len(new_bdl_list[m].branch_pos))
        new_bdl_list[m].branch_pos[:] = [new_bdl_list[m].branch_pos[i] for i in q \
                                         if len(new_bdl_list[m].branchn[i])!=0]
        new_bdl_list[m].branch_pt[:] = [new_bdl_list[m].branch_pt[i] for i in q \
                                         if len(new_bdl_list[m].branchn[i])!=0]
        new_bdl_list[m].branchn[:] = [new_bdl_list[m].branchn[i] for i in q \
                                         if len(new_bdl_list[m].branchn[i])!=0]
        for i in range(len(new_bdl_list[m].cross_bdl)): #cross mts
            for j in range(len(new_bdl_list[m].cross_mts[i])):
                if id_map[new_bdl_list[m].cross_mts[i][j]] == None:
                    print(new_bdl_list[m].cross_mts[i][j])
                new_bdl_list[m].cross_mts[i][j] = id_map[new_bdl_list[m].cross_mts[i][j]] # new mt ids
                if new_bdl_list[m].cross_mts[i][j] == None:
                    print('ERROR', new_bdl_list[m].number, new_bdl_list[m].cross_pt[i], new_bdl_list[m].angle)
                    global bdl_
                    bdl_ = new_bdl_list[m]
                    assert new_bdl_list[m].cross_mts[i][j] != None
            new_bdl_list[m].cross_bdl[i] = bdl_map[new_bdl_list[m].cross_bdl[i]] #note use of bdl map NOT mt map here
        for i in range(len(new_bdl_list[m].branchn)):
            new_bdl_list[m].branchn[i] = [branch_map[brn] for brn in new_bdl_list[m].branchn[i] if branch_map[brn] != None]
            if None in new_bdl_list[m].branchn[i]:
                print(new_bdl_list[m].branch_pt[i])
                assert None not in new_bdl_list[m].branchn[i]
        #for ext/prev bdl info
        if new_bdl_list[m].ext_bdl != None:
            new_bdl_list[m].ext_bdl = bdl_map[new_bdl_list[m].ext_bdl]
        if new_bdl_list[m].prev_bdl != None:
            new_bdl_list[m].prev_bdl = bdl_map[new_bdl_list[m].prev_bdl]
    #edit new_mts and new_bdls
    for i in range(l):
        mt = new_mt_list[i]
        r = mt.region
        for j in range(len(mt.traj)):
            mt.traj[j] = region_map[r][mt.traj[j]]
            assert mt.traj[j] <= len(new_region_list[r].angle)-1
            if mt.traj[j] == None:
                print(mt.traj)
                sys.exit()
    for i in range(b):
        bdl = new_bdl_list[i]
        r = bdl.region
        max_no = len(new_region_list[r].angle)-1
        for j in range(len(bdl.traj)):
            bdl.traj[j] = region_map[r][bdl.traj[j]]
            assert bdl.traj[j] <= max_no
        # for j in range(len(bdl.branch_traj)): #TODO delete branch traj?
        #     bdl.branch_traj[j] = region_map[r][bdl.branch_traj[j]]
        #     assert bdl.branch_traj[j] <= max_no
        if bdl.ext_traj != None:
            bdl.ext_traj = region_map[r][bdl.ext_traj]
            assert bdl.ext_traj != None
            assert bdl.ext_traj <= max_no
        if bdl.prev_traj != None:
            bdl.prev_traj = region_map[r][bdl.prev_traj]
            assert bdl.prev_traj != None
            assert bdl.prev_traj <= max_no
        assert len(bdl.branch_pt) == len(bdl.branch_pos)
    #edit branches
    for i in range(len(new_branch_list)):
        if i > 0:
            br = new_branch_list[i]
            br.number = branch_map[br.number]
            assert br.number != None
            br.ref_bdln = bdl_map[br.ref_bdln]
            br.branch_bdln = bdl_map[br.branch_bdln]
            assert br.branch_bdln != None
            br.mts[:] = [id_map[mtn] for mtn in br.mts]
            assert None not in br.mts
            r = new_bdl_list[br.ref_bdln].region
            br.traj = region_map[r][br.traj]
            br.twin = branch_map[br.twin]
    #do the same for events
    L = len(event_list1)
    event_map = []
    for p in range(L): #iterate in regional event lists
        LL = len(event_list1[p])
        event_map.append([True]*LL) #need to delete cross_bdl, uncross involving dead bdls
        for k in range(LL):
            ev = event_list1[p][k]
            if ev.policy in ['cross_bdl','uncross', 'uncross_m']: #cross places may no longer exist
                if bdl_map[ev.bdl2] == None: #bdl 2 is dead
                    event_map[p][k] = False #event must be deleted
            elif ev.policy == 'cross_br': #branch may no longer exist
                # if ev.bdl2 != 0:
                bdl = bdl_listo[ev.bdl1]
                br_idx = bdl.branch_pt.index(ev.pt) #check if it's at the end
                if bdl.branch_pos[br_idx] != 0:# and (len(bdl.branch_mts[br_idx])==0):
                    if bdl_map[ev.bdl2] == None: #if not special case and branch died
                        event_map[p][k] = False
        event_list1[p][:] = [event_list1[p][i] for i in range(LL) if event_map[p][i]] #delete found events
        LL = len(event_list1[p])
        for k in range(LL): #replace event labels
                event = event_list1[p][k]
                ref = [0,0]
                if event.mt1_n != None:
                    event.mt1_n = id_map[event.mt1_n]
                    assert event.mt1_n != None
                if event.mt2_n != None:
                    event.mt2_n = id_map[event.mt2_n]
                    assert event.mt1_n != None
                if event.bdl1 != None:
                    ref[0] = event.bdl1
                    event.bdl1 = bdl_map[event.bdl1]
                    assert event.bdl1 != None
                if event.bdl2 not in [None,0]: #careful for 0 case at the end of a free bdl
                    ref[1] = event.bdl2
                    event.bdl2 = bdl_map[event.bdl2]
                    if event.bdl2 == None and event.policy == 'cross_br':
                        b_idx = bdl_listo[ref[0]].branch_pt.index(event.pt)
                        assert bdl_listo[ref[0]].branch_pos[b_idx] == 0
                        event.bdl2 = 0
                    if event.bdl2 == None and event.policy != 'cross_br':
                        print('ERROR',event.policy,ref,event.pt)
                        assert event.bdl2 != None
    #do the same for the last result
    if last_result1 != None:
        last_result1 = id_map[last_result1]
    for p in range(len(event_list1)):
        for k in range(len(event_list1[p])):
            if event_list1[p][k].mt1_n != None:
                assert event_list1[p][k].mt1_n != event_list1[p][k].mt2_n
    # event_listn = copy.deepcopy(event_list1)
    # bdl_listn, mt_listn, region_listn = copy.deepcopy(new_bdl_list), copy.deepcopy(new_mt_list), copy.deepcopy(new_region_list)
    r_state = rnd.getstate() #for restarting at this state if needed
    return(new_mt_list,new_bdl_list,new_branch_list,new_region_list,event_list1,last_result1,global_p1,r_state)

def read(result): #unpack result object
    return(result.mt1_n,result.mt2_n,result.policy,result.bdl1,result.bdl2,result.pt,result.t,result.calc_t,result.prev)
# global t_list
# t_list = []
def simulate(seed, final_idx, save_path, verbose = False, plot = True, troubleshoot = False): #whole simulation packaged in a function
    current = current_process().name #get process ID
    print('Seed '+str(seed)+' started with '+current+'\n')
    rnd.seed(seed)
    N = 1 #number of initial MTs
    #create lists, then fill
    mt_list = []
    mt_sublist = [] #seperate list that is divided by region, for quicker search
    bdl_list = []
    region_list = []
    branch_list = [branch(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, special=True)]
    global_p = global_prop()
    if pseudo_bdry:
        create_border_mts(mt_list) #create bdry mts
    mt_list.sort(key=lambda x: x.number) #preserve list index <-> mt number
    for i in range(N):
        j = len(mt_list)
        mt_list.append(mt(j,free = True))
        x,y = rnd.uniform(xdomain[0],xdomain[1]), rnd.uniform(ydomain[0],ydomain[1])
        th = rnd.uniform(0,2*pi)
        mt_list[j].seg = [[x,y]]
        mt_list[j].region = which_region([x,y])
        angle = mt_list[j].next_path(th, del_l2, global_p)
        mt_list[j].angle = [angle]
        mt_list[j].init_angle = th
        mt_list[j].bdl = j
        mt_list[j].tread = tread_bool
        mt_list[j].tread_t = 0
        global_p.nucleate(mt_list[j],j)
    for i in range(grid_l*grid_w): #fill regions
        region_list.append(region_traj(mt_list,i))
        mt_sublist.append([mt for mt in mt_list if mt.region == i])
    for i in range(len(mt_list)): #after region info is filled, create bdls (mts are bdls by default)
        bdl_list.append(bundle(mt_list[i], bdl_list, Policy = 'start'))
    t = 0 #rescaled time
    tau = 0 #real time (hr)
    start = time()
    event_list = [[] for i in range(grid_l*grid_w+1)] #by convention, -1 corresponds to stochastic list
    last_result = None
    policy = None
    Result_tuple = None
    #times for hrly printouts
    ti = start
    tf = start
    #times for verbose printouts
    tiv = start
    tfv = start
    #
    i=0
    diff = 2000
    #
    # time_snap = np.arange(1,final_hr+2) #list of checkpoints to save (hr)
    from parameters import time_snap, order_snap
    snap_idx = 0 #current checkpoint
    final_hr = time_snap[final_idx]
    if troubleshoot:
        time_snap = np.arange(0.1,1.1,0.1)
    #
    # time_len = len(time_snap)
    # order_snap = np.linspace(0,final_hr, 10)
    # order_snap = np.delete(order_snap, 0)
    order = [] #store order param info here
    order_t = [] #and time
    order_idx = 0
    #
    save_file = True
    n, m = 0, 0 #for recording list lengths if verbose
    while tau < final_hr:
        i+=1
        if verbose and i%10000==0:
            tfv = time()
            print('Seed '+str(seed)+' info:',\
            '\n Sim time: ', t*conv/(60*60), 'hr or ', t*conv,'s',\
            '\n Wall time elapsed for 10000 steps: ',tfv-tiv,'/',(tfv-tiv)/60,'/',(tfv-tiv)/(60*60),'s/min/hr',\
            '\n Wall time elapsed total: ', time()-start,'/',(time()-start)/60,'/',(time()-start)/(60*60),'s/min/hr',\
            '\n Iteration:',i,\
            '\n Length of event list: ', len(event_list),\
            '\n Length of bundle list: ', len(bdl_list),\
            '\n Length of MT list: ', len(mt_list),\
            '\n Total MT #: ', len(global_p.grow_mt)+len(global_p.shrink),'\n')
            tiv = time()
        Result_tuple = update_event_list(mt_list,mt_sublist,event_list,bdl_list,branch_list,region_list,t,global_p, last_result, result2 = Result_tuple)
        Result = Result_tuple[1]
        #store order param at select times BEFORE the next update
        check_again = True #if the event time gets modified due to time skip (earliest_event call), need to do this funky stuff to preserve accuracy
        if Result.policy != 'cross_br': #timeskip is impossible in this case
            check_again = False
        if tau <= order_snap[order_idx] and Result.t*conv/(60*60) > order_snap[order_idx] and not troubleshoot:
            t2 = order_snap[order_idx]*60*60/conv #intermediate time
            order_regions = [] #mt order info, partitioned by region
            for reg in range(grid_l*grid_w):
                order_regions.append(order_hist(mt_sublist[reg],t2))
            order.append(order_regions) #store
            order_t.append(order_snap[order_idx])
            order_idx+=1
            check_again = False #info already recorded, no need to check again
        #continue update
        last_result = update(mt_list,mt_sublist,bdl_list,branch_list,region_list,event_list,Result,global_p)
        t = Result.t
        if check_again and not troubleshoot: #check again if needed
            if tau <= order_snap[order_idx] and Result.t*conv/(60*60) > order_snap[order_idx]:
                t2 = order_snap[order_idx]*60*60/conv #intermediate time
                order_regions = [] #mt order info, partitioned by region
                for reg in range(grid_l*grid_w):
                    order_regions.append(order_hist(mt_sublist[reg],t2))
                order.append(order_regions) #store, info will be innacurate to a < single tubulin unit length
                order_t.append(order_snap[order_idx])
                order_idx+=1
        tau = t*conv/(60*60)
        # t_list.append(tau)
        #there are bunch of events where erasing info makes things weird prior to executing them
        if len(mt_list) > diff and Result.policy not in ['disap', 'cross_bdl', 'uncross', 'uncross_m', 'cross_br', 'follow_br','disap_tread', 'freed_not_free', \
                                                         'sp_catastrophe', 'rescue', 'grow_to_pause', 'shrink_to_pause', 'pause_to_grow', 'pause_to_shrink', 'freed_rescue',\
                                                        'entrain_other_region','entrain_spaced']: #if too many mts in list, delete and renumber
            if tau >= final_hr:
                break
            if verbose: #list lengths BEFORE purge
                n, m = len(mt_list), len(bdl_list)
            new_lists = purge_list2(mt_list, bdl_list, branch_list, event_list, last_result, region_list, global_p)
            mt_list, bdl_list, branch_list, region_list, event_list, last_result, global_p = new_lists[0], new_lists[1], new_lists[2], new_lists[3], new_lists[4], new_lists[5], new_lists[6]
            mt_total = 0
            mt_sublist = []
            for r in range(grid_l*grid_w): #re-def sublist
                mt_sublist.append([mt for mt in mt_list if mt.region == r])
                mt_total+=len(mt_sublist[r])
            assert mt_total == len(mt_list)
            if verbose:
                print('Seed '+str(seed)+' purge info:',\
                      '\n Prev len(mt_list, bdl_list) =', (n,m), \
                      '\n New purged len(mt_list, bdl_list) =', (len(mt_list),len(bdl_list)), ', at iteration',i, '\n', 't=',t, '\n')
            diff = len(mt_list) + 2000
            if t*conv/(60*60) > time_snap[snap_idx] and save_file:
                #store state
                tf = time()
                print('Seed '+str(seed)+' @',t*conv/(60*60),'hr w/ wall time', tf-start,'/',(tf-start)/60,'/',(tf-start)/(60*60),'s/min/hr',\
                      '\n Elapsed wall time since last hr', tf-ti,'/',(tf-ti)/60,'/',(tf-ti)/(60*60),'s/min/hr \n')
                idx_string = str(snap_idx) + 'idx'#name file by index
                if troubleshoot:
                    idx_string = "{:.1f}".format(time_snap[snap_idx]) + 'hr'
                if plot:
                    plot_snap(mt_list,t,i,save_path+str(snap_idx),save=save_file,region=False, nodes=False)
                #store sim state
                file = save_path+'states/simstate_seed'+str(seed)+'_'+idx_string
                pickle_out = open(file+'.pickle','wb')
                pickle.dump(new_lists,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
                pickle_out.close()
                if not troubleshoot:
                    #store order param
                    file2 = save_path+'orderp_seed'+str(seed)+'_'+idx_string
                    pickle_out = open(file2+'.pickle','wb')
                    pickle.dump((order,order_t),pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
                    pickle_out.close()
                    order, order_t = [], [] #empty the lists
                    #store other global measurements
                    file3 = save_path+'globalp_seed'+str(seed)+'_'+idx_string
                    pickle_out = open(file3+'.pickle','wb')
                    pickle.dump(global_p,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
                    pickle_out.close()
                global_p.mtlen, global_p.mtlen_t, global_p.init_angle = [], [], [] #empty lists
                #continue
                snap_idx +=1
                ti = time()
    end = time()
    print('Total wall time for seed '+str(seed)+' :', end-start,'/',(end-start)/60,'/',(end-start)/(60*60),'s/min/hr', \
        '\n Nuc, free, cross, t:', global_p.nuc,global_p.free,global_p.cross,t,'\n')
    sys.stdout.flush()
    #save state
    lists = (mt_list, bdl_list, branch_list, region_list, event_list, last_result, global_p, rnd.getstate())
    file = save_path+'states/simstate_seed'+str(seed)+'_'+str(snap_idx)+'idx'
    pickle_out = open(file+'.pickle','wb')
    pickle.dump(lists,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
    pickle_out.close()
    #save order
    file2 = save_path+'orderp_seed'+str(seed)+'_'+str(snap_idx)+'idx'
    pickle_out = open(file2+'.pickle','wb')
    pickle.dump((order,order_t),pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
    pickle_out.close()
    #store other global measurements
    file3 = save_path+'globalp_seed'+str(seed)+'_'+str(snap_idx)+'idx'
    pickle_out = open(file3+'.pickle','wb')
    pickle.dump(global_p,pickle_out,protocol=pickle.HIGHEST_PROTOCOL) 
    pickle_out.close()
    if plot:
        plot_snap(mt_list,t,i,save_path+str(snap_idx),save=save_file,region=False, nodes=False)

def rerun(seed, start_idx, final_idx, save_path, verbose = False, plot = False, troubleshoot = False): #restarting from checkpoint
    current = current_process().name #get process ID
    print('Seed '+str(seed)+' started with '+current+'\n')
    # save_path = './troubleshoot/'+str(seed)
    #load stuff in
    pick_in = open(save_path+'states/simstate_seed'+str(seed)+'_'+str(start_idx)+'idx.pickle','rb')
    res_list = pickle.load(pick_in)
    pick_in.close()
    print('Loaded!')
    new_lists = res_list
    mt_list, bdl_list, branch_list, region_list, event_list, last_result, global_p = new_lists[0], new_lists[1], new_lists[2], new_lists[3], new_lists[4], new_lists[5], new_lists[6]
    rnd.setstate(new_lists[-1])
    Result_tuple = sort_and_find_events(event_list, []) #needed for first event
    t = Result_tuple[1].t
    mt_total = 0
    mt_sublist = []
    for r in range(grid_l*grid_w): #re-def sublist
        mt_sublist.append([mt for mt in mt_list if mt.region == r])
        mt_total+=len(mt_sublist[r])
    assert mt_total == len(mt_list)
    tau = t*conv/(60*60) #real time (hr)
    start = time()
    policy = None
    #times for hrly printouts
    ti = start
    tf = start
    #times for verbose printouts
    tiv = start
    tfv = start
    #
    i=0
    diff = 2000
    from parameters import time_snap, order_snap
    snap_idx = start_idx+1 #current checkpoint
    final_hr = time_snap[final_idx]
    if troubleshoot:
        snap_idx = 0
        time_snap = np.arange(0.1,1.1,0.1)+start_idx+1
    #need to load in order parameter array previously calculated
    # file2 = save_path+'orderp_seed'+str(seed)+'_'+str(start_hr)+'hr.pickle'
    # order_pick = open(file2,'rb')
    # orderp = pickle.load(order_pick)
    # order_pick.close()
    order = [] #store order param info here
    order_t = [] #and time
    order_idx = [i for i,t in enumerate(order_snap) if t<= tau][-1] + 1 #index in order_snap for this time
    #
    save_file = True
    n, m = 0, 0 #for recording list lengths if verbose
    while tau < final_hr:
        i+=1
        if verbose and i%10000==0:
            tfv = time()
            print('Seed '+str(seed)+' info:',\
            '\n Sim time: ', t*conv/(60*60), 'hr or ', t*conv,'s',\
            '\n Wall time elapsed for 10000 steps: ',tfv-tiv,'/',(tfv-tiv)/60,'/',(tfv-tiv)/(60*60),'s/min/hr',\
            '\n Wall time elapsed total: ', time()-start,'/',(time()-start)/60,'/',(time()-start)/(60*60),'s/min/hr',\
            '\n Iteration:',i,\
            '\n Length of event list: ', len(event_list),\
            '\n Length of bundle list: ', len(bdl_list),\
            '\n Length of MT list: ', len(mt_list),\
            '\n Total MT #: ', len(global_p.grow_mt)+len(global_p.shrink),'\n')
            tiv = time()
        Result_tuple = update_event_list(mt_list,mt_sublist,event_list,bdl_list,branch_list,region_list,t,global_p, last_result, result2 = Result_tuple)
        Result = Result_tuple[1]
        #store order param at select times BEFORE the next update
        check_again = True #if the event time gets modified due to time skip (earliest_event call), need to do this funky stuff to preserve accuracy
        if Result.policy != 'cross_br': #timeskip is impossible in this case
            check_again = False
        if tau <= order_snap[order_idx] and Result.t*conv/(60*60) > order_snap[order_idx] and not troubleshoot:
            t2 = order_snap[order_idx]*60*60/conv #intermediate time
            order_regions = [] #mt order info, partitioned by region
            for reg in range(grid_l*grid_w):
                order_regions.append(order_hist(mt_sublist[reg],t2))
            order.append(order_regions) #store
            order_t.append(order_snap[order_idx])
            order_idx+=1
            check_again = False #info already recorded, no need to check again
        #continue update
        last_result = update(mt_list,mt_sublist,bdl_list,branch_list,region_list,event_list,Result,global_p)
        t = Result.t
        if check_again and not troubleshoot: #check again if needed
            if tau <= order_snap[order_idx] and Result.t*conv/(60*60) > order_snap[order_idx]:
                t2 = order_snap[order_idx]*60*60/conv #intermediate time
                order_regions = [] #mt order info, partitioned by region
                for reg in range(grid_l*grid_w):
                    order_regions.append(order_hist(mt_sublist[reg],t2))
                order.append(order_regions) #store, info will be innacurate to a < single tubilin unit length
                order_t.append(order_snap[order_idx])
                order_idx+=1
        tau = t*conv/(60*60)
        if len(mt_list) > diff and Result.policy not in ['disap', 'cross_bdl', 'uncross', 'uncross_m', 'cross_br', 'follow_br','disap_tread', 'freed_not_free', \
                                                         'sp_catastrophe', 'rescue', 'grow_to_pause', 'shrink_to_pause', 'pause_to_grow', 'pause_to_shrink', 'freed_rescue',\
                                                        'entrain_other_region','entrain_spaced']: #if too many mts in list, delete and renumber
            if tau >= final_hr:
                break
            if verbose: #list lengths BEFORE purge
                n, m = len(mt_list), len(bdl_list)
            new_lists = purge_list2(mt_list, bdl_list, branch_list, event_list, last_result, region_list, global_p)
            mt_list, bdl_list, branch_list, region_list, event_list, last_result, global_p = new_lists[0], new_lists[1], new_lists[2], new_lists[3], new_lists[4], new_lists[5], new_lists[6]
            mt_total = 0
            mt_sublist = []
            for r in range(grid_l*grid_w): #re-def sublist
                mt_sublist.append([mt for mt in mt_list if mt.region == r])
                mt_total+=len(mt_sublist[r])
            assert mt_total == len(mt_list)
            if verbose:
                print('Seed '+str(seed)+' purge info:',\
                      '\n Prev len(mt_list, bdl_list) =', (n,m), \
                      '\n New purged len(mt_list, bdl_list) =', (len(mt_list),len(bdl_list)), ', at iteration',i, '\n', 't=',t, '\n')
            diff = len(mt_list) + 2000
            if t*conv/(60*60) > time_snap[snap_idx] and save_file:
                #store state
                tf = time()
                print('Seed '+str(seed)+' @',t*conv/(60*60),'hr w/ wall time', tf-start,'/',(tf-start)/60,'/',(tf-start)/(60*60),'s/min/hr',\
                      '\n Elapsed wall time since last hr', tf-ti,'/',(tf-ti)/60,'/',(tf-ti)/(60*60),'s/min/hr \n')
                if plot:
                    plot_snap(mt_list,t,i,save_path+str(snap_idx),save=save_file,region=False, nodes=False)
                idx_string = str(snap_idx) + 'idx'#name file by index
                if troubleshoot:
                    idx_string = "{:.1f}".format(time_snap[snap_idx]) + 'hr'
                if plot:
                    plot_snap(mt_list,t,i,save_path+str(snap_idx),save=save_file,region=False, nodes=False)
                #store sim state
                file = save_path+'states/simstate_seed'+str(seed)+'_'+idx_string 
                pickle_out = open(file+'.pickle','wb')
                pickle.dump(new_lists,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
                pickle_out.close()
                if not troubleshoot:
                    #store order param
                    file2 = save_path+'orderp_seed'+str(seed)+'_'+idx_string 
                    pickle_out = open(file2+'.pickle','wb')
                    pickle.dump((order,order_t),pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
                    order, order_t = [], [] #empty the lists
                    pickle_out.close()
                    #store other global measurements
                    file3 = save_path+'globalp_seed'+str(seed)+'_'+idx_string 
                    pickle_out = open(file3+'.pickle','wb')
                    pickle.dump(global_p,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
                    pickle_out.close()
                global_p.mtlen, global_p.mtlen_t, global_p.init_angle = [], [], [] #empty lists
                #continue
                snap_idx +=1
                ti = time()
    end = time()
    print('Total wall time for seed '+str(seed)+' :', end-start,'/',(end-start)/60,'/',(end-start)/(60*60),'s/min/hr', \
        '\n Nuc, free, cross, t:', global_p.nuc,global_p.free,global_p.cross,t,'\n')
    sys.stdout.flush()
    #save state
    lists = (mt_list, bdl_list, branch_list, region_list, event_list, last_result, global_p, rnd.getstate())
    file = save_path+'states/simstate_seed'+str(seed)+'_'+str(snap_idx)+'idx'
    pickle_out = open(file+'.pickle','wb')
    pickle.dump(lists,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
    pickle_out.close()
    #save order
    file2 = save_path+'orderp_seed'+str(seed)+'_'+str(snap_idx)+'idx'
    pickle_out = open(file2+'.pickle','wb')
    pickle.dump((order,order_t),pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
    pickle_out.close()
    #store other global measurements
    file3 = save_path+'globalp_seed'+str(seed)+'_'+str(snap_idx)+'idx'
    pickle_out = open(file3+'.pickle','wb')
    pickle.dump(global_p,pickle_out,protocol=pickle.HIGHEST_PROTOCOL) 
    pickle_out.close()
    if plot:
        plot_snap(mt_list,t,i,save_path+str(snap_idx),save=save_file,region=False, nodes=False)

def unpack_events(event_list):
    gevent_list = []
    for i in range(len(event_list)):
        gevent_list += event_list[i]
    gevent_list.sort(key=lambda x: x.t)
    return(gevent_list)
    

def create_border_mts(mt_list):
    '''
    Creates pseudo mts that act as boundaries for mts to entrain upon

    Parameters
    ----------
    mt_list : List of MTs
        DESCRIPTION.

    Returns
    -------
    None. Adds mts to to mt_list

    '''
    for i in range(grid_w): #make bdry pseudo mts
        k1 = i #top/bottom row grids
        k2 = grid_w*(grid_l-1) + i
        updown = [k1, k2]
        mt_no = None
        for j in updown:
            up = False #whether we are at top or bottom
            if updown.index(j) == 1:
                up = True
            if up:
                mt_no = grid_w + i
            else:
                mt_no = i
            assert mt_no not in [m.number for m in mt_list]
            mt_list.append(mt(mt_no, free=False))
            mtb = mt_list[-1]
            x1, x2 = x_interval[j][0], x_interval[j][1]
            y1, y2 = y_interval[j][0], y_interval[j][1]
            yupdown = y1 + 1e-10 #vertical pos depends on top/bottom
            if up:
                yupdown = y2 - 1e-10
            mtb.seg = [[x1,yupdown], [x2,yupdown]]
            mtb.region = j 
            mtb.angle = [0,0]
            mtb.seg_dist.append(x2-x1)
            mtb.tread = False
            mtb.tread_t = None
            mtb.grow = False
            mtb.hit_bdry = True
            mtb.from_bdry = True
            mtb.pseudo_bdry = True
            mtb.bdl = mtb.number
            if j in [0, grid_w*(grid_l-1)]:
                mtb.prev_mt = mt_no + grid_w-1
                mtb.ext_mt = mt_no + 1
            elif j in [grid_w-1, grid_w*grid_l-1]:
                mtb.ext_mt = mt_no + 1-grid_w
                mtb.prev_mt = mt_no - 1
            else:
                mtb.prev_mt = mt_no - 1
                mtb.ext_mt = mt_no + 1
                

def check_result(result,mt_list,t):
    #for easy troubleshooting
    rr = [which_region(result.pt)]
    pt_list = [result.pt]
    list1 = [m for m in mt_list if m.region in rr]
    plot_snap(list1,t,0,'.',save=False,region=True, nodes=False, show=True, color=True, result = Result, plot_pts = pt_list, plot_region=rr)

if __name__ == '__main__':
    import sys
    import os
    from datetime import datetime
    # rerun(5,0,9,'./2024.04.20_18:45/',plot=False, verbose = True, troubleshoot = True)
    # sys.exit()
    # # path = './'+datetime.now().strftime('%Y.%m.%d_%H:%M:%S') +'/'
    # # os.mkdir(path)
    # # simulate(52,0.5,path,verbose=True)
    # # print('l_0 = ',(4/(r_n*3))**(1/3))
    # # sys.exit()
    # # pick_in = open('simstate_seed107_10hr.pickle','rb')
    # # res_list = pickle.load(pick_in)
    # # pick_in.close()
    # # m_list,e_list = res_list[0], res_list[3]
    # # print(order_param(m_list,e_list[0].t))
    # # t = 0
    # # angle = np.linspace(0,pi/2,100)
    # # res = []
    # # for j in range(len(angle)):
    # #     mt_list = []
    # #     # dx = np.linspace(0,1,5)
    # #     # for i in range(len(dx)):
    # #     a = angle[j]
    # #     mt_list.append(mt(0,free = True))
    # #     mt_list[0].update_t = [0]
    # #     mt_list[0].seg = [[0,0],[np.cos(a),np.sin(a)]]
    # #     mt_list[0].angle = [a,a]
    # #     mt_list[0].seg_dist = [1]
    # #     mt_list[0].hit_bdry = True
    # #     res.append(order_param(mt_list,0)[-2])
    # #     # print(order_param(mt_list,0),a)
    # # import matplotlib.pyplot as plt
    # # plt.plot(angle,res)
    # path = './2022.07.19_20:40/'
    # #load stuff in
    # pick_in = open(path+'simstate_seed0_10hr.pickle','rb')
    # res_list = pickle.load(pick_in)
    # pick_in.close()
    # print('Loaded!')
    # new_lists = res_list
    # mt_list, bdl_list, region_list, event_list, last_result, global_p = new_lists[0], new_lists[1], new_lists[2], new_lists[3], new_lists[4], new_lists[5]
    # t = event_list[0].t
    # hist_res = order_hist(mt_list,t)
    # print(s2(hist_res))
    # print(s2_new(hist_res))
    # import matplotlib.pyplot as plt
    # binss = np.linspace(-pi/2,pi/2,90)
    # plt.hist(hist_res[0],binss,weights=hist_res[1],density=True)
    # plt.show()
    # mt_list = []
    # mt_list.append(mt(0,free=True))
    # mt_list[0].seg = [[0,0],[1,0],[2,0],[3,0]]
    # mt_list[0].angle = [0,0,0,0]
    # mt_list[0].seg_dist = [1,1,1]
    # mt_list[0].tread = True
    # mt_list[0].tread_t = 3+2.25/2-2*0.1
    # mt_list[0].update_t=[3]
    # mt_list[0].vt = 0.5
    # mt_list[0].grow = False
    # mt_list[0].hit_bdry = False
    # res = order_hist(mt_list,3+2.25/2)
    # sys.exit()
    # print('Troubleshoot settings. Must change: paramters, grids, plotting')
    # print('Also check nucleation probabilities')
    fresh_start = True
    if fresh_start:
        seed = 10
        rnd.seed(seed)
        N = 1#number of initial MTs
        mt_list = []
        mt_sublist = [] #seperate list that is divided by region, for quicker search
        bdl_list = []
        region_list = []
        branch_list = [branch(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, special=True)]
        global_p = global_prop()
        if pseudo_bdry:
            create_border_mts(mt_list) #create bdry mts
        mt_list.sort(key=lambda x: x.number) #preserve list index <-> mt number
        for i in range(N):
            j = len(mt_list)
            mt_list.append(mt(j,free = True))
            # x,y = 1, 1
            # th = pi/12
            # if i==0:
            #     y = 1-1e-10
            # if i==2:
            #     y = 1+1e-10
            x,y = rnd.uniform(xdomain[0],xdomain[1]), rnd.uniform(ydomain[0],ydomain[1])
            th = rnd.uniform(0,2*pi)
            mt_list[j].seg = [[x,y]]
            mt_list[j].region = which_region([x,y])
            angle = None
            # if i==2:
            #     angle = mt_list[j].next_path(th, del_l2*4, global_p)
            # else:
            angle = mt_list[j].next_path(th, del_l2, global_p)
            mt_list[j].angle = [angle]
            mt_list[j].init_angle = th
            mt_list[j].bdl = j
            mt_list[j].tread = tread_bool
            mt_list[j].tread_t = 0
            global_p.nucleate(mt_list[j],j)
        for i in range(grid_l*grid_w): #fill regions
            region_list.append(region_traj(mt_list,i))
            mt_sublist.append([mt for mt in mt_list if mt.region == i])
        for i in range(len(mt_list)): #after region info is filled, create bdls (mts are bdls by default)
            bdl_list.append(bundle(mt_list[i], bdl_list, Policy = 'start'))
        t = 0
        start = time()
        #XXX
        event_list = [[] for i in range(grid_l*grid_w+1)] #by convention, -1 corresponds to stochastic list
        #XXX
        event_vis = [] #TODO how to change?
        pevent_list = []#past events
        last_result = None
        policy = None
        ti = start
        tf = 0
        i=0
        diff = 2000#float("inf")
        event_listo = []
        time_snap = [.95,0.96,0.97,0.98,0.99,3]
        snap_idx = 0
        final_hr = 3
        save_file = False
        path='.'
        Result = None
        Result_tuple = None
        if save_file:
            path = './'+datetime.now().strftime('%Y.%m.%d_%H:%M:%S') +'/'
            os.mkdir(path) #new dir to save stuff in
            # path ='./2024.01.26_20:30:27/'
        # while i < 1000:
        tau = 0
        order_idx = 0
        order_snap = np.linspace(0.01,final_hr,100)
        vert_crossing = []
        unique_vert_crossing = []
        horiz_crossing = []
        unique_horiz_crossing = []
        S2 = []
        from domain_analysis import count_crossings_sample, count_crossings
        while t*conv/(60*60) < final_hr:
        # while t < 0.9518263249094895:
            i+=1
            if i%50000==0:
                tf = time()
                print('Sim time: ', t*conv/(60*60), 'hr or ', t*conv,'s')
                print('Wall time elapsed for 50000 steps: ',tf-ti)
                print('Wall5 time elapsed total: ', time()-start)
                print('Iteration:',i)
                print('Length of event list: ', np.sum([len(lists) for lists in event_list]))
                print('Length of bundle list: ', len(bdl_list))
                print('Length of MT list: ', len(mt_list))
                print('Total MT #: ', len(global_p.grow_mt)+len(global_p.shrink) + len(global_p.pause_mt))
                print('Total additional anchoring events: ', global_p.more_anchoring,'\n')
                order_list = [m for m in mt_list if not m.pseudo_bdry]
                plot_snap(order_list,t,i,'.',save=False,region=True, nodes=False, show=True, color=True, result=Result)
                plot_hist_mtlist(order_list, t)
                ti = time()
                # pevent_list = []
            # if Result != None:
            #     if i>0 and Result.policy in ['branched_nucleation', 'unbound_nucleation', 'parallel_nucleation']:
            #         rr = [0,1,2,3]
            #         list1 = [m for m in mt_list if m.region in rr]
            #         pt_list = [Result.pt]
            #         if Result.policy in ['branched_nucleation', 'unbound_nucleation', 'parallel_nucleation']:
            #             pt_list.append(Result.prev_pt)
            #         plot_snap(list1,t,i,'.',save=False,region=True, nodes=False, show=True, color=True, result = Result, plot_pts = pt_list, plot_region=rr)
            Result_tuple = update_event_list(mt_list,mt_sublist,event_list,bdl_list,branch_list,region_list,t,global_p, last_result, result2 = Result_tuple)
            Result = Result_tuple[1]
            #store order param at select times BEFORE the next update
            check_again = True #if the event time gets modified due to time skip (earliest_event call), need to do this funky stuff to preserve accuracy
            if Result.policy != 'cross_br': #timeskip is impossible in this case
                check_again = False
            # if tau <= order_snap[order_idx] and Result.t*conv/(60*60) > order_snap[order_idx]:
            #     cross_info = count_crossings(bdl_list,mt_list,branch_list)
            #     vert_crossing.append(cross_info['vert_cross'])
            #     unique_vert_crossing.append(cross_info['unique_vert_cross'])
            #     horiz_crossing.append(cross_info['horiz_cross'])
            #     unique_horiz_crossing.append(cross_info['unique_horiz_cross'])
            #     S2.append(s2(order_hist(mt_list,t))[0])
            #     order_idx+=1
            #     check_again = False #info already recorded, no need to check again
            #continue update
            
            pevent_list.append(Result)
            last_result = update(mt_list,mt_sublist,bdl_list,branch_list,region_list,event_list,Result,global_p)
            t = Result.t
            # if check_again: #check again if needed
            #     if tau <= order_snap[order_idx] and Result.t*conv/(60*60) > order_snap[order_idx]:
            #         cross_info = count_crossings(bdl_list,mt_list,branch_list)
            #         vert_crossing.append(cross_info['vert_cross'])
            #         unique_vert_crossing.append(cross_info['unique_vert_cross'])
            #         horiz_crossing.append(cross_info['horiz_cross'])
            #         unique_horiz_crossing.append(cross_info['unique_horiz_cross'])
            #         S2.append(s2(order_hist(mt_list,t))[0])
            #         order_idx+=1
            tau = t*conv/(60*60)
            if len(mt_list) > diff and Result.policy not in ['disap', 'cross_bdl', 'uncross', 'uncross_m', 'cross_br', 'follow_br','disap_tread', 'freed_not_free', \
                                                             'sp_catastrophe', 'rescue', 'grow_to_pause', 'shrink_to_pause', 'pause_to_grow', 'pause_to_shrink', 'freed_rescue',\
                                                             'entrain_other_region','entrain_spaced']: #if too many mts in list, delete and renumber
                n, m = len(mt_list), len(bdl_list)
                # event_listo = copy.deepcopy(event_list)
                new_lists = purge_list2(mt_list, bdl_list, branch_list, event_list, last_result, region_list, global_p)
                mt_list, bdl_list, branch_list, region_list, event_list, last_result, global_p = new_lists[0], new_lists[1], new_lists[2], new_lists[3], new_lists[4], new_lists[5], new_lists[6]
                mt_total = 0
                mt_sublist = []
                for r in range(grid_l*grid_w): #re-def sublist
                    mt_sublist.append([mt for mt in mt_list if mt.region == r])
                    mt_total+=len(mt_sublist[r])
                assert mt_total == len(mt_list)
                print('Prev len(mt_list, bdl_list) =', (n,m))
                print('New purged len(mt_list, bdl_list) =', (len(mt_list),len(bdl_list)), ',at iteration',i, '\n', 't=',t, '\n')
                diff = len(mt_list) + 2000
                if t*conv/(60*60) > time_snap[snap_idx] and save_file:
                    plot_snap(mt_list,t,i,path+str(snap_idx),save=save_file,region=False, nodes=False)
                    #store state
                    file = path+'simstate_seed'+str(seed)+'_'+str(time_snap[snap_idx])+'idx'
                    pickle_out = open(file+'.pickle','wb')
                    pickle.dump(new_lists,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
                    pickle_out.close()
                    #
                    snap_idx +=1
        end = time()
        print('Total time: ',end-start)
        print('Nuc, free, cross, t:', global_p.nuc,global_p.free,global_p.cross,t)
        # import matplotlib.pyplot as plt
        # plt.plot(order_snap, vert_crossing, label='vert', linewidth=0.5)
        # plt.plot(order_snap, unique_vert_crossing, label='u_vert',linewidth=0.5, linestyle='--')
        # plt.plot(order_snap, horiz_crossing, label = 'horiz',linewidth=0.5)
        # plt.plot(order_snap, unique_horiz_crossing, label = 'u_horiz',linewidth=0.5, linestyle='--')
        # plt.legend()
        # plt.show()
        # plt.clf()
        # plt.plot(order_snap, S2)
        # plt.show()
        # plt.clf()
        # plt.plot(order_snap, np.array(vert_crossing)/np.array(horiz_crossing), label = 'not u',linewidth=0.5)
        # plt.plot(order_snap, np.array(unique_vert_crossing)/np.array(unique_horiz_crossing), label = 'u',linewidth=0.5)
        # plt.legend()
        # plt.show()
        # plt.clf()
        # plt.plot(order_snap[:-500], vert_crossing[:-500], label='vert', linewidth=0.5)
        # plt.plot(order_snap[:-500], unique_vert_crossing[:-500], label='u_vert',linewidth=0.5, linestyle='--')
        # plt.plot(order_snap[:-500], horiz_crossing[:-500], label = 'horiz',linewidth=0.5)
        # plt.plot(order_snap[:-500], unique_horiz_crossing[:-500], label = 'u_horiz',linewidth=0.5, linestyle='--')
        # plt.legend()
        # lists = (mt_list, bdl_list, branch_list, region_list, event_list, last_result, global_p, rnd.getstate())
        # file = 'example_normal_kon2'
        # pickle_out = open(file+'.pickle','wb')
        # # print(len(event_list))
        # pickle.dump([mt_list,event_list],pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        # print(t)
        # pickle_out.close()
        # # plot_snap(mt_list,t,i,'./test'+str(t),save=False,region=True,show=True, nodes=False)
        # # event_vis = [unpack_res(x) for x in event_list]
        # # pevent_vis = [unpack_res(x) for x in pevent_list]
        # # #save state
        # lists = (mt_list, bdl_list, region_list, event_list, last_result, global_p, rnd.getstate())
        # file = path+'simstate_seed'+str(seed)+'_'+str(final_hr)+'hr'
        # pickle_out = open(file+'.pickle','wb')
        # pickle.dump(lists,pickle_out)
        # pickle_out.close()
        # if Result.policy not in ['disap', 'cross_bdl', 'uncross','cross_br', 'follow_br']:
        #     new_lists = purge_list2(mt_list, bdl_list, event_list, last_result,global_p)
        #     mt_list, bdl_list, event_list, last_result, global_p = new_lists[0], new_lists[1], new_lists[2], new_lists[3], new_lists[4]
        # plot_snap(mt_list,t,i,path+str(snap_idx),save=save_file,region=False, nodes=False, show=True)
    else: #start from checkpoint
        seed=97
        mt_list = []
        mt_sublist = [] #seperate list that is divided by region, for quicker search
        bdl_list = []
        region_list = []
        branch_list = []
        global_p = global_prop()
        t = 0
        start = time()
        event_list = []
        event_vis = []
        pevent_list = []#past events
        last_result = None
        policy = None
        ti = start
        tf = 0
        i=0
        diff = 2000
        event_listo = []
        time_snap = [np.inf]
        snap_idx = 0
        final_hr = 10
        save_file = False
        # if save_file:
        #     path = './seed3_troubleshoot/'
        #     os.mkdir(path) #new dir to save stuff in
        path = './PNAS_data/mtkon_100um/'
        #load stuff in
        pick_in = open(path+'simstate_seed0_9.4hr.pickle','rb')
        res_list = pickle.load(pick_in)
        pick_in.close()
        print('Loaded!')
        new_lists = res_list
        mt_list, bdl_list, branch_list, region_list, event_list, last_result, global_p = new_lists[0], new_lists[1], new_lists[2], new_lists[3], new_lists[4], new_lists[5], new_lists[6]
        rnd.setstate(new_lists[-1])
        Result_tuple = sort_and_find_events(event_list, [])
        t = Result_tuple[1].t
        mt_total = 0
        mt_sublist = []
        for r in range(grid_l*grid_w): #re-def sublist
            mt_sublist.append([mt for mt in mt_list if mt.region == r])
            mt_total+=len(mt_sublist[r])
        assert mt_total == len(mt_list)
        diff = len(mt_list) + 2000
        print('Starting sim!', t*conv/(60*60))
        while t*conv/(60*60) < final_hr:
        # while t < 59.73721594659152:
        # while i < 1:
            i+=1
            if i%50000==0:
                tf = time()
                print('Sim time: ', t*conv/(60*60), 'hr or ', t*conv,'s')
                print('Wall time elapsed for 50000 steps: ',tf-ti)
                print('Wall time elapsed total: ', time()-start)
                print('Iteration:',i)
                print('Length of event list: ', len(event_list))
                print('Length of bundle list: ', len(bdl_list))
                print('Length of MT list: ', len(mt_list))
                print('Total MT #: ', len(global_p.grow_mt)+len(global_p.shrink),'\n')
                plot_snap(mt_list,t,i,'.',save=False,region=False, nodes=False, show=True, color=True, result=Result)
                ti = time()
                # pevent_list = []
            # if i>0 and i%100==0 :
            #     rr = [212]
            #     list1 = [m for m in mt_list if m.region in rr]
            #     pt_list = [[0.9841075759131519, 2.0084254907012395]]
            #     if Result.policy in ['branched_nucleation', 'unbound_nucleation', 'parallel_nucleation']:
            #         pt_list.append(Result.prev_pt)
            #     plot_snap(list1,t,i,'.',save=False,region=True, nodes=False, show=True, color=True, result = Result, plot_pts = pt_list, plot_region=rr)
            Result_tuple = update_event_list(mt_list,mt_sublist,event_list,bdl_list,branch_list,region_list,t,global_p, last_result, result2 = Result_tuple)
            Result = Result_tuple[1]
            pevent_list.append(Result)
            last_result = update(mt_list,mt_sublist,bdl_list,branch_list,region_list,event_list,Result,global_p)
            t = Result.t
            if len(mt_list) > diff and Result.policy not in ['disap', 'cross_bdl', 'uncross', 'uncross_m', 'cross_br', 'follow_br','disap_tread', \
                                                             'sp_catastrophe', 'rescue', 'grow_to_pause', 'shrink_to_pause', 'pause_to_grow', 'pause_to_shrink', 'freed_rescue']: #if too many mts in list, delete and renumber
                n, m = len(mt_list), len(bdl_list)
                # event_listo = copy.deepcopy(event_list)
                new_lists = purge_list2(mt_list, bdl_list, branch_list, event_list, last_result, region_list, global_p)
                mt_list, bdl_list, branch_list, region_list, event_list, last_result, global_p = new_lists[0], new_lists[1], new_lists[2], new_lists[3], new_lists[4], new_lists[5], new_lists[6]
                mt_total = 0
                mt_sublist = []
                for r in range(grid_l*grid_w): #re-def sublist
                    mt_sublist.append([mt for mt in mt_list if mt.region == r])
                    mt_total+=len(mt_sublist[r])
                assert mt_total == len(mt_list)
                print('Prev len(mt_list, bdl_list) =', (n,m))
                print('New purged len(mt_list, bdl_list) =', (len(mt_list),len(bdl_list)), ',at iteration',i, '\n', 't=',t, '\n', 'time', t*conv/(60*60), '\n')
                diff = len(mt_list) + 2000
                if t*conv/(60*60) > time_snap[snap_idx] and save_file:
                    # plot_snap(mt_list,t,i,path+str(snap_idx),save=save_file,region=False, nodes=False)
                    #store state
                    file = path+'simstate_seed'+str(seed)+'_'+str(time_snap[snap_idx])+'idx'
                    pickle_out = open(file+'.pickle','wb')
                    pickle.dump(new_lists,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
                    pickle_out.close()
                    #
                    print('SAVED AT', t)
                    snap_idx +=1
        end = time()
        print('Total time: ',end-start)
        print('Nuc, free, cross, t:', global_p.nuc,global_p.free,global_p.cross,t)
        # event_vis = [read(x) for x in event_list]
        # pevent_vis = [read(x) for x in pevent_list]
        #save state
        if save_file:
            lists = (mt_list, bdl_list, region_list, event_list, last_result, global_p, rnd.getstate())
            file = path+'simstate_seed'+str(seed)+'_'+str(final_hr)+'idx'
            pickle_out = open(file+'.pickle','wb')
            pickle.dump(lists,pickle_out)
            pickle_out.close()
            # if Result.policy not in ['disap', 'cross_bdl', 'uncross','cross_br', 'follow_br']:
            #     new_lists = purge_list2(mt_list, bdl_list, event_list, last_result,global_p)
            #     mt_list, bdl_list, event_list, last_result, global_p = new_lists[0], new_lists[1], new_lists[2], new_lists[3], new_lists[4]
            # plot_snap(mt_list,t,i,path+str(snap_idx),save=save_file,region=False, nodes=False)
