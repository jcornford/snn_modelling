from neuron import h
import numpy as np

def get_valid_dendrites(pv_cell, min_max_distance):
    origin = h.distance(0,0.0,sec=pv_cell.root)
    possible_dends = set()
    for dend in pv_cell.dendrites:
        for seg in dend:
            distance = h.distance(seg.x, sec = dend)
            if np.logical_and(distance > min_max_distance[0], distance <min_max_distance[1]):
                possible_dends.add(dend)
    possible_dends = [dend for dend in possible_dends]

    return possible_dends

def get_uncaging_rois_per_dendrite(pv_cell,
                                   possible_dends,
                                   roi_area_micron,
                                   min_max_distance):
    # let's roll with ~ 30 µM area - should be really 20?
    u_exp_locations = {}
    u_exp_distances = {}
    early_distance_limits = {}
    for dend in possible_dends:
        #print dend.name()
        segs = [seg for seg in dend if np.logical_and(h.distance(seg.x, sec = dend) > min_max_distance[0], h.distance(seg.x, sec = dend) <min_max_distance[1])]
        distances = [h.distance(seg.x, sec = dend) for seg in segs]
        tot_distance = h.distance(segs[-1].x, sec = dend) - h.distance(segs[0].x, sec = dend)
        n_locs = int(np.rint(tot_distance /roi_area_micron)) # 35 is the region over which we are uncaging here here
        if n_locs: # trying to find uncaging locations per loaction
            #print tot_distance, 'total distance,', distances[0], distances[-1]
            #print n_locs, 'location/s'
            for i in range(n_locs):
                key = dend.name()+'_'+str(i)
                lower_threshold = distances[0] + ((i) * (tot_distance/ n_locs))
                upper_threshold =  distances[0] + ((i+1) * (tot_distance/ n_locs))
                #eary_distance_limits[key] = ('min:', lower_threshold,'max:', upper_threshold,  (upper_threshold-lower_threshold))
                early_distance_limits[key] = (lower_threshold,upper_threshold)
                segs_for_exp, distances_for_exp = [],[]
                for i,seg in enumerate(segs):
                    if np.logical_and(distances[i]>=lower_threshold,distances[i]<=upper_threshold):
                        segs_for_exp.append(seg) # seg and distances are in sync, to check print line below
                        #print(distances[i], h.distance(seg.x, sec = dend))
                        #print(seg.x)
                        distances_for_exp.append(distances[i]) # not using
                u_exp_locations[key] = segs_for_exp
                u_exp_distances[key] = distances_for_exp # not using
                #print(distances_for_exp)
    print(len(u_exp_locations.keys()), ' areas')
    return u_exp_locations

def get_param_dict(pv_cell, min_max_distance = (40, 200)):
    # code to set up uncaging experiments

    # first get dendrites that cover experimental distances
    possible_dends = get_valid_dendrites(pv_cell, min_max_distance)
    #print(len(possible_dends))

    # now work out how many 'uncaging spots per dendrite'
    # u_exp_locations[key] contains the a list of the segments to be used in each uncaging 'experiment'
    u_exp_locations = get_uncaging_rois_per_dendrite(pv_cell, possible_dends, roi_area_micron =30, min_max_distance = min_max_distance)

    # now work out the location range to pass to the make syanpses method
    dend_loc_range, final_midpoints = get_location_range_for_make_synapse_method(u_exp_locations, possible_dends)

    # make list of simulation parameters
    param_dict_list = []
    for flag in [False]:
        param_dict = {}
        for dend_name in dend_loc_range.keys():
            param_dict['dendrite_name'] = dend_name
            param_dict['location_selection'] = dend_loc_range[dend_name] # this is seg.x of (rounded ceil and floor)
            param_dict['passive_flag'] = flag
            param_dict['location_info'] = (final_midpoints[dend_name],early_distance_limits[dend_name])
            param_dict['n_segs'] = len(u_exp_locations[dend_name])

            #print(dend_loc_range[dend_name])
            if 'bdend' in dend_name:
                # oriens
                param_dict['nmda_gmax'] = 3500# 3500
            elif 'adend' in dend_name:
                # radiatum
                param_dict['nmda_gmax'] = 1000#1000
            param_dict_list.append(param_dict.copy())
            #print dend_name[:-2], dend_loc_range[dend_name]

def get_location_range_for_make_synapse_method(u_exp_locations, possible_dends):
    dend_loc_range = {}
    final_midpoints= {}
    for k in u_exp_locations.keys():
        #print(k)
        #print(u_exp_distances[k])
        #print (u_exp_locations[k][0].x)
        min_loc = np.floor(u_exp_locations[k][0].x*10)/10 # why this * and divide? embarassing rounding...
        #print(min_loc)
        max_loc = np.ceil(u_exp_locations[k][-1].x*10)/10
        mid_loc = ((max_loc-min_loc)/2.0) + min_loc
        #print u_exp_locations[k][0].x,u_exp_locations[k][-1].x
        dend_loc_range[k] = (min_loc,max_loc) # why not use the actual seg.x's? dont think this is optimal

        # calcuate middle dendrite part - don't you already know this though?/ need it?
        for dend in possible_dends:

            if dend.name() == k[:-2]:
                min_diff = 1
                mid_seg = None
                for seg in dend:
                    diff = abs(seg.x - mid_loc)
                    if min_diff > diff:
                        min_diff = diff
                        mid_seg = seg
                        #print (diff)
                #print('***,',min_diff)
                final_midpoints[k] = h.distance(mid_seg.x, sec = dend)

                if str(k[:-5]) == 'jc_tree2_adend2':
                    pass
                    #print k
                    #print h.distance(min_seg.x, sec = dend)
                    #print h.distance(u_exp_locations[k][0].x, sec = dend),
                    #print h.distance(u_exp_locations[k][-1].x, sec = dend)
        #print("***")

    # lets just check that distances here are the same as distances from above?
    return dend_loc_range, final_midpoints

