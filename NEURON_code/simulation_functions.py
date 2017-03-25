from neuron import h
from neuron import hoc
from neuron import load_mechanisms

import numpy as np
import pandas as pd
from scipy.stats import sem
import multiprocessing

import os
import random
import sys

from .basket_cell import BasketCell


def arithemetic_sum(n_syn = 15, exp_dendrite = 'jc_tree2_bdend1[2]',p_conc = 0,
                    nmda_flag = False, u_delay = 1, nmda_gmax =3000,ampa_gmax = 2000,
                    loc_range = (0,1), passive_flag =False ):

    v_soma_list, v_dend_list = [],[]
    for i in range(n_syn):
        pv_cell = BasketCell('jc_tree2.hoc')
        if passive_flag:
            pv_cell.biophysics(Ra=170, cm = 0.9,gnabar= 0,gkbar= 0, gl= 1./7000.0, vshift= 0.0)
        pv_cell.select_recording_dendrite(exp_dendrite)
        pv_cell.add_uncaging_spots(spots = [i],
                                   ampa_gmax=ampa_gmax,
                                   nmda_gmax = nmda_gmax,
                                   pconc = p_conc,
                                   n_synapses=n_syn,
                                   uncaging_dendrite = exp_dendrite,
                                   delay = u_delay,
                                   nmda = nmda_flag,
                                   loc_range = loc_range)
        v,t,d = pv_cell.run_simulation()
        v_soma_list.append(v)
        v_dend_list.append(d)
    v_soma = np.array(v_soma_list).T
    v_dend = np.array(v_dend_list).T

    # now calculate arithemetic sum
    bl_index = np.logical_and(t>=15,t<=18)
    summed_waveform_soma = np.zeros(v_soma[:,0].shape)
    summed_waveform_dend = np.zeros(v_soma[:,0].shape)
    arithmetic_soma_list,arithmetic_dend_list = [],[]
    for i in range(n_syn):
        soma_trace = v_soma[:,i] - np.mean(v_soma[bl_index, i])
        dend_trace = v_dend[:,i] - np.mean(v_dend[bl_index, i])
        summed_waveform_soma += soma_trace
        summed_waveform_dend += dend_trace
        arithmetic_soma_list.append(summed_waveform_soma.copy())
        arithmetic_dend_list.append(summed_waveform_dend.copy())

    arithmetic_soma = np.array(arithmetic_soma_list).T
    arithmetic_dend = np.array(arithmetic_dend_list).T
    return arithmetic_soma,arithmetic_dend,t

def increasing_forward(n_syn = 15, exp_dendrite = 'jc_tree2_bdend1[2]',p_conc = 0,
                    nmda_flag = False, u_delay = 1, nmda_gmax =3000,ampa_gmax = 2000,
                      loc_range = (0,1),passive_flag =False):

    v_soma_list, v_dend_list = [],[]
    for i in range(n_syn):
        if_list = [ii for ii in range(i+1)]
        pv_cell = BasketCell('jc_tree2.hoc')
        if passive_flag:
            pv_cell.biophysics(Ra=170, cm = 0.9,gnabar= 0,gkbar= 0, gl= 1./7000.0, vshift= 0.0)
        pv_cell.select_recording_dendrite(exp_dendrite)
        pv_cell.add_uncaging_spots(spots = if_list,
                                   ampa_gmax=ampa_gmax,
                                   nmda_gmax = nmda_gmax,
                                   pconc = p_conc,
                                   n_synapses=n_syn,
                                   uncaging_dendrite = exp_dendrite,
                                   delay = u_delay,
                                   nmda = nmda_flag,
                                  loc_range = loc_range)
        v,t,d = pv_cell.run_simulation()
        bl_index = np.logical_and(t>=15,t<=18)
        v_soma_list.append(v - np.mean(v[bl_index]))
        v_dend_list.append(d - np.mean(d[bl_index]))
    v_soma = np.array(v_soma_list).T
    v_dend = np.array(v_dend_list).T
    return v_soma, v_dend, t

def quantify_nonlinearity_i2(measured, as_sum):
    n = measured.shape[0]
    result = 0
    for i in range(1, n):
        result += (measured[i]/as_sum[i] - 1)
    result = 100 * (result / (n-1)) # n-1 as we miss one off
    return result

def uncaging_simulation(n_syn = 15, exp_dendrite = 'jc_tree2_bdend1[2]',p_conc = 0,
                        nmda_flag = False, u_delay = 1, nmda_gmax =3000,ampa_gmax = 2000,
                        loc_range = (0,1), passive_flag = False):

    as_soma, as_dend, t = arithemetic_sum(n_syn = n_syn, exp_dendrite = exp_dendrite,
                                          p_conc = p_conc,nmda_flag = nmda_flag,
                                          u_delay = u_delay, nmda_gmax =nmda_gmax,
                                          ampa_gmax = ampa_gmax,loc_range = loc_range,
                                          passive_flag = passive_flag)

    m_soma, m_dend, t = increasing_forward(n_syn = n_syn, exp_dendrite = exp_dendrite,
                                           p_conc = p_conc,nmda_flag = nmda_flag,
                                           u_delay = u_delay, nmda_gmax =nmda_gmax,
                                           ampa_gmax = ampa_gmax,loc_range = loc_range,
                                           passive_flag = passive_flag)
    as_maxes = np.max(as_soma,axis = 0)
    m_maxes = np.max(m_soma,axis = 0)
    as_maxes_dend = np.max(as_dend,axis = 0)
    m_maxes_dend = np.max(m_dend,axis = 0)
    non_lin_soma      = quantify_nonlinearity_i2(m_maxes,as_maxes)
    non_lin_dend      = quantify_nonlinearity_i2(m_maxes_dend,as_maxes_dend)
    results_dict = {'as_soma':as_soma,
                    'as_dend':as_dend,
                    'm_soma' :m_soma,
                    'm_dend' :m_dend,
                    'time'   :t,
                    'm_maxes':m_maxes,
                    'as_maxes':as_maxes,
                    'as_maxes_dend': as_maxes_dend,
                    'm_maxes_dend':m_maxes_dend,
                    'non_lin_dend':non_lin_dend,
                    'non_lin_soma':non_lin_soma,
                   }
    return results_dict
