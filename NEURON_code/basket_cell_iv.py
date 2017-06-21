import os
import numpy as np
import matplotlib.pyplot as plt
from neuron import h

from .basket_cell import BasketCell

class BasketCellIV(BasketCell):

    def __init__(self,morphology_file):
        BasketCell.__init__(self, morphology_file)

    def run_vclamp(self, vc_range, tstop = 50.0, ):
        h.load_file('stdrun.hoc')
        rec = {}
        for label in 't','v','i':
            rec[label] = h.Vector()

        rec['t'].record(h._ref_t)
        rec['i'].record(self.clamp._ref_i)
        rec['v'].record(self.root(0.5)._ref_v)

        v_list,i_list,t_list = [],[],[]
        h.tstop = tstop
        h.celsius = 32.0
        for vc in vc_range:
            h.init()
            #h.finitialize(v_init)
            self.clamp.amp2 = vc
            h.run()
            i = rec['i'].to_python()
            v = rec['v'].to_python()
            t = rec['t'].to_python()
            v_list.append(v)
            i_list.append(i)
            t_list.append(t)

        v = np.array(v_list).transpose()
        i = np.array(i_list).transpose()
        t = np.array(t_list).transpose()

        return v,i,t

    def get_synapse_iv(self, synpase_type = 'ampa', p_conc = 0,
               ampa_gmax = 5000, nmda_gmax = 5000, t_rel = 40.0, vc_range = np.arange(-90.0,50.0,5.0), nmda_mech = 'h.NMDA_Mg_T', vshift = None):

        '''

        Method for generating large IV curves... Not sure if best, use synapse __ref__i
        '''

        self.clamp = h.SEClamp(0.5,sec=self.root)
        self.clamp.dur1 = 10
        self.clamp.dur2 = 60
        self.clamp.dur3 = 10
        self.clamp.amp1= -70
        self.clamp.amp2 = -70
        self.clamp.amp3 = -70
        self.clamp.rs = 0.0001

        pre = h.Section()
        pre.diam = 1.0
        pre.L = 1.0
        pre.insert('rel')
        pre.dur_rel = 0.5
        pre.amp_rel = 3.0
        pre.del_rel = t_rel

        if synpase_type.lower() == 'ampa':
            print ( 'running simulation of AMPAR IV with',p_conc,'µM polyamines')

            cpampa = h.cpampa12st(0.5,sec=self.root)
            cpampa.pconc = p_conc
            cpampa.gmax  = ampa_gmax

            h.setpointer(pre(0.5).rel._ref_T,'C',cpampa)

        if synpase_type.lower() == 'nmda':
            print ('running simulation of NMDA IV')
            nmda = eval(nmda_mech+'(0.5,sec=self.root)')
            print('inserted: '+nmda_mech+'(0.5,sec=self.root)')
            #nmda = h.NMDA_Mg_T(0.5,sec=self.root)
            if vshift:
                print('vshift: '+str(vshift))
                nmda.vshift = vshift
            nmda.gmax =  nmda_gmax
            h.setpointer(pre(0.5).rel._ref_T,'C',nmda)


        v,i,t = self.run_vclamp(vc_range, tstop = 80)
        i_list, v_list = self.calc_iv_relationship(v,i,t)
        return i_list, v_list,

    def get_ampa_iv_2017(self, ampa_gmax, p_conc = 0, vc_range = [-60,60], tstop = 220,t_rel = 40.0 ):
        self.clamp = h.SEClamp(0.5,sec=self.root)
        self.clamp.dur1 = 10
        self.clamp.dur2 = 200
        self.clamp.dur3 = 10
        self.clamp.amp1= -70
        self.clamp.amp2 = -70
        self.clamp.amp3 = -70
        self.clamp.rs = 0.0001

        pre = h.Section()
        pre.diam = 1.0
        pre.L = 1.0
        pre.insert('rel')
        pre.dur_rel = 0.5
        pre.amp_rel = 3.0
        pre.del_rel = t_rel
        print('amp 3')

        cpampa = h.cpampa12st(0.5,sec=self.root)
        cpampa.pconc = p_conc
        cpampa.gmax  = ampa_gmax

        h.setpointer(pre(0.5).rel._ref_T,'C',cpampa)

        h.load_file('stdrun.hoc')
        rec = {}
        for label in 't','v','i_ampa','g_ampa':
            rec[label] = h.Vector()

        rec['t'].record(h._ref_t)
        rec['i_ampa'].record(cpampa._ref_i)
        rec['g_ampa'].record(cpampa._ref_g)
        rec['v'].record(self.root(0.5)._ref_v)

        v_list,i_list,t_list,g_list = [],[],[],[]
        h.tstop = tstop
        h.celsius = 32.0

        for vc in vc_range:
            h.init()
            #h.finitialize(v_init)
            self.clamp.amp2 = vc
            h.run()
            i = rec['i_ampa'].to_python()
            g = rec['g_ampa'].to_python()
            v = rec['v'].to_python()
            t = rec['t'].to_python()
            v_list.append(v)
            i_list.append(i)
            t_list.append(t)
            g_list.append(g)

        v = np.array(v_list).transpose()
        i = np.array(i_list).transpose()*1000 # for pA
        t = np.array(t_list).transpose()
        g = np.array(g_list).transpose() * 1*10**6 # this is defined as microSiemens in the mod file: 1000000, so for pico 1e-6

        return t,i,v,g  # in pA

    def get_nmda_iv_2017(self, nmda_gmax, vc_range = [-60,60], tstop = 220, t_rel = 40.0 ):
        self.clamp = h.SEClamp(0.5,sec=self.root)
        self.clamp.dur1 = 10
        self.clamp.dur2 = 200
        self.clamp.dur3 = 10
        self.clamp.amp1= -70
        self.clamp.amp2 = -70
        self.clamp.amp3 = -70
        self.clamp.rs = 0.0001

        pre = h.Section()
        pre.diam = 1.0
        pre.L = 1.0
        pre.insert('rel')
        pre.dur_rel = 0.5
        pre.amp_rel = 3.0
        pre.del_rel = t_rel

        nmda = h.NMDA_Mg_T(0.5,sec=self.root)
        # overwriting briefly... testing difference before  changing to ampa properly 2017_05_26
        # also for comapring presyantuic glutamte
        #print('overwritten for glu!')
        #nmda = h.cpampa12st(0.5,sec=self.root)
        #nmda.pconc = 0
        nmda.gmax  = nmda_gmax

        h.setpointer(pre(0.5).rel._ref_T,'C',nmda)

        h.load_file('stdrun.hoc')
        rec = {}
        for label in 't','v','i_nmda','g_nmda':
            rec[label] = h.Vector()

        rec['t'].record(h._ref_t)
        rec['i_nmda'].record(nmda._ref_i)
        rec['g_nmda'].record(nmda._ref_g)
        rec['v'].record(self.root(0.5)._ref_v)

        v_list,i_list,t_list,g_list = [],[],[],[]
        h.tstop = tstop
        h.celsius = 32.0
        for vc in vc_range:
            h.init()
            #h.finitialize(v_init)
            self.clamp.amp2 = vc
            h.run()
            i = rec['i_nmda'].to_python()
            g = rec['g_nmda'].to_python()
            v = rec['v'].to_python()
            t = rec['t'].to_python()
            v_list.append(v)
            i_list.append(i)
            t_list.append(t)
            g_list.append(g)

        v = np.array(v_list).transpose()
        i = np.array(i_list).transpose()*1000 # for pA
        t = np.array(t_list).transpose()
        g = np.array(g_list).transpose()

        return t,i,v,g # in pA



    def calc_iv_relationship(self, v,i,t):
        '''
        Hardcoded for when release is at 40ms into sweep.
        Note that vcomm is rounded to nearest int.
        '''
        t_trace = t[:,0]
        bl_index = np.logical_and(t_trace>=30,t_trace<=35)
        iv_index = np.logical_and(t_trace>=30,t_trace<=65)
        time = t_trace[iv_index]
        i_cut = np.subtract(i[iv_index], np.mean(i[bl_index],axis = 0))
        v_cut = v[iv_index]

        v_list = np.rint(np.mean(v_cut, axis = 0)) # round as very close!
        i_list = []
        for index in range(i_cut.shape[1]):
            sweep = i_cut[:,index]
            x = max(abs(sweep))
            if x == abs(np.min(sweep)):
                x = np.min(sweep)
            elif x == np.max(sweep):
                x = x
            i_list.append(x)
        return i_list, v_list

    @staticmethod
    def central_origin_plot(ylabel = 'pA',xlabel = 'mV',arrows=False, textSize = 15,tickTextSize = 15 ):
        '''
        Args:
        the art1, and art2 that are returned are for saving the labels inside the plot.
        call plt.savefig('pyramidalIV.pdf',bbox_extra_artists=(art1,art2), bbox_inches='tight')
        '''

        ax = plt.gca()
        for  location in ["left", "bottom"]:
            ax.spines[location].set_position('zero')

        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(labelsize = tickTextSize)
        ax.yaxis.set_tick_params(labelsize = tickTextSize)

        for spine in ['right', 'top']:
            ax.spines[spine].set_color('none')
        arrow_length = 10
        art1 = ax.annotate(ylabel, xy=(0, 1), xycoords=('data', 'axes fraction'),
                        xytext=(0, arrow_length), textcoords='offset points', fontsize = textSize)
        art2 = ax.annotate(xlabel, xy=(1,0), xycoords=('axes fraction','data'),
                        xytext=(0, arrow_length), textcoords='offset points',fontsize = textSize)
        if arrows:
            #-- Decorate the spins
            arrow_length = 10# In points

            # X-axis arrow & Label
            ax.annotate('', xy=(1, 0), xycoords=('axes fraction', 'data'),
                        xytext=(arrow_length, 0), textcoords='offset points',
                        arrowprops=dict(arrowstyle='<|-', fc='black'),fontsize = textSize)

            # Y-axis arrow & Label

            ax.annotate('', xy=(0, 1), xycoords=('data', 'axes fraction'),
                        xytext=(0, arrow_length), textcoords='offset points',
                        arrowprops=dict(arrowstyle='<|-', fc='black'),fontsize = textSize)

        return art1,art2

    def get_nmda_ampa_ratio(self, p_conc,
               ampa_gmax, nmda_gmax, t_rel = 40.0, vc_range = [-60,60]):

        self.clamp = h.SEClamp(0.5,sec=self.root)
        self.clamp.dur1 = 10
        self.clamp.dur2 = 150
        self.clamp.dur3 = 10
        self.clamp.amp1= -70
        self.clamp.amp2 = -70
        self.clamp.amp3 = -70
        self.clamp.rs = 0.0001

        pre = h.Section()
        pre.diam = 1.0
        pre.L = 1.0
        pre.insert('rel')
        pre.dur_rel = 0.5
        pre.amp_rel = 3.0
        pre.del_rel = t_rel

        syn_list = []
        locs = np.linspace(0,1,35)
        for l in locs:
            cpampa = h.cpampa12st(l,sec=self.root)
            cpampa.pconc = p_conc
            cpampa.gmax  = ampa_gmax

            h.setpointer(pre(0.5).rel._ref_T,'C',cpampa)

            nmda = h.NMDA_Mg_T(l,sec=self.root)
            nmda.gmax =  nmda_gmax
            h.setpointer(pre(0.5).rel._ref_T,'C',nmda)

            syn_list.append((nmda,cpampa))
        
        v,i,t = self.run_vclamp(vc_range, tstop = 170)

        return v,i,t

    def get_iv(self, synpase_type = 'ampa', p_conc = 0,
               ampa_gmax = 5000, nmda_gmax = 5000, t_rel = 40.0):

        self.clamp = h.SEClamp(0.5,sec=self.root)
        self.clamp.dur1 = 10
        self.clamp.dur2 = 60
        self.clamp.dur3 = 10
        self.clamp.amp1= -70
        self.clamp.amp2 = -70
        self.clamp.amp3 = -70
        self.clamp.rs = 0.0001

        pre = h.Section()
        pre.diam = 1.0
        pre.L = 1.0
        pre.insert('rel')
        pre.dur_rel = 0.5
        pre.amp_rel = 3.0
        pre.del_rel = t_rel

        if synpase_type.lower() == 'ampa':
            print('running simulation of AMPAR IV with',p_conc,'µM polyamines')

            cpampa = h.cpampa12st(0.5,sec=self.root)
            cpampa.pconc = p_conc
            cpampa.gmax  = ampa_gmax

            h.setpointer(pre(0.5).rel._ref_T,'C',cpampa)

        if synpase_type.lower() == 'nmda':
            print ('running simulation of NMDA IV')

            nmda = h.NMDA_Mg_T(0.5,sec=self.root)
            nmda.gmax =  nmda_gmax
            h.setpointer(pre(0.5).rel._ref_T,'C',nmda)

        vc_range = np.arange(-90.0,50.0,10.0)
        v,i,t = self.run_vclamp(vc_range, tstop = 80)
        i_list, v_list = self.calc_iv_relationship(v,i,t)
        return i_list, v_list
