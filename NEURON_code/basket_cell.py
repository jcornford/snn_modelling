import os

import numpy as np
import matplotlib.pyplot as plt
from neuron import h

from .neuron_model import NeuronModel

class BasketCell(NeuronModel):

    def __init__(self,morphology_file = 'jc_tree2.hoc'):
        self.verbose = False
        assert os.path.exists(morphology_file)
        h.load_file('stdlib.hoc', 'String')
        h.load_file('import3d.hoc')
        self.morphology = morphology_file # 'saraga_2006_cella.hoc', #morphology = '190809c4.swc'
        self._load_morphology()           # load the morphology and make sectionlists
        self.set_nsegs()                  # check this works? - delete
        #segment info above?
        self.biophysics()
        self.unique_dendrites = [dend for dend in set(dendrite.name().split('[')[0] for dendrite in self.dendrites)]

        self.cpampa_list = []
        self.nmda_list = []
        self.pre_list= []

        self.dend_to_patch = self.root
        self.u_dend = None

    def _load_morphology(self):
        ''' internal function: loads morphology and creates section lists'''
        fileEnding = self.morphology.split('.')[-1]
        if fileEnding == 'hoc':
            h.load_file(1, self.morphology)
        else:
            h('objref this') # why do i need this?
            if fileEnding == 'swc':
                Import = h.Import3d_SWC_read()
                Import.input(self.morphology)
                imprt = h.Import3d_GUI(Import, 0)
            imprt.instantiate(h.this)
        h.define_shape() # not sure what this does either

        # set up section names: You need to clarify differences between h.allsec(), h.SectionList() - just makes a new #sectionlist - to do with if have multple cells?
        self.all_section_names = []
        self.sec_list = h.SectionList()
        self.dendrites = h.SectionList()
        self.axon = h.SectionList()
        self.soma = h.SectionList()
        self.root = 'none'

        self.nsec = 0
        for sec in h.allsec():
            self.all_section_names.append(sec.name())
            self.sec_list.append(sec=sec)
            self.nsec +=1
            # Set up categories for different cell regions
            if sec.name().find('soma') >= 0:
                self.soma.append(sec=sec)
                if sec.name().find('0') >= 0:
                    self.root = sec
            if sec.name().find('dend') >=0:
                self.dendrites.append(sec=sec)
            if sec.name().find('axon') >=0:
                self.dendrites.append(sec=sec)

    def get_section_diameters(self):

        diameter_dict = {}
        for sec in self.sec_list:
            sec_diameters = []
            for seg in sec:
                sec_diameters.append(sec.diam)
            diameter_dict[sec.name()] = sec_diameters

        return diameter_dict

    def increase_dendrite_diameter(self, factor, reset_dlamda = True):
        '''
        diameter will effect the length constant, so you should redo the calculations for number of segments
        after playing with the diameter

        happily changing segment numbers inherits the parameters! Though you should double check this to be sure.
        '''
        for sec in self.dendrites:
            for seg in sec:
                seg.diam *= factor
        if reset_dlamda:
            self.set_nsegs()


    def calculate_segment_lengths(self):
        """
        Calculates the lengths of all the segments using the segments area and diameter
        self.segment_lengths
        """
        self.segment_lengths = []
        for sec in self.sec_list:
            for seg in sec:
                area = seg.area()
                r = seg.diam/2
                l = (area - 2 *np.pi * (r**2))/(2*np.pi*r)
                self.segment_lengths.append(l)

    def increase_nseg_by_factor(self, multiplication_factor = 3):
        # Starting in version 3.2, a change to nseg re-uses information contained in the old segments.
        self.total_nseg = 0
        for sec in self.sec_list:
            sec.nseg *=multiplication_factor
            self.total_nseg += sec.nseg

    def set_nsegs(self, frequency = 1000, d_lambda = 0.1):

        ''' Set the number of segments for section according to the
        d_lambda rule for a given input frequency

        Inputs:
        frequency: [1000]: AC frequency for method 'lambda_f'
        d_lambda: [0.1]: parameter for d_lambda rule
        '''
        self.total_nseg = 0
        for sec in self.sec_list:
            sec.nseg = int((sec.L / (d_lambda*h.lambda_f(frequency,sec=sec)) + .999)/ 2 )*2 + 1
            self.total_nseg += sec.nseg

    def biophysics(self,
                   Ra       =170,
                   cm       = 0.9,
                   gnabar   = 200*10**-4,
                   gkbar    = 300*10**-4,
                   gl       = 1./5000.0,
                   vshift   = -12.0,
                   el       = -65.0,
                   egk      = -90.0,
                   ):
        ''' .

        20170303 This used to be default values in mod file:
        values from the mod file 'hh_wbm from Jonas's 2001/2? paper
        - 0.035 gnabar, gkbar 0.009, gl 0.0001, vshift -14, el -65,egk = -90

        - mod file units are mho/cm2

        Now we are using:
         - 200 ps µm-2 gnabar  which translates to 200*10**-4, or 0.02
         - 300 ps µm-2 gkbar   which translates to 300*10**-4, or 0.03, this is much bigger... check the origins of 0.009
         - gl 1/5000 for 5 kohm cm2 or 0.0002



        '''
        self.leak_reversal = el
        for sec in self.sec_list:
            sec.Ra = Ra
            sec.cm = cm

        # run the lamda rule having set Ra and cm
        # todo this should be after you have set membrane resistance (though you call a second time anyway)
        self.set_nsegs(frequency=1000, d_lambda=0.1)

        # set origin for distance measurement
        # todo this works even though my reading of the docs would require root to be passed?
        origin = h.distance(0,0.0,sec=self.root)
        segment_count = 1
        distances_from_root = []
        for sec in self.dendrites:
            for seg in sec:
                dist = h.distance(seg.x,sec=sec) # x is the soma(0.5) proerty. location in a section
                distances_from_root.append(dist)

        #border_threshold = np.max(distances_from_root)
        border_threshold = 120
        if self.verbose:
            print ('Biophysics method reporting in: ')
            print ('Max dendrite distance from soma = ',np.max(distances_from_root))
            print ('Border threshold (1/3 max), set at: ', border_threshold/3)
            for section in self.sec_list:
                print (section.nseg, 'segments in ',section.name())

        # having calculated distances, insert conductances.
        for sec in self.dendrites:
            sec.insert('hh_wbm')
            for seg in sec:
                dist = h.distance(seg.x, sec=sec)

                if dist >= border_threshold:
                    seg.hh_wbm.gnabar = gnabar/2
                    seg.hh_wbm.gkbar  = gkbar
                    seg.hh_wbm.gl     = gl/10
                    seg.hh_wbm.el     = el
                    seg.hh_wbm.egk     = egk
                    seg.hh_wbm.vshift = vshift

                else:
                    seg.hh_wbm.gnabar = gnabar
                    seg.hh_wbm.gkbar  = gkbar
                    seg.hh_wbm.gl     = gl
                    seg.hh_wbm.el     = el
                    seg.hh_wbm.egk     = egk
                    seg.hh_wbm.vshift = vshift
                segment_count += 1

            segment_count = 1
        for sec in self.soma:
            sec.insert('hh_wbm')
            for seg in sec:
                seg.hh_wbm.gnabar = gnabar*5
                seg.hh_wbm.gkbar  = gkbar
                seg.hh_wbm.gl     = gl
                seg.hh_wbm.el     = el
                seg.hh_wbm.egk     = egk
                seg.hh_wbm.vshift = vshift

        self.set_nsegs(frequency=1000, d_lambda=0.1)

    def select_recording_dendrite(self, dend_string):
        for section in self.sec_list:
            if section.name() == dend_string:
                self.dend_to_patch = section

    def calc_r_input(self,iamp = 0.025, return_arrays = False):

        v,t, i = self.iclamp_sweep(iamp, iclamp_delay=50, iclamp_dur = 200, tstop=300.0)
        v_change = abs( abs(min(v)) -abs(max(v)))
        rinput_mohm_pos = (v_change/abs(iamp*1000))*1000 # first 1k converts na to pa, second 1k gohm to mohm

        v2,t2, i2 = self.iclamp_sweep(-iamp, iclamp_delay=50, iclamp_dur = 200, tstop=300.0)
        v_change2 = abs( abs(min(v2)) -abs(max(v2)))
        rinput_mohm_neg = (v_change2/abs(iamp*1000))*1000

        self.input_resistance = np.mean([rinput_mohm_neg,rinput_mohm_pos])

        print('Input Resistance: Neg:' +str(rinput_mohm_neg)+' Pos: ' + str(rinput_mohm_pos)+ ' mean: '+str(self.input_resistance) )

        if return_arrays:
            return [t,v,v2,i,i2]

    def iclamp_sweep(self, iclamp_amp, iclamp_delay =25, iclamp_dur=50, v_init = -65.00, tstop = 100.0):
        h.load_file('stdrun.hoc')

        self.iclamp = h.IClamp(0.5,sec = self.root)
        rec = {}
        for label in 't','v','i':
            rec[label] = h.Vector()

        rec['t'].record(h._ref_t)
        rec['v'].record(self.root(0.5)._ref_v)
        rec['i'].record(self.iclamp._ref_i)

        self.iclamp.delay = iclamp_delay
        self.iclamp.dur = iclamp_dur
        self.iclamp.amp = iclamp_amp

        h.tstop = tstop
        h.v_init = v_init
        h.celsius = 32.0
        #print(h.dt)
        h.dt = 0.01
        h.run()

        i = rec['i'].to_python()
        v = rec['v'].to_python()
        t = rec['t'].to_python()

        return np.array(v),np.array(t), np.array(i)

    def run_simulation(self,tstop = 100.0,v_init = -65.0):
        h.load_file('stdrun.hoc')
        rec = {}
        for label in 't','v','vdend':
            rec[label] = h.Vector()
        rec['t'].record(h._ref_t)
        rec['vdend'].record(self.dend_to_patch(0.5)._ref_v)
        rec['v'].record(self.root(0.5)._ref_v)

        h.tstop = tstop
        h.celsius = 32.0
        h.dt = 0.01
        h.finitialize(v_init)
        h.run()

        v_dend = rec['vdend'].to_python()
        v_soma = rec['v'].to_python()
        t      = rec['t'].to_python()

        return np.array(v_soma),np.array(t), np.array(v_dend)

    def add_uncaging_spots(self,
                          random_seed = 7,
                          n_synapses = 15,
                          uncaging_dendrite = 'jc_tree2_bdend1[2]',
                          ampa_gmax = 500,
                          nmda_gmax = 500,
                          rel_time = 20,
                          delay = 0.5,
                          pconc = 100,
                          nmda = False,
                          spots = 'all',
                          loc_range = (0,1),
                          wipe_lists = True):

        # grab dendrite
        for section in self.sec_list:
            if section.name() == uncaging_dendrite:
                self.u_dend = section

        # do these neeed to be here?
        if wipe_lists:
            self.cpampa_list = []
            self.nmda_list = []
            self.pre_list= []
        ##

        locs = np.linspace(loc_range[0],loc_range[1],n_synapses)

        np.random.seed(random_seed)
        possible_locs = np.random.permutation(locs)

        release_times = np.array([rel_time + delay*i for i in range(n_synapses)])

        if spots == 'all':
            print ('uncaging on all spots:')
            uncaging_spots = possible_locs
        else:
            uncaging_spots = possible_locs[spots]
            release_times = release_times[spots]

        for i,loc in enumerate(uncaging_spots):
            pre = h.Section()
            pre.diam = 1.0
            pre.L = 1.0
            pre.insert('rel')
            pre.dur_rel = 0.5
            pre.amp_rel = 3.0
            pre.del_rel = release_times[i]
            cpampa = h.cpampa12st(loc,sec=self.u_dend)
            cpampa.pconc = pconc
            cpampa.gmax  = ampa_gmax

            h.setpointer(pre(0.5).rel._ref_T,'C',cpampa)

            self.cpampa_list.append(cpampa)
            self.pre_list.append(pre)

            if nmda:
                    nmda = h.NMDA_Mg_T(loc,sec=self.u_dend)
                    nmda.gmax = nmda_gmax
                    h.setpointer(pre(0.5).rel._ref_T,'C',nmda)
                    self.nmda_list.append(nmda)

    def get_uncaging_distances(self, dendrite = False):
        origin = h.distance(0,0.0,sec=self.root)
        distances = []
        for syn in self.cpampa_list:
            distance = h.distance(syn.get_segment().x, sec = self.u_dend)
            distances.append(distance)
        print (min(distances),'to', max(distances), 'from soma')
        print ('Middle is', min(distances) + (max(distances) - min(distances))/2.0)
        print ('inter spot distances are', np.diff(sorted(np.unique(distances))))
        print (len(np.unique(distances)), 'unique spots for', len(self.cpampa_list), 'synapses')

        if dendrite:
            print ('Dendrite has', self.u_dend.nseg, 'segments:')
            for i, segment in enumerate(self.u_dend):
                distance = h.distance(segment.x, sec = self.u_dend)
                print ('Seg', i, 'is', distance, 'microns from soma')

    def stim_electrodes(self):
        print ('not implemented - maybe second model')
