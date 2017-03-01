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

    def set_nsegs(self,frequency = 1000, d_lambda = 0.1):

        ''' Set the number of segments for section according to the
        d_lambda rule for a given input frequency'''

        for sec in self.sec_list:
            sec.nseg = int((sec.L / (d_lambda*h.lambda_f(frequency,sec=sec)) + .9)/ 2 )*2 + 1

    def biophysics(self,
                   Ra       =190,
                   cm       = 0.9,
                   gnabar   = 0.035,
                   gkbar    = 0.009,
                   gl       = 0.0001,
                   vshift   = -14.0,#0.0
                   el       = -65.0,
                   egk      = -90.0,
                   ):

        ''' The values for gnabar, gkbar and gl are the default
        values from the mod file 'hh_wbm from Jonas's 2001/2? paper.

        I have added vshift to the mod file
        '''
        self.leak_reversal = el
        for sec in self.sec_list:
            sec.Ra = 170
            sec.cm = 0.9

        # run the lamda rule having set Ra and cm
        frequency = 1000
        d_lambda = 0.1
        for sec in self.sec_list:
            sec.nseg = int((sec.L / (d_lambda*h.lambda_f(frequency,sec=sec)) + .9)/ 2 )*2 + 1

        # set origin for distance measurement
        origin = h.distance(0,0.0,sec=self.root)
        segment_count = 1
        distances_from_root = []
        for sec in self.dendrites:
            for seg in sec:
                dist = h.distance(seg.x,sec=sec) # x is the soma(0.5) proerty. location in a section
                distances_from_root.append(dist)

        border_threshold = np.max(distances_from_root)
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

    def select_recording_dendrite(self, dend_string):
        for section in self.sec_list:
            if section.name() == dend_string:
                self.dend_to_patch = section

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
