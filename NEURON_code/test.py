#from basket_cell import BasketCell
from neuron import h
from neuron import hoc
from neuron import load_mechanisms
import numpy as np

import os

def main():
    pv_cell = BasketCell('jc_tree2.hoc')

    from neuron import gui

class NeuronModel():

    def get_2d_position(self, sec, loc, xoffset = 50):
        '''
        method to return the 2d coordinates of a location,
        between 0-1, along a given section.
        '''
        x,y,z,d = self.get_coordinates(sec)
        x += xoffset
        x_arc =  np.append(np.zeros((1)),np.cumsum(abs(np.diff(x))))
        y_arc =  np.append(np.zeros((1)),np.cumsum(abs(np.diff(y))))
        z_arc =  np.append(np.zeros((1)),np.cumsum(abs(np.diff(z))))
        arc_len = x_arc+y_arc+z_arc
        arc_res = np.linspace(0,arc_len[-1],101)
        xres = np.interp(arc_res, arc_len,x)
        yres = np.interp(arc_res, arc_len,y)
        zres = np.interp(arc_res, arc_len,z)
        i =  int((loc*100))
        return (xres[i], yres[i])

    @staticmethod
    def get_coordinates(sec):
        sec.push() # make section the currently activated section
        x, y, z, d = [],[],[],[]
        # loop through 3d locations in the currently accessed section.
        for i in range(int(h.n3d())):
            x.append(h.x3d(i))
            y.append(h.y3d(i))
            z.append(h.z3d(i))
            d.append(h.diam3d(i))
        h.pop_section()
        return (np.array(x),np.array(y),np.array(z),np.array(d))

    @staticmethod
    def plot_electrode(x,y,lcolor = 'k', dpatch_left = False, dx=125,
                       dy = 12, lw = 1, gap = 1, linestyle = '-'):
        if dpatch_left:
            dx = -dx
        plt.plot((x, x+dx),(y+gap, y+dy), color = lcolor, linewidth = lw, linestyle = linestyle)
        plt.plot((x, x+dx),(y-gap, y-dy), color = lcolor, linewidth = lw, linestyle = linestyle)


    def plot_morphology(self,
                            figsize_tuple = (10,10),
                            synapses = False,
                            color_dendrites = False,
                            synapse_marker_r = 5,
                            synapse_marker_alpha = 0.5,
                            plot_electrodes = False,
                            xy = (512,512),
                            dpatch_left = False,
                            selection_string = 'cpampa_list',
                            e_color = 'k',
                            elw     = 1):

        '''
        This needs to be cleaned up a little.
        '''

        plt.figure(figsize = figsize_tuple)
        im = Image.new('RGB',xy , (255, 255, 255))
        draw = ImageDraw.Draw(im)

        mc_ints = {'b' :(77, 117, 179),
              'r' :(210, 88, 88),
              'k' :(38,35,35),
              'grey':(197,198,199)}

        mc_f = {key: tuple([x/255.0 for x in mc_ints[key]]) for key in mc_ints.keys()}

        if color_dendrites:
            import seaborn as sns
            self.cp = sns.color_palette('hls', len(self.unique_dendrites))
            self.dend_to_index = {str(dend):i for i,dend in enumerate(self.unique_dendrites)}

        # plot the morph
        for sec in h.allsec():
            if color_dendrites:
                if sec.name().split('[')[0] == 'jc_tree2_soma':
                    fill_color = (0,0,0)
                else:
                    fill_color = tuple([int(f*255.0) for f in self.cp[self.dend_to_index[sec.name().split('[')[0]]]])
            else:
                fill_color = (0,0,0)
            x,y,z,d = self.get_coordinates(sec)
            x +=50
            xy = list(zip(x,y))
            for i in range(len(x)-1):
                draw.line((xy[i], xy[i+1]), fill = fill_color, width = int(d[i]))

        if synapses:
            if selection_string == 'stim_dict':
                syncolor_dict = {'stim1':mc_f['r'],'stim2':mc_f['b'] }
                for key in self.cpampa_stim_dict.keys():
                    cpampa_list = self.cpampa_stim_dict[key]
                    for syn in cpampa_list:
                        syn_loc = syn.get_segment().x
                        sec = syn.get_segment().sec
                        x,y = self.get_2d_position(sec, syn_loc)
                        plt.plot(x, y,'o',
                                 alpha = synapse_marker_alpha,
                                 color = syncolor_dict[key],
                                 markersize = synapse_marker_r)
                # make legend
                stim1_patch = mpatches.Patch(color = syncolor_dict['stim1'], label = 'stim1')
                stim2_patch = mpatches.Patch(color = syncolor_dict['stim2'], label = 'stim2')
                plt.legend(handles = [stim1_patch, stim2_patch])

            elif selection_string == 'cpampa_list':
                syn_color = mc_f['r']

                for syn in self.cpampa_list:
                    syn_loc = syn.get_segment().x
                    sec = syn.get_segment().sec
                    x,y = self.get_2d_position(sec, syn_loc)
                    plt.plot(x, y,'o',
                             alpha = synapse_marker_alpha,
                             color = syn_color,
                             markersize = synapse_marker_r)

        if plot_electrodes:
            xs,ys = self.get_2d_position(self.root, 0.5)
            self.plot_electrode(xs,ys, lcolor = e_color, lw = elw)
            if self.dend_to_patch:
                xd,yd = self.get_2d_position(self.dend_to_patch, 0.5)
                self.plot_electrode(xd,yd,lcolor = 'grey', dpatch_left=dpatch_left,lw = elw, linestyle = '--')



        plt.imshow(np.asarray(im))
        plt.axis('off')

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

        self.set_nsegs(frequency=1000, d_lambda=0.1)

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

if __name__=='__main__':
    main()