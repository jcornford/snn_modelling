import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from neuron import h
from neuron import hoc
from neuron import load_mechanisms

from PIL import Image, ImageDraw

def iclamp_sweep(cell_object,
                 iclamp_amp,
                 stim_obj,
                 iclamp_delay = 150,
                 iclamp_dur = 200,
                 tstop = 500,
                 v_init = -65.0,
                 degC = 32):
    
    ''' Sweeps a cell with multiple values of stim.app connected to cell_object.root
        Returns arrays of the voltage, current, and time'''
    
    rec = {} # make a dictionary called rec
    for label in 't','v','i':
        rec[label] = h.Vector()
    
    rec['t'].record(h._ref_t)  
    rec['v'].record(cell_object.root(0.5)._ref_v)     
    rec['i'].record(stim_obj._ref_i)

    stim_obj.delay = iclamp_delay
    stim_obj.dur = iclamp_dur
    stim_obj.amp = iclamp_amp

    h.load_file('stdrun.hoc')
    h.tstop = tstop
    h.v_init = v_init
    h.celsius = degC
    h.run()

    i = rec['i'].to_python()
    v = rec['v'].to_python()
    t = rec['t'].to_python()
    
    results = pd.DataFrame(np.array([t,i,v]).T, columns = ['Time (ms)','I_inj (nA)','V (mV)'] )
    return results


def sublinear_exp_1(stim1_locs, stim2_locs):
    pv_cell = Basket4Integration('jc_tree2.hoc')
    pv_cell.biophysics(Ra=170, cm = 0.9,gnabar= 0,gkbar= 0,gl= 1./7000.0, vshift= 0.0)
    pv_cell.add_cpampars_for_stim_electrodes(stim1_locs,      50, 'stim1',gmax = 1000, pconc = 100)
    stim1_waveform_cp, time = pv_cell.simulate(pandas = False)

    pv_cell = Basket4Integration('jc_tree2.hoc')
    pv_cell.biophysics(Ra=170, cm = 0.9,gnabar= 0,gkbar= 0,gl= 1./7000.0, vshift= 0.0)
    pv_cell.add_cpampars_for_stim_electrodes(stim2_locs,      50, 'stim2',gmax = 1000, pconc = 100)
    stim2_waveform_cp = pv_cell.simulate(pandas = False)[0]

    pv_cell = Basket4Integration('jc_tree2.hoc')
    pv_cell.biophysics(Ra=170, cm = 0.9,gnabar= 0,gkbar= 0,gl= 1./7000.0, vshift= 0.0)
    pv_cell.add_cpampars_for_stim_electrodes(stim1_locs,      50, 'stim1',gmax = 1000, pconc = 100)
    pv_cell.add_cpampars_for_stim_electrodes(stim2_locs,      50, 'stim2',gmax = 1000, pconc = 100)
    sum_waveform_cp = pv_cell.simulate(pandas = False)[0]

    pv_cell = Basket4Integration('jc_tree2.hoc')
    pv_cell.biophysics(Ra=170, cm = 0.9,gnabar= 0,gkbar= 0,gl= 1./7000.0, vshift= 0.0)
    pv_cell.add_cpampars_for_stim_electrodes(stim1_locs,      50, 'stim1',gmax = 450, pconc = 0)
    stim1_waveform_wt= pv_cell.simulate(pandas = False)[0]

    pv_cell = Basket4Integration('jc_tree2.hoc')
    pv_cell.biophysics(Ra=170, cm = 0.9,gnabar= 0,gkbar= 0,gl= 1./7000.0, vshift= 0.0)
    pv_cell.add_cpampars_for_stim_electrodes(stim2_locs,      50, 'stim2',gmax = 450, pconc = 0)
    stim2_waveform_wt= pv_cell.simulate(pandas = False)[0]

    pv_cell = Basket4Integration('jc_tree2.hoc')
    pv_cell.biophysics(Ra=170, cm = 0.9,gnabar= 0,gkbar= 0,gl= 1./7000.0, vshift= 0.0)
    pv_cell.add_cpampars_for_stim_electrodes(stim1_locs,      50, 'stim1',gmax = 450, pconc = 0)
    pv_cell.add_cpampars_for_stim_electrodes(stim2_locs,      50, 'stim2',gmax = 450, pconc = 0)
    sum_waveform_wt = pv_cell.simulate(pandas = False)[0]

    data_array = np.hstack([time[:,None],stim1_waveform_cp[:,None],stim2_waveform_cp[:,None],\
                       sum_waveform_cp[:,None],stim1_waveform_wt[:,None],\
                       stim2_waveform_wt[:,None],sum_waveform_wt[:,None]])
    results_frame = pd.DataFrame(data_array,columns = ['time','stim1_cp','stim2_cp','sum_cp', 'stim1_wt',
                                           'stim2_wt','sum_wt'])

    return results_frame

def get_sublinearity_percent(results_dframe):
    stim1_cp = results_dframe.stim1_cp.max() - results_dframe.stim1_cp.min()
    stim2_cp = results_dframe.stim2_cp.max()- results_dframe.stim2_cp.min()
    sum_cp = results_dframe.sum_cp.max()- results_dframe.sum_cp.min()
    expected_cp = stim1_cp+stim2_cp
    print '100uM Polyamines:',((sum_cp/expected_cp)-1) *100,'%'

    stim1_wt = results_dframe.stim1_wt.max() - results_dframe.stim1_wt.min()
    stim2_wt = results_dframe.stim2_wt.max()- results_dframe.stim2_wt.min()
    sum_wt = results_dframe.sum_wt.max()- results_dframe.sum_wt.min()
    expected_wt = stim1_wt+stim2_wt
    print 'Without Polyamines:',((sum_wt/expected_wt)-1) *100,'%'


class Basket_cell():

    def __init__(self,morphology_file,verbose = False,Jonas_cell = False):
        assert os.path.exists(morphology_file)

        self.cpampa_list = []

        h.load_file('stdlib.hoc', 'String')
        h.load_file('import3d.hoc')
        self.morphology = morphology_file # 'saraga_2006_cella.hoc', #morphology = '190809c4.swc'
        self.verbose = verbose
        self._load_morphology(Jonas_cell) # load the morphology and make sectionlists
        self.set_nsegs() # check this works? - delete
        #segment info above?
        self.biophysics()
        self.unique_dendrites = [dend for dend in set(dendrite.name().split('[')[0] for dendrite in self.dendrites)]

    def _load_morphology(self,Jonas_cell):
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
        self.nsec = 0
        self.dendrites = h.SectionList() # maybe these should be h.SectionList()? rather than python lists
        self.axon = h.SectionList()
        self.soma = h.SectionList()
        self.root = 'none'

        for sec in h.allsec():
            self.all_section_names.append(sec.name())
            self.sec_list.append(sec=sec)
            self.nsec +=1
            # Set up categories for different cell regions
            if sec.name().find('soma') >= 0:
                self.soma.append(sec=sec)
                if sec.name().find('0') >= 0:
                    self.root = sec
                    if self.verbose:
                        print sec.name(),'is root'
                elif Jonas_cell:
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
            print 'Biophysics method reporting in: '
            print 'Max dendrite distance from soma = ',np.max(distances_from_root)
            print 'Border threshold (1/3 max), set at: ', border_threshold/3
            for section in self.sec_list:
                print section.nseg, 'segments in ',section.name()

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

    def add_cpampars(self,locations,dendrite_strings,gmax = 500,release_times = [20],pconc = 100,
        NMDARs = False):
        self.cpampa_list = []
        self.pre_cpampa_list = []
        self.NMDARs_list = []
        # find the dendrite of choice!
        for string in dendrite_strings:
            for sec in self.sec_list:
                if sec.name().find(string) >=0:
                    self.dendrite = sec
            for x,loc in enumerate(locations):
                pre = h.Section()
                pre.diam = 1.0
                pre.L = 1.0
                pre.insert('rel')
                pre.dur_rel = 0.5
                pre.amp_rel = 1.0
                pre.del_rel = release_times[x]
                cpampa = h.cpampa12st(locations[x],sec=self.dendrite)
                cpampa.pconc = pconc
                cpampa.gmax  = gmax

                if NMDARs:
                    NMDA = h.NMDA_Mg_T(locations[x],sec=self.dendrite)
                    NMDA.gmax = gmax
                    h.setpointer(pre(0.5).rel._ref_T,'C',NMDA)
                    self.NMDARs_list.append(NMDA)
                h.setpointer(pre(0.5).rel._ref_T,'C',cpampa)

                self.cpampa_list.append(cpampa)
                self.pre_cpampa_list.append(pre)

    def print_cpampa_locations(self):
        for i in self.cpampa_list:
            pyseg = i.get_segment()
            print pyseg.sec.name(),pyseg.x

    def print_sec_seg_numbers(self):
        self.segment_list = []
        self.section_names = []
        for section in self.sec_list:
            self.section_names.append(section.name())
            for seg in section:
                self.segment_list.append(seg)
        print len(self.section_names),':sections. ',len(self.segment_list),':segments.'

    def select_recording_dendrite(self, dend_string):
        for section in self.sec_list:
            if section.name() == dend_string:
                self.dend_to_patch = section


class Basket4Integration(Basket_cell):

        def __init__(self, morphology_file):
            Basket_cell.__init__(self,morphology_file)
            self.cpampa_list = []
            self.cpampa_stim_dict = {}
            self.pre_cpampa_list = []
            self.dend_to_patch = None

        ### These two methods should be done in a more clever way...!
        def add_cpampars(self, locations, section_name, release_time, gmax = 500,pconc = 100):

            for section in self.dendrites:
                if section.name() == section_name:
                    self.dendrite = section
            #print 'adding synapses to', self.dendrite.name()

            pre = h.Section()
            pre.diam = 1.0
            pre.L = 1.0
            pre.insert('rel')
            pre.dur_rel = 0.5
            pre.amp_rel = 1.0
            pre.del_rel = release_time
            self.pre_cpampa_list.append(pre)

            for loc in locations:
                cpampa = h.cpampa12st(loc,sec=self.dendrite)
                cpampa.pconc = pconc
                cpampa.gmax  = gmax
                h.setpointer(pre(0.5).rel._ref_T,'C',cpampa)
                self.cpampa_list.append(cpampa)

        def add_cpampars_for_stim_electrodes(self, section_loc_list, release_time,stim_string, gmax = 500, pconc = 100):
            '''
            Method adds across sections, expects a list of tuples containing (section(str), loc(float)).
            Stores the stims in a dictionary...
            '''
            # should make this a method
            pre = h.Section()
            pre.diam = 1.0
            pre.L = 1.0
            pre.insert('rel')
            pre.dur_rel = 0.5
            pre.amp_rel = 1.0
            pre.del_rel = release_time
            self.pre_cpampa_list.append(pre)

            stim_syns = []
            for sec_loc in section_loc_list:
                section_name = sec_loc[0]
                loc = sec_loc[1]
                #print 'adding at:', section_name, loc
                for section in self.dendrites:
                    if section.name() == section_name:
                        dendrite = section
                cpampa = h.cpampa12st(loc,sec=dendrite)
                cpampa.pconc = pconc
                cpampa.gmax  = gmax
                h.setpointer(pre(0.5).rel._ref_T,'C',cpampa)
                stim_syns.append(cpampa)
            self.cpampa_stim_dict[stim_string] = stim_syns
            # hacky just go with last dendrite for recording - change this!
            self.dendrite = dendrite
        ####
        def simulate(self, tstop = 100.0, dendrite_to_record = False, pandas = False):
            if dendrite_to_record:
                for section in self.dendrites:
                    if section.name() == dendrite_to_record:
                        self.dendrite = section
            if self.dend_to_patch:
                print 'havent overwritten the simulate rdend argument - use it!'
            h.load_file('stdrun.hoc')

            self.rec = {}
            for label in 't','v','vdend':
                self.rec[label] = h.Vector()
            self.rec['t'].record(h._ref_t)
            self.rec['vdend'].record(self.dendrite(0.5)._ref_v)
            self.rec['v'].record(self.root(0.5)._ref_v)

            h.tstop = tstop
            h.v_init = self.leak_reversal
            h.celsius = 32.0
            h.dt = 0.01
            h.run()

            v_dend = self.rec['vdend'].to_python()
            v_soma = self.rec['v'].to_python()
            t= self.rec['t'].to_python()

            results = pd.DataFrame(np.array([t,v_soma,v_dend]).T, columns = ['time','v_soma','v_dend'] )
            if pandas:
                return results
            else:
                if dendrite_to_record:
                    return np.array(v_soma),np.array(t), np.array(v_dend)
                else:
                    return np.array(v_soma),np.array(t)

        def _get_error(self, gmax, verbose = 'False'):
            for syn in self.cpampa_list:
                syn.gmax = gmax
            results = self.simulate()
            measured_vshift = results['v_soma'].max() - results['v_soma'].min()
            error = self.vshift-measured_vshift

            if verbose:
                print gmax,'ps: error',error,
                print results['v_dend'].max()
            return error, results['v_dend'].max(), results['v_soma'].max()

        def calc_gmax_for_vshift(self,gmax_start, vshift = 10, verbose = False):
            self.vshift = vshift
            test_gmax = gmax_start
            error = 10.0
            run = 0
            while abs(error) > 1:
                test_gmax += (error+2) *2000
                error, mxdend,mxsoma = self._get_error(test_gmax,verbose = verbose)
                run +=1
                if run >20:
                    break
            print test_gmax,'pS gmax,', error,'error,', mxdend,'mV dend,', mxsoma,'mV soma'
            return test_gmax

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



        def plot_morphology(self,
                            figsize_tuple = (10,10),
                            synapses = False,
                            color_dendrites = False,
                            synapse_marker_r = 5,
                            synapse_marker_alpha = 0.5,
                            plot_electrodes = False,
                            xy = (512,512),
                            dpatch_left = False,
                            selection_string = 'stim_dict'):

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

                self.cp = sns.color_palette('hls', len(self.unique_dendrites))
                self.cp = {'b' :(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
                           'r' :(0.7686274509803922, 0.3058823529411765, 0.3215686274509804)}
                self.dend_to_index = {str(dend):0 for dend in self.unique_dendrites}
                for dend in self.unique_dendrites:
                    if str(dend).split('_')[2][0] == 'a':

                        self.dend_to_index[str(dend)] = 1


            # plot the morph
            for sec in h.allsec():
                if color_dendrites:
                    if sec.name().split('[')[0] == 'jc_tree2_soma':
                        fill_color = (0,0,0)
                    else:
                        dname = sec.name().split('[')[0]
                        if dname.split('_')[2][0] == 'a':
                              fill_color = tuple([int(f*255.0) for f in (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])
                        else:
                              fill_color = tuple([int(f*255.0) for f in (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)])

                else:
                    fill_color = (0,0,0)
                x,y,z,d = self.get_coordinates(sec)
                x +=50
                xy = zip(x,y)
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
                    for key in self.cpampa_list:
                        cpampa_list = self.cpampa_list
                        for syn in cpampa_list:
                            syn_loc = syn.get_segment().x
                            sec = syn.get_segment().sec
                            x,y = self.get_2d_position(sec, syn_loc)
                            plt.plot(x, y,'o',
                                     alpha = synapse_marker_alpha,
                                     color = syn_color,
                                     markersize = synapse_marker_r)

            if plot_electrodes:
                xs,ys = self.get_2d_position(self.root, 0.5)
                self.plot_electrode(xs,ys,'k')
                if self.dend_to_patch:
                    xd,yd = self.get_2d_position(self.dend_to_patch, 0.5)
                    self.plot_electrode(xd,yd,'grey',dpatch_left,linestyle = '--')



            plt.imshow(np.asarray(im))
            plt.axis('off')

        @staticmethod
        def plot_electrode(x,y,color = 'k', dpatch_left = False, dx=125,
                           dy = 12, lw = 1.5, gap = 1,linestyle = '-'):
            if dpatch_left:
                dx = -dx
            plt.arrow(x,y+gap, dx,  dy, fc=color, ec = color, linewidth = lw,linestyle =linestyle )
            plt.arrow(x,y-gap, dx, -dy, fc=color, ec = color, linewidth = lw,linestyle = linestyle)


class simple_cell():

    def __init__(self):
        self.soma = h.Section(name = 'soma')
        self.soma.nseg = 10
        self.soma.diam = 10
        self.soma.L = 10
        self.soma.Ra = 100
        self.soma.insert('pas')

        self.synapse_list = []
        self.pre_list = []

    def add_cpampa(self, pconc = 100 ,gmax = 1000, release_time = 50):
        syn = h.cpampa12st(0.5,sec=self.soma)
        syn.pconc = pconc
        syn.gmax = gmax

        pre = h.Section()
        pre.diam = 1.0
        pre.L=1.0
        pre.insert('rel')
        pre.dur_rel = 0.5
        pre.amp_rel = 2.0
        pre.del_rel = release_time

        h.setpointer(pre(0.5).rel._ref_T,'C',syn)

        self.syn = syn
        self.pre = pre

    def add_vclamp(self):
        clamp = h.SEClamp(0.5,sec=self.soma)
        clamp.dur1 = 10
        clamp.dur2 = 10
        clamp.dur3 = 10
        clamp.amp1= -70
        clamp.amp2 = -70
        clamp.amp3 = -70
        clamp.rs = 0.0001

        self.clamp = clamp

    def simulate(self, tstop = 100.0, vinit = -70.0):
        self.rec = {}
        for label in 't','v','syn_i':
            self.rec[label] = h.Vector()
        self.rec['t'].record(h._ref_t)
        self.rec['v'].record(self.soma(0.5)._ref_v)
        self.rec['syn_i'].record(self.syn._ref_i)
        h.load_file('stdrun.hoc')
        h.tstop = tstop
        h.v_init = vinit
        h.celsius = 32.0
        h.dt = 0.01
        h.run()
        v_soma = self.rec['v'].to_python()
        t= self.rec['t'].to_python()
        syn_i = self.rec['syn_i'].to_python()

        return t, v_soma, syn_i

        