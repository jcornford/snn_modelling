import os
import numpy as np
import matplotlib.pyplot as plt
from neuron import h

from .basket_cell import BasketCell

class Basket4Integration(BasketCell):

        def __init__(self, morphology_file):
            BasketCell.__init__(self,morphology_file)
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
