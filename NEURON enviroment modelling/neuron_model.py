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
    def plot_electrode(x,y,color = 'k', dpatch_left = False, dx=125,
                       dy = 12, lw = 1.5, gap = 1, linestyle = '-'):
        if dpatch_left:
            dx = -dx
        plt.plot((x, x+dx),(y+gap, y+dy), color = 'k', linewidth = 1.5, linestyle = linestyle)
        plt.plot((x, x+dx),(y-gap, y-dy), color = 'k', linewidth = 1.5, linestyle = linestyle)
                
        
    def plot_morphology(self,
                            figsize_tuple = (10,10),
                            synapses = False,
                            color_dendrites = False,
                            synapse_marker_r = 5,
                            synapse_marker_alpha = 0.5,
                            plot_electrodes = False,
                            xy = (512,512),
                            dpatch_left = False, 
                            selection_string = 'cpampa_list'):
        
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