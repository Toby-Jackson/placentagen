
from unittest import TestCase

import numpy as np
import unittest
import placentagen
import os
TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'Testdata/Small.exnode')
TESTDATA_FILENAME1 = os.path.join(os.path.dirname(__file__), 'Testdata/Small.exelem')

class Test_Terminal_Br(TestCase):
        
    def test_terminal_br(self):
        eldata   = placentagen.import_exelem_tree(TESTDATA_FILENAME1)
        noddata = placentagen.import_exnode_tree(TESTDATA_FILENAME)
        term_br  = placentagen.calc_terminal_branch(noddata['nodes'],eldata['elems'])
        print(term_br['terminal_elems'])
        self.assertTrue(term_br['terminal_elems'][0] == 1)
        
    def test_terminal_br_total(self):
        eldata   = placentagen.import_exelem_tree(TESTDATA_FILENAME1)
        noddata = placentagen.import_exnode_tree(TESTDATA_FILENAME)
        term_br  = placentagen.calc_terminal_branch(noddata['nodes'],eldata['elems'])
        self.assertTrue(term_br['total_terminals'] == 2)

      
class test_pl_vol_in_grid(TestCase):
        
    def test_pl_vol_margin(self):
        thickness =  (3.0 * 1 / (4.0 * np.pi)) ** (1.0 / 3.0) * 2.0  # mm
        ellipticity = 1.00  # no units
        spacing = 1.0  # mm
        volume=1
        rectangular_mesh = {}
        rectangular_mesh['nodes'] = [[0., 0., 0.], [ thickness/2.0, 0., 0.],[0., thickness/2.0, 0.],[ thickness/2.0, thickness/2.0, 0.],[0., 0., thickness/2.0], [ thickness/2.0, 0., thickness/2.0],[0., thickness/2.0,thickness/2.0],[ thickness/2.0, thickness/2.0, thickness/2.0]]
        rectangular_mesh['elems'] = [[ 0,  0,  1,  2,  3,  4, 5, 6, 7]]
        rectangular_mesh['total_nodes'] =8
        rectangular_mesh['total_elems'] = 1
        pl_vol=placentagen.ellipse_volume_to_grid(rectangular_mesh, volume, thickness, ellipticity, 25)
        self.assertTrue(np.isclose(pl_vol['pl_vol_in_grid'][0], 0.12485807941))
        self.assertTrue(abs(pl_vol['pl_vol_in_grid']-1./8.)/(1./8)<1e-2)#looking for less than 1% error in expected volume of 1/8

    def test_pl_vol_complete_inside(self):
        thickness =  2  # mm
        ellipticity = 1.6  # no units
        spacing = 0.5  # mm
        volume=5
        rectangular_mesh = {}
        rectangular_mesh['nodes'] = [[-1., -1.5, -1.],[-0.5 ,-1.5, -1.],[-1., -1., -1.] ,[-0.5,-1., -1.],[-1.,-1.5,-0.5],[-0.5,-1.5,-0.5],[-1.,-1.,-0.5] ,[-0.5,-1.,-0.5]]
        rectangular_mesh['elems'] = [[0, 0, 1, 2, 3, 4, 5, 6, 7]]
        rectangular_mesh['total_nodes'] =8
        rectangular_mesh['total_elems'] = 1
        pl_vol=placentagen.ellipse_volume_to_grid(rectangular_mesh, volume, thickness, ellipticity, 25)
        self.assertTrue(np.isclose(pl_vol['pl_vol_in_grid'][0], 0.0))
      

    def test_pl_vol_complete_outside(self):
        thickness =  2  # mm
        ellipticity = 1.6  # no units
        spacing = 0.5  # mm
        volume=5
        rectangular_mesh = {}
        rectangular_mesh['nodes'] = [[-0.5,-0.5,-0.5],[ 0., -0.5,-0.5],[-0.5, 0.,-0.5],[ 0., 0. ,-0.5],[-0.5, -0.5 ,0. ],[ 0., -0.5 ,0.],[-0.5, 0.,0.],[0.,0.,0.]]
        rectangular_mesh['elems'] = [[0,  0,  1,  2,  3,  4, 5, 6, 7]]
        rectangular_mesh['total_nodes'] =8
        rectangular_mesh['total_elems'] = 1
        pl_vol=placentagen.ellipse_volume_to_grid(rectangular_mesh, volume, thickness, ellipticity, 0.125)
        self.assertTrue(np.isclose(pl_vol['pl_vol_in_grid'][0], spacing*spacing*spacing))
 

class Test_terminals_in_sampling_grid_fast(TestCase):
        
    def test_terminals_in_grid_present(self):
        noddata = placentagen.import_exnode_tree(TESTDATA_FILENAME)
        term_br={}
        term_br['terminal_nodes']=[3]
        term_br['total_terminals']=1
        rectangular_mesh = {}
        rectangular_mesh['nodes'] =np.array( [[ 0.,  0.,  0.],[ 1.,  0. , 0.],[ 0.,  1. , 0.],[ 1. , 1. , 0.],[ 0.,  0. , 1.],[ 1.,  0. , 1.],[ 0. , 1. , 1.],[ 1. , 1. , 1.]])
        rectangular_mesh['elems']=[[0, 0, 1, 2, 3, 4, 5, 6, 7]]
        term_grid =placentagen.terminals_in_sampling_grid_fast(rectangular_mesh, term_br, noddata['nodes'])
        self.assertTrue(term_grid['terminals_in_grid'][0] == 1)
        
     
    def test_terminal_elems_present(self):
        noddata = placentagen.import_exnode_tree(TESTDATA_FILENAME)
        term_br={}
        term_br['terminal_nodes']=[3]
        term_br['total_terminals']=1
        rectangular_mesh = {}
        rectangular_mesh['nodes'] =np.array( [[ 0.,  0.,  0.],[ 1.,  0. , 0.],[ 0.,  1. , 0.],[ 1. , 1. , 0.],[ 0.,  0. , 1.],[ 1.,  0. , 1.],[ 0. , 1. , 1.],[ 1. , 1. , 1.]])
        rectangular_mesh['elems']=[[0, 0, 1, 2, 3, 4, 5, 6, 7]]
        term_grid =placentagen.terminals_in_sampling_grid_fast(rectangular_mesh, term_br, noddata['nodes'])
        self.assertTrue(term_grid['terminal_elems'][0] == 0)#this zero does not mean branch are not located. it means samp_grid el 0


class Test_terminals_in_sampling_grid_general(TestCase):

    def test_terminals_in_grid_general_present(self):
        noddata = placentagen.import_exnode_tree(TESTDATA_FILENAME)
        term_br = {}
        term_br['terminal_nodes'] = [3]
        term_br['total_terminals'] = 1
        placenta_list = [7]
        rectangular_mesh = {}
        rectangular_mesh['elems'] = np.zeros((8, 9), dtype=int)
        rectangular_mesh['nodes'] = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.], [0., 0., 1.], [1., 0., 1.],
                                     [0., 1., 1.], [1., 1., 1.]]
        rectangular_mesh['elems'][7] = [0, 0, 1, 2, 3, 4, 5, 6, 7]
        term_grid = placentagen.terminals_in_sampling_grid(rectangular_mesh, placenta_list, term_br, noddata['nodes'])
        self.assertTrue(term_grid['terminals_in_grid'][7] == 1)

    def test_terminals_elem_general_present(self):
        noddata = placentagen.import_exnode_tree(TESTDATA_FILENAME)
        term_br = {}
        term_br['terminal_nodes'] = [3]
        term_br['total_terminals'] = 1
        placenta_list = [7]
        rectangular_mesh = {}
        rectangular_mesh['elems'] = np.zeros((8, 9), dtype=int)
        rectangular_mesh['nodes'] = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.], [0., 0., 1.], [1., 0., 1.],
                                     [0., 1., 1.], [1., 1., 1.]]
        rectangular_mesh['elems'][7] = [0, 0, 1, 2, 3, 4, 5, 6, 7]
        term_grid = placentagen.terminals_in_sampling_grid(rectangular_mesh, placenta_list, term_br, noddata['nodes'])
        self.assertTrue(term_grid['terminal_elems'][0] == 7)

    def test_terminals_in_grid_general_absent(self):
        noddata = placentagen.import_exnode_tree(TESTDATA_FILENAME)
        eldata = placentagen.import_exelem_tree(TESTDATA_FILENAME1)
        term_br = placentagen.calc_terminal_branch(noddata['nodes'], eldata['elems'])
        placenta_list = [1]
        rectangular_mesh = {}
        rectangular_mesh['elems'] = np.zeros((8, 9), dtype=int)
        rectangular_mesh['nodes'] = [[0., -1., -1.], [1., -1., -1.], [0., 0., -1.], [1., 0., -1.], [0., -1., 0.],
                                     [1., -1., 0.], [0., 0., 0.], [1., 0., 0.]]
        rectangular_mesh['elems'][1] = [0, 0, 1, 2, 3, 4, 5, 6, 7]
        term_grid = placentagen.terminals_in_sampling_grid(rectangular_mesh, placenta_list, term_br, noddata['nodes'])
        self.assertTrue(
            np.sum(term_grid['terminals_in_grid']) == 0)  # all must be zero as could not locate any terminal br

    def test_terminals_elem_general_absent(self):
        noddata = placentagen.import_exnode_tree(TESTDATA_FILENAME)
        eldata = placentagen.import_exelem_tree(TESTDATA_FILENAME1)
        term_br = placentagen.calc_terminal_branch(noddata['nodes'], eldata['elems'])
        placenta_list = [1]
        rectangular_mesh = {}
        rectangular_mesh['elems'] = np.zeros((8, 9), dtype=int)
        rectangular_mesh['nodes'] = [[0., -1., -1.], [1., -1., -1.], [0., 0., -1.], [1., 0., -1.], [0., -1., 0.],
                                     [1., -1., 0.], [0., 0., 0.], [1., 0., 0.]]
        rectangular_mesh['elems'][1] = [0, 0, 1, 2, 3, 4, 5, 6, 7]
        term_grid = placentagen.terminals_in_sampling_grid(rectangular_mesh, placenta_list, term_br, noddata['nodes'])
        self.assertTrue(
            np.sum(term_grid['terminal_elems']) == 0)  # all must be zero as could not locate any terminal br
      
     
if __name__ == '__main__':
   unittest.main()
