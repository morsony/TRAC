import numpy as np
import pickle
import shelve
from matplotlib import pyplot as plt
#from testpointbypoint2 import *
#from testpointbypoint2_file import *
from testpointbypoint2_file_lazzati_2017 import *
from blandford_mckee_full_brian import *
#from blandford_mckee_full_brian_davide_spring2021 import *
from afterglow_lineofsight_bmk_morsony_mem_reduced_small import *


# Where to save energy distibution
savefile = './savefile.short.GW170817.bestfit.p'   #.n-4
 
# Where to save final output
#outputfile = './afterglow_sych_spectrum.lazzati_2017.fitting.lowres.20deg.0.02ee.1.0e-3n.2.10p.bm2.p'   # .n-4
outputfile = './afterglow_sych_spectrum.GW170817.bestfit.p'   # .n-4


# Specify the jet internal parameters for electron energy, magnetic field energy, and electron powerlaw index.
    # these 3 parameters will change the spectrum only.
epsilon = 0.06899849433816842 #0.018385496124867763 #0.1        #electron energy fraction
eqparb = 0.0008071802460746388 #0.0012486418213311383 #0.01        #magnetic energy fraction
pindex = 2.1270281642904076 #2.1258562476698484 #2.5 #2.335643   # electron spectrum index

# This is the observer angle relative to the jet.  Better to use negative numbers.
    # posititon relative to jet axis ( degrees off axis).  Make angle negative to put center of jet at zero in phi slices
rot_angle = -21.512340070947978 #-22.7732977940544 #-25 #-0.0 

# Specify external density distribution.
    # external_nodensity is the density of the ISM for a flat ditribution, or 
    # the normalization parameter for a wind distribution.
external_nodensity = 0.0008774239400563791 #0.002926426490058228 #1.0e-3 #1.e-3 #1.0    ## for ISM
    
    # k_index is 0 for ISM external density, 2 for wind external density   
k_index = 0. #for ISM
    # m_index allows for impulsive or continuous energy injection in Blandford-McKee solution.  Just use implusive for now.
m_index = 3. - k_index  # for impulsive injection



# Set the times that you want to calculate you model at.  Ususally distributed in log space.
##t_arr=10.**(np.arange(10)/3.+6.)    # This will do 10 times, starting at 1e6 seconds and going to 1e9 seconds.
#num_t = 19 #7  # 10
#t_min = np.log10(1e5)  # 1e6)
#t_max = np.log10(1e8)  # 1e9)
#t_arr = np.logspace(t_min,t_max,num_t)


#num_t = 85 #31 #11   #7  # 10
#t_min = np.log10(1.e-2) #1e2)  # 1e6)
#t_max = np.log10(1.e12) #1e7)  # 1e9)

num_t = 73 #7  # 10
t_min = np.log10(1e-2)  # 1e6)
t_max = np.log10(1e10)  # 1e9)
t_arr = np.logspace(t_min,t_max,num_t)

    #t_arr=10.**(np.arange(num_t)/(num_t-1.)*3.+6.)
print(t_arr)


# These 3 parameters set the resoluton of the model you create in the phi, s and z directions
    # Number of bins in phi for the full energy calculation (i.e. number of independent slices)
phi_slices = 54  #18  # 18 is alright, 54 is probably better.  If you are doing an on-axis models, you an set this to 2.
    # Number of points in vertical direction to use for each slice (y points)
s_points = 30 #15 #30    # 30 seems okay
    # Number of points into the shell to use for each slice (x points)
z_points = 100 #50 #100   # 100 seems okay


# Set the frequency range to go over, usually in log space.  The array wmax is the range of frequency (omega) to go over, apparently in units of eV.  Probably don't need to change the range, but might change the number of frequencies
h = 4.135667662e-15     #eV*s   # constant to convert Hz to eV.  

n_elemf = 101    # Number of frequencies to calculate at
wmin0=1.e-11 #1.*h       # 8e7*h    # minimum frequency (in eV ?)
wmax0=1.e9 #8e15       # maximum frequency (in eV ?) 
wmax = 10**(np.arange(n_elemf)/(n_elemf-1)*np.log10(wmax0/wmin0))*wmin0    # changed form wmax to wmax0 - Brian, 4/10/2017  # removed - Brian, 4/6/2017



# Specify the resolution of the energy distribution.  Usually don't need to change these.
    #number of bins in the theta direction for the initial energy distribuition
bins = 1800
    # number of bins in phi for initial energy distribution
phi_bins = 360



# Set up the energy distribution.  Can either be a file from a simulation, or an analytic model.
    # If you are using a jet from a simulation, specifiy the file containing the energy distibution here
datafile='summarized_data_lazzati_2017.dat'    # Jet from short grb simulation
    #datafile='./grb/stall/3.e51_15.0_tracer/rhd_jet_quick_taub_hdf5_plt_cnt_0236'    
    #datafile='0'   # Use datafile='0' if you want to use an analytic jet.

#The following is if you are using an analytic jet model only.  If you are using a jet from a simulation, ignore this

    # half opening angle of GRB jet (degrees)
opening_angle = 10. #90. #16. #.05*180/np.pi #10. #16. # 10.


    # powerlaw index of energy vs. distribution within the jet
powerlaw_index = -0. #-1.

    #E_iso value at 1 degree from jet axis
eiso = 1.e54 #1e53 #3.1e49 #1.5e49 #1e47 #2.5e51 #1.7e51 # 1.e49  # 2.5e51 # 1.e51

    # center_angle is the size of the flat part in the center.  Need this so things don't diverge at the exis.
center_angle = 1.

    # eiso_off is the isotropic energy outside the jet.  Ususally, set to 1.e-15 * eiso
eiso_off = eiso*1.e-15 #1e35 #1e35 #1.8e41

    # gamma_init is the initial Lorentz factor of the jet material, before any interaction occurs.
    # Set to 'file' if you want to use the values from the data file.
gamma_init = 'file' #4. #400 #5.0 #1.7 #200.

    # shock_thikcness is the thickness of the shock, in light seconds
shock_thickness = 0.1  # * gv.cl 


# If you want the jet to physically spread sideways as it slows down, you can turn on spreading here.  
# This still needs more vetting. - Brian, 9/20/2018
    # You can trun on different formulas for jet speading.  0 = no spreading.
spread=0


## Create initial energy distribution

#datafile='0'

pointbypoint_powerlaw_varedge_spread(eiso, bins, phi_bins, rot_angle, powerlaw_index, savefile, opening_angle, eiso_off, center_angle, gamma_init, k_index, external_nodensity,spread=spread, datafile=datafile, var='gamma_inf', cutoff=2.)

#pointbypoint_powerlaw_varedge_spread(eiso, bins, phi_bins, rot_angle, powerlaw_index, savefile, opening_angle, eiso_off, center_angle, gamma_init, k_index, external_nodensity,spread=spread)







wmin = min(wmax)
n_elem = len(wmax)


# Redshift will change what frequencies you are looking at and the time.
##add redshift  correction for LIGO
# Don't change this.  It' just picking some default redshift, but the I shift everything when intergrating the data to make plots anyway.
z = 0.047 # 0.09   # 0.047 corresponds to co-moving radial distance of 200 Mpc (dl=210 Mpc), 0.09 to 380 Mpc (dl=414 Mpc)
t_arr = t_arr/(1+z)
wmax = wmax*(1+z)




eiso_floor = 0
    
# Run the actual code to make a model.

afterglow_aniso_bmk(savefile, s_points, z_points, phi_slices, outputfile, 
                        t_arr, wmax, wmin, n_elem, epsilon, eqparb, external_nodensity, 
                        k_index, m_index, eiso_floor, pindex, rot_angle, shock_thickness)
                        
print('DONE WITH CODE!!!!')                      
