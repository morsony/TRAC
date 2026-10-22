import numpy as np
import pickle
import shelve
import analytic_afterglow_granot
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import integrate
from scipy import interpolate
from global_var import sysvars as gv


def integrate_total_power_interp(savefile, radio_f, z=0.47, z_origional=0.47, dl=209.8, xray_min=0.3, xray_max=2.0, optical_f='0', time='0'):

    h = 4.135667662e-15     # Plank's constant, eV*s (Demi 9/2026)
    to_ev = h /2/np.pi  # Reduced Plank's constant (Demi 9/2026)
    Mpc = 3.08e24  # Mpc in cm

    phi_bins=3600

    print(radio_f)
    
    # Convert optical r, i, and z-band frequencies from their wavelengths to ordinary frequency in Hz (Demi 9/2026)
    rband_f=1/np.array([(626e-7 + 69e-7), (626e-7), (626e-7 - 69e-7)])*h*gv.cl/(to_ev*2*np.pi)
    iband_f=1/np.array([(767e-7 + 67e-7), (767e-7), (767e-7 - 67e-7)])*h*gv.cl/(to_ev*2*np.pi)
    zband_f=1/np.array([(910e-7 + 69e-7), (910e-7), (910e-7 - 69e-7)])*h*gv.cl/(to_ev*2*np.pi)

    # Convert the selected X-ray photon energies (0.3–8 keV) to frequencies (Demi 9/2026)
    # in Hz so all observing bands use the same frequency coordinate. (Demi 9/2026)
    xray_f=np.array([3e2, 1e3, 3e3, 8e3])/(to_ev*2*np.pi)


    # If no optical frequency is provided use R-band instead (Demi 9/2026)
    if (optical_f=='0'):
        optical_f=rband_f


#Older Comments 

#z_origional = 0.047 # Redshift everything was run at

#z = 0.047 #1 #4 #10.0 #0.047 #0.5 # 0.047  # 0.09     # Redshift (need to know this for each file)

#dl = 209.8 #6701.2 #36594. #106118. #209.8 #2584.7 #209.8  # 414.2   
             # Luminosity distance in Mpc for z=0.09, calculated using Ned Wright's Cosmology Calculator (http://www.astro.ucla.edu/~wright/CosmoCalc.html)
             # with H_0 = 69.6, Omega_M = 0.286, Omega_vac = 0.714 
             # dl is Angular diameter distance times (1+z)**2, taking into account factor of (1+z) for time dialation, frequency change, 
             # and 2 for beaming, because source is moving away. - Brian, 4/14/2017

                                                                                                        #Older Comments^^^^

    #Load in previously stored data from binary file (Demi 9/2026)
    data_in = open(savefile, 'rb')
    data = pickle.load(data_in)
    data_in.close()


    # Load in Emission power, Freqency, and Time arrays (Demi 9/2026)
    total_power_arr=data['total_power_arr']      
    # [time,freq,phi] Load in central emission power array (Demi 9/2026)
    # units of erg/s/radian/Hz  (per Hz of angular frequency?) (Demi 9/2026)
    freq_arr=data['freq_arr']             
    # [time,freq,phi] Load in angular frequency array in source frame (Demi 9/2026)
    #[number of times, number of frequency elements, number of φ slices] (Demi 9/2026)
    t_arr=data['t_arr']                          
    # [time] Load in emission time array in source frame (Demi 9/2026)

    #Unclear on what glow does (Demi 9/2026)
    glow_arr=data['glow_arr']

    #print(total_power_arr[0,:,0])
    #print(freq_arr[0,:,0])
    #print(t_arr)

    where_are_NaNs = np.isnan(total_power_arr) # Identify NaN values in emission power array (Demi 9/2026)
    total_power_arr[where_are_NaNs] = 1e-99 # Replace NaN values with very small value, negligible emission (Demi 9/2026)
    total_power_arr[np.where(total_power_arr <= 0)] = 1e-99 # Replace 0's and negative values with the same floor value (Demi 9/2026)


    freq=np.copy(freq_arr[0,:,0])/(1+z)/2/np.pi  # Convert source frame frequency to observer frequency by considering red shift (Demi 9/2026)
    n_freq=len(freq)                             # number of frequencies observed

    #time=np.copy(t_arr)*(1+z)                    # emission time in observer frame

    #time = 10.**(np.arange(101)/100*8.)

    time_origional = np.copy(t_arr)*(1+z_origional)  # emission time in original file observer fame 

    if (time=='0'):
        time=time_origional

    phi_slices=len(total_power_arr[0,0,:])       # Determine the number of slices in the phi direction from the 3rd axis of the total power array
    # (Demi 9/2026)
    phi=(np.arange(phi_slices/2)/(phi_slices/2))**2 * 90.  # Reconstruct the phi boundaries used when generatinf the emission file 
    # Squared spacing allows for finer angular resolution near Phi = 0 degrees (Demi 9/2026)
    phi=np.concatenate((phi,[90],180-phi[::-1])) # Combine the angles into an array from 0 degrees to 180 degrees with 90 being our median (Demi 9/2026)            

    #    phi_gap=(np.floor(phi[1:]*(phi_bins/360))-np.floor(phi[0:len(phi)-1]*(phi_bins/360))) * (360/phi_bins) * 2.   # d_phi between all angles, multiply by two becuase we are only ding half a circle when we make data. - Brian 4/14/2017

    phi_gap=(phi[1:]-phi[0:len(phi)-1]) * 2   # Calculate the space in between each slice. Multiply by 2 since we're only using half the circle for our calculations (Demi 9/2026)

    print(phi_gap)
    
    phi_gap_arr=np.zeros([len(time_origional),len(phi_gap)])
    for i in range(len(time_origional)):
        phi_gap_arr[i,:]=np.copy(phi_gap)      # Repeat the phi_gap steps for the original time and fill array with those values (Demi 9/2026)
        # Shape: [time, phi slice].


    #where_are_NaNs = np.isnan(total_power_arr)
    #total_power_arr[where_are_NaNs] = 0

    # Converts the source frame emission power into observed spectral flux density at every time and frequency (Demi 9/2026)
    flux=np.zeros([len(time_origional),n_freq])
    
    # Integrate the source-frame spectral power over all phi slices.
    # Converting phi from degrees to radians
    # Convert the luminosity into flux density 
    for i in range(n_freq):
        flux[:,i] = np.sum(total_power_arr[:,i,:]*phi_gap_arr*np.pi/180,axis=1) * 2*np.pi / (4*np.pi*(dl*Mpc)**2)    
        
        
        # I think the 2*np.pi is becuase total_power_arr should be per Hz of angular freqnecy (omega), but we really want per Hz of freqency (nu) but I an not completely sure.  - Brian, 4/14/2017
        # One factor of 2*pi get picked up because total_power_arr is per radian, but a circle cover 2*pi radians.  This part comes from summing over phi_gap_arr. - Brian, 4/14/2014


    #print(flux[0,:])

# Now have flux in units of erg/s/cm^2/Hz


# for radio data:

    jy = np.copy(flux) * 1.e23   # Convert flux from cgs units to units of Janksys (Jy) (Demi 9/2026)


    # try 2d interpolation instead, allow for different time basis. - Brian, 7/26/2017

    #Initiate array for storing radio data (Demi 9/2026)
    radio=np.zeros((len(time),len(radio_f)))

#    print(np.log(freq))
#    print(np.log(time_origional))
    #print(np.log(jy[4,:]))


    #Construct a 2D log-log interpolation of flux density in Jy as a function of source-frame time and frequency. (Demi 9/2026) 
    interp_radio = interpolate.interp2d(np.log(freq),np.log(time_origional),np.log(jy[:,:]), bounds_error = False, kind='linear') # Approximates the flux locally as power law (Demi 9/2026)
    
    for t in range(len(time)):
        radio[t,:]=np.exp(interp_radio(np.log(radio_f),np.log(time[t])))   # Convert from Log(Jy) back to Jy and store in radio array (Demi 9/2026)

#    print(freq)
#    print(time_origional)
#    print(jy[0:1,:])
#    print(time[0])
#    print(radio[0,:])

# for X-ray data:

    # Want to integrate over some range in energy

    # Set X-ray limits (Demi 9/2026)
    l1=xray_f[0]/.3 * xray_min #.5  # Assumes xfreq[0] is 0.3 keV.  So lower limit is 0.5 keV
    l2=xray_f[0]/.3 * xray_max #2.  # Assumes xfreq[0] is 0.3 keV.  So upper limit is 2. keV
    
    # Create frequency grid (Demi 9/2026)
    xfreqi=10**((np.arange(101)/100)*(np.log10(l2)-np.log10(l1))+np.log10(l1))   # 101 log spaced points from .5 to 2 keV

    xray=np.zeros([len(time)])
    
    #Construct a 2D log-log interpolation of flux density in Jy as a function of source-frame time and frequency. (Demi 9/2026) 
    xray_interf = interpolate.interp2d(np.log(freq), np.log(time_origional), np.log(flux[:,:]), bounds_error = False)
    
    for t in range(len(time)):
        xray[t]=np.trapz(np.exp(xray_interf(np.log(xfreqi),np.log(time[t]))),x=xfreqi)   # Flux in erg/cm^2/s


# For optical:

    # Frequencies at which optical data was calculated
    
    #Initiate array for storing optical data (Demi 9/2026)
    # "optical" uses an r-band-specific zero point while "optical_ab" uses the standard AB magnitude zero point.
    optical=np.zeros([len(time)])
    optical_ab=np.zeros([len(time)])

    #Construct a 2D log-log interpolation of flux density in Jy as a function of source-frame time and frequency. (Demi 9/2026) 
    interp_jy = interpolate.interp2d(np.log(freq),np.log(time_origional),np.log(jy[:,:]), bounds_error = False)

    for t in range(len(time)):
        optical[t]=8.67-2.5*np.log10( np.trapz(np.exp(interp_jy(np.log(optical_f),np.log(time[t])))/(max(optical_f)-min(optical_f)),x=optical_f ))          # Convert the band-averaged flux density to AB magnitude using the standard AB zero-point flux density of 3631 Jy.
        optical_ab[t]=-2.5*np.log10( np.trapz(np.exp(interp_jy(np.log(optical_f),np.log(time[t]))),x=optical_f/(3631*(max(optical_f)-min(optical_f))) ))  # AB magnitude


    print('DONE WITH INTERPOLATION!!!!')                      

    # Return the interpolated radio flux densities, integrated X-ray flux, optical magnitudes, frequency/time grids, source flux-density grid, original emission times, and glow data.
    return radio, xray, optical, optical_ab, time, freq, jy, time_origional, glow_arr
