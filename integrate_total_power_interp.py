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

    h = 4.135667662e-15     #eV/s
    to_ev = 4.1356691e-15 /2/np.pi  #convert omega(frequency) to eV
    Mpc = 3.08e24  # Mpc in cm

    phi_bins=3600

    print(radio_f)
    
    #radio_f=np.array([8.635e8,1.4e9,1.2e10])*h/(to_ev*2*np.pi)
    rband_f=1/np.array([(626e-7 + 69e-7), (626e-7), (626e-7 - 69e-7)])*h*gv.cl/(to_ev*2*np.pi)
    iband_f=1/np.array([(767e-7 + 67e-7), (767e-7), (767e-7 - 67e-7)])*h*gv.cl/(to_ev*2*np.pi)
    zband_f=1/np.array([(910e-7 + 69e-7), (910e-7), (910e-7 - 69e-7)])*h*gv.cl/(to_ev*2*np.pi)
    xray_f=np.array([3e2, 1e3, 3e3, 8e3])/(to_ev*2*np.pi)


    if (optical_f=='0'):
        optical_f=rband_f


#z_origional = 0.047 # Redshift everything was run at

#z = 0.047 #1 #4 #10.0 #0.047 #0.5 # 0.047  # 0.09     # Redshift (need to know this for each file)

#dl = 209.8 #6701.2 #36594. #106118. #209.8 #2584.7 #209.8  # 414.2   
             # Luminosity distance in Mpc for z=0.09, calculated using Ned Wright's Cosmology Calculator (http://www.astro.ucla.edu/~wright/CosmoCalc.html)
             # with H_0 = 69.6, Omega_M = 0.286, Omega_vac = 0.714 
             # dl is Angular diameter distance times (1+z)**2, taking into account factor of (1+z) for time dialation, frequency change, 
             # and 2 for beaming, because source is moving away. - Brian, 4/14/2017



    data_in = open(savefile, 'rb')
    data = pickle.load(data_in)
    data_in.close()


    total_power_arr=data['total_power_arr']      # [time,freq,phi] - calcualted emission power in source frame, units of erg/s/radian/Hz  (per Hz of angular frequency?)
    freq_arr=data['freq_arr']             # [time,freq,phi] - angular frequency (omega) in source frame
    t_arr=data['t_arr']                          # [time] - emission time in source frame

    glow_arr=data['glow_arr']

    #print(total_power_arr[0,:,0])
    #print(freq_arr[0,:,0])
    #print(t_arr)

    where_are_NaNs = np.isnan(total_power_arr)
    total_power_arr[where_are_NaNs] = 1e-99

    total_power_arr[np.where(total_power_arr <= 0)] = 1e-99


    freq=np.copy(freq_arr[0,:,0])/(1+z)/2/np.pi  # freqency (nu) in observer frame
    n_freq=len(freq)                             # number of frequencies observed

    #time=np.copy(t_arr)*(1+z)                    # emission time in observer frame

    #time = 10.**(np.arange(101)/100*8.)

    time_origional = np.copy(t_arr)*(1+z_origional)  # emission time in origional file observer fame 

    if (time=='0'):
        time=time_origional


    phi_slices=len(total_power_arr[0,0,:])       # number of slices in phi direction

    phi=(np.arange(phi_slices/2)/(phi_slices/2))**2 * 90.  # the **2 is needed for the current version, must match how phi is distributed when making the file - Brian, 4/14/2017
    phi=np.concatenate((phi,[90],180-phi[::-1]))               

    #    phi_gap=(np.floor(phi[1:]*(phi_bins/360))-np.floor(phi[0:len(phi)-1]*(phi_bins/360))) * (360/phi_bins) * 2.   # d_phi between all angles, multiply by two becuase we are only ding half a circle when we make data. - Brian 4/14/2017

    phi_gap=(phi[1:]-phi[0:len(phi)-1]) * 2.   # d_phi between all angles, multiply by two becuase we are only ding half a circle when we make data. - Brian 4/14/2017

    print(phi_gap)
    
    phi_gap_arr=np.zeros([len(time_origional),len(phi_gap)])
    for i in range(len(time_origional)):
        phi_gap_arr[i,:]=np.copy(phi_gap)       # filled array of [freq,phi] containting all phi_gap


    #where_are_NaNs = np.isnan(total_power_arr)
    #total_power_arr[where_are_NaNs] = 0

    flux=np.zeros([len(time_origional),n_freq])
    for i in range(n_freq):
        flux[:,i] = np.sum(total_power_arr[:,i,:]*phi_gap_arr*np.pi/180,axis=1) * 2*np.pi / (4*np.pi*(dl*Mpc)**2)    
        # I think the 2*np.pi is becuase total_power_arr should be per Hz of angular freqnecy (omega), but we really want per Hz of freqency (nu) but I an not completely sure.  - Brian, 4/14/2017
        # One factor of 2*pi get picked up because total_power_arr is per radian, but a circle cover 2*pi radians.  This part comes from summing over phi_gap_arr. - Brian, 4/14/2014


    #print(flux[0,:])

# Now have flux in units of erg/s/cm^2/Hz


# for radio data:

    jy=np.copy(flux) * 1.e23   # in units of Janksys (Jy).  (Radio) flux is often given in Jy or mJy


    # try 2d interpolation instead, allow for different time basis. - Brian, 7/26/2017

    radio=np.zeros((len(time),len(radio_f)))

#    print(np.log(freq))
#    print(np.log(time_origional))
    #print(np.log(jy[4,:]))


    interp_radio = interpolate.interp2d(np.log(freq),np.log(time_origional),np.log(jy[:,:]), bounds_error = False, kind='linear')
    for t in range(len(time)):
        radio[t,:]=np.exp(interp_radio(np.log(radio_f),np.log(time[t])))   # flux density in Jy

#    print(freq)
#    print(time_origional)
#    print(jy[0:1,:])
#    print(time[0])
#    print(radio[0,:])

# for X-ray data:

    # Want to integrate over some range in energy

    l1=xray_f[0]/.3 * xray_min #.5  # Assumes xfreq[0] is 0.3 keV.  So lower limit is 0.5 keV
    l2=xray_f[0]/.3 * xray_max #2.  # Assumes xfreq[0] is 0.3 keV.  So upper limit is 2. keV
    xfreqi=10**((np.arange(101)/100)*(np.log10(l2)-np.log10(l1))+np.log10(l1))   # 101 points from .5 to 2 keV

    xray=np.zeros([len(time)])

    xray_interf = interpolate.interp2d(np.log(freq), np.log(time_origional), np.log(flux[:,:]), bounds_error = False)
    for t in range(len(time)):
        xray[t]=np.trapz(np.exp(xray_interf(np.log(xfreqi),np.log(time[t]))),x=xfreqi)   # Flux in erg/cm^2/s


# For optical:

    # Frequencies at which optical data was calculated

    optical=np.zeros([len(time)])
    optical_ab=np.zeros([len(time)])
    interp_jy = interpolate.interp2d(np.log(freq),np.log(time_origional),np.log(jy[:,:]), bounds_error = False)
    for t in range(len(time)):
        optical[t]=8.67-2.5*np.log10( np.trapz(np.exp(interp_jy(np.log(optical_f),np.log(time[t])))/(max(optical_f)-min(optical_f)),x=optical_f ))        # magnitude (r-band specific
        optical_ab[t]=-2.5*np.log10( np.trapz(np.exp(interp_jy(np.log(optical_f),np.log(time[t]))),x=optical_f/(3631*(max(optical_f)-min(optical_f))) ))  # AB magnitude


    print('DONE WITH INTERPOLATION!!!!')                      

    return radio, xray, optical, optical_ab, time, freq, jy, time_origional, glow_arr
