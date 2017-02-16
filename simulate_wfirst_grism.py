# Call on command line as:  /Volumes/Apps_and_Docs/JRR_Utils/anaconda/bin/python  simulate_wfirst_grism.py 

from astropy.io import fits
from astropy.convolution import convolve
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from astropy.io import ascii

tbl = ascii.read("vandenberk-2001.table1")  # SDSS QSO composite of VandenBerk et al. 2001
tbl.colnames    #print tbl['wave']  # should be wave, flam, sig

#rebin to WFIRST grism resolution (R=600, which is 3x lower than SDSS template)
rebin_factor = 3    # Uses integer rebin factors. Clunky
rms = lambda V, axis=None: np.sqrt(np.sum(np.square(V), axis))  # add in quadrature
     # above adapted from rms on stackoverflowm
binby = (rebin_factor,)  # stupid syntax to make a 1-element tuple
rb_wave  = block_reduce(tbl['wave'] , block_size=binby, func=np.mean) 
rb_flam  = block_reduce(tbl['flam'] , block_size=binby, func=np.sum)
rb_sig   = block_reduce(tbl['sig']  , block_size=binby, func=rms)

# Scale the observed flux to match flam_scale in wavelength range wrange
zz = 4.0                 #  z=4 QSO at knee of LF.
flam_scale = 1.89E-18    #  units are erg/cm2/s/A
wrange = (1.75, 1.8)     #  wavelength range (micron) where we want to scale flambda

n_wave = rb_wave * (1+zz) / 1E4  # redshift, and convert units to micron
Hband = np.where((n_wave > wrange[0]) & (n_wave < wrange[1]))     
scale_factor =  flam_scale /   np.mean(rb_flam[Hband]) # units are erg/s/cm^2/A
n_flam =  rb_flam * scale_factor   # Scale the flambda to match flam_scale
n_sig  =  rb_sig  * scale_factor   # ditto for 1sigma uncertainty

# add gaussian noise 
noise = np.random.standard_normal(rb_flam.shape) * 4E-19 #erg/s/cm^2/A, gaussian
      # HLS Noise  = 4E-19  erg/s/cm^2/A 

HLS = 4E-19 * (np.zeros(n_wave.shape) + 1.0)  #  flambda in erg/s/cm^2/A. HLS sensitivity.

plt.plot(n_wave, noise + n_flam)  # plot the simulated noisy spectrum
plt.plot(n_wave, n_flam)          # plot the intrinsic spectrum
plt.plot(n_wave, HLS)             # plot a line at the HLS sensitivity
plt.xlim(1.35,1.95)
plt.ylim(0,0.5E-17)
plt.xlabel('wavelength (A)')
plt.ylabel('f_lambda (erg/s/cm^2)')
plt.show()
