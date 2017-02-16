'''This is a quick simulation tool that Jane Rigby wrote in 2015,
to simulate spectra of high-z QSOs w JWST.  It's kludgy in places,
and may in parts be superceded by the JWST Exposure Time Calculator (ETC),
but it is a place to get started.
jane.rigby@nasa.gov, Feb 2017 '''

from astropy.io import fits
from astropy.convolution import convolve
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from astropy.io import ascii

tbl = ascii.read("vandenberk-2001.table1", comment="#")
tbl.colnames    #print tbl['wave']  # should be wave, flam, sig

#rebin to WFIRST grism resolution (R=100 compared to SDSS template at R=1800)
rebin_factor = 18    # Uses integer rebin factors. Clunky 
rms = lambda V, axis=None: np.sqrt(np.sum(np.square(V), axis))  # add in quadrature
     # above adapted from rms on stackoverflowm
binby = (rebin_factor,)  # stupid syntax to make a 1-element tuple
rb_wave  = block_reduce(tbl['wave'] , block_size=binby, func=np.mean) 
rb_flam  = block_reduce(tbl['flam'] , block_size=binby, func=np.sum)
rb_sig   = block_reduce(tbl['sig']  , block_size=binby, func=rms)

# Scale the flux. Want H_AB=26 =1.7E-20 erg/s/cm^2/A.     z=8, that's 1800 rest-frame 
zz = 8.0
flam_scale = 1.7E-20     #z=8 QSO, AB=26 at 1.6um.  --> flam=1.7E-20 erg/s/cm^2/A.
wrange = (1.75, 1.8)     #  wavelength range (micron) where we want to scale flambda

n_wave = rb_wave * (1+zz) / 1E4  # redshift, change to micron
Hband = np.where((n_wave > wrange[0]) & (n_wave < wrange[1]))
scale_factor =  flam_scale /   np.mean(rb_flam[Hband]) # units are erg/s/cm^2/A
n_flam = scale_factor * rb_flam   # Scale the flambda to match flam_scale
n_sig  = scale_factor * rb_sig    # ditto for 1sigma uncertainty

noise = np.random.standard_normal(rb_flam.shape) * 4.3E-22 # erg/s/cm^2/A, gaussian.
# this is NIRSpec/JWST noise level at 3um, 1.3E-8 Jy, R=100, from jrigby's sens plots on the web.
#  	  = 1.3E-8 1E-23 erg/s/cm^2/Hz * 3.E14 um/s / 3um  / 30000A = 4.3E-22 erg/s/cm^2/A

plt.plot(n_wave, n_flam + noise)
plt.plot(n_wave, n_flam)
plt.xlim(0.6,5.0)
plt.xlabel('wavelength (micron)')
plt.ylabel('f_lambda (erg/s/cm^2)')
plt.show()
