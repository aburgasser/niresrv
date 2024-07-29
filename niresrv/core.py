"""

	NIRES-RVFIT

	Package Description
	-------------------

	NIRES-RVFIT performs and MCMC fit of NIRES (or equivalent) spectral data of cool stars and brown dwarfs
	to extract radial velocities, rotational velocities, and other model parameters. This code is a stripped
	down version of the SMART package by Hsu, Theissen, & Burgasser

	Pre-set models
	--------------
	NIRES-RVFIT comes with the following models pre-loaded in the models/ folder:

	* TBD


"""

# CODE UPDATES TO DO:
#	* change parameter array to a dictionary with predefined quantities
#	* add in additional models
#	* compute barycentric correction given precise day and time (not just date)
#	* allow more flexibility in input model parameters

# standard packages
import copy
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import scipy
import scipy.stats as stats
from scipy.optimize import minimize,curve_fit
from scipy.interpolate import make_interp_spline,interp1d
from scipy.interpolate import griddata
#from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm

# SPLAT packages
import splat
import splat.model as spmdl
import splat.plot as splot
import splat.empirical as spem
import splat.evolve as spev
import splat.simulate as spsim
import splat.database as spdb

# astropy packages
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, CartesianRepresentation, CartesianDifferential, Galactic, Galactocentric

# other packages
import corner

#######################################################
#######################################################
#################   INITIALIZATION  ###################
#######################################################
#######################################################

# code parameters
VERSION = '28 July 2024'
GITHUB_URL = 'http://www.github.com/aburgasser/nires-rv/'
ERROR_CHECKING = True
CODE_PATH = os.path.dirname(os.path.abspath(__file__))+'/../'
MODEL_FOLDER = os.path.join(CODE_PATH,'models/')

# keck coordinate
KECK = EarthLocation.from_geodetic(lat=19.8283*u.deg, lon=-155.4783*u.deg, height=4160*u.m)

# MODEL_FILE_PREFIX = 'models_'
# WAVE_FILE_PREFIX = 'wave_'

# defaults
DEFAULT_FLUX_UNIT = u.erg/u.s/u.cm/u.cm/u.micron
DEFAULT_WAVE_UNIT = u.micron

# baseline wavelength grid
# DEFAULT_WAVE_RANGE = [0.9,2.45]
# DEFAULT_RESOULTION = 300

# model parameters
FIT_PARAMETERS = {
	'teff': {'name':'effective temperature', 'unit': u.K, 'label': r'T$_{eff}$ (K)', 'default_range': [300,3000], 'default_step': 100},
	'logg': {'name':'surface gravity', 'unit': u.dex, 'label': r'$\log{g}$ (cm/s$^2$)', 'default_range': [3.0,6.0], 'default_step': 0.25},
	'z': {'name':'metallicity', 'unit': u.dex, 'label': '[M/H]', 'default_range': [-3,1], 'default_step': 0.25},
	'alpha': {'name':'telluric scaling', 'unit': '', 'label': r'$\alpha$', 'default_range': [0,2], 'default_step': 0.1},
	'rv': {'name':'radial velocity', 'unit': u.km/u.s, 'label': 'RV (km/s)', 'default_range': [-200,200], 'default_step': 2},
	'vsini': {'name':'rotational velocity', 'unit': u.km/u.s, 'label': r'v$\sin{i}$ (km/s)', 'default_range': [1,200], 'default_step': 2},
	'vbroad': {'name':'instrumental broadening', 'unit': u.km/u.s, 'label': r'v$_{broad}$ (km/s)', 'default_range': [1,200], 'default_step': 2},
	'woff': {'name':'additive wavelength offset', 'unit': u.Angstrom, 'label': r'$\delta_\lambda$ (Ang)', 'default_range': [-10.,10.], 'default_step': 1.e-2},
	'foff': {'name':'multiplicative flux offset', 'unit': '', 'label': r'$\epsilon_{F}$', 'default_range': [-0.1,0.1], 'default_step': 1.e-3},
	'chi': {'name':'chi square', 'unit': '', 'label': r'$\chi^2$', 'default_range': [0,1.e9], 'default_step': -1},
}

# default parameters
# parameters p = [RV (km/s), vsini (km/s), alpha, instrument broadening (km/s), wavelength offset (microns), flux offset (normalized)]
DEFAULT_PARAMETERS = [30., 30., 0.5, 50., 0., 0.]
DEFAULT_PARAMETERS_LOW_LIMITS = [-300, 1, 0, 1, -50., -0.1]
DEFAULT_PARAMETERS_HIGH_LIMITS = [300, 200, 2, 200., 50., 0.1]
DEFAULT_PARAMETERS_STEPS = [2., 2., 0.03, 2., 1.e-2, 1.e-3]
DEFAULT_PARAMETERS_LABELS = ['RV (km/s)',r'v$\sin{i}$ (km/s)',r'$\alpha$',r'v$_{broad}$ (km/s)',r'$\delta_\lambda$ (Ang)',r'$\epsilon_{F}$']
DEFAULT_MODEL = 'sonora21'

# default fitting bands
FIT_BANDS = {
	'K': {'index': 0, 'fitrng':[1.95,2.38], 'description': 'full K-band'},
	'K2': {'index': 0, 'fitrng':[2.03,2.23], 'description': 'K-band H2O/CH4'},
	'K3': {'index': 0, 'fitrng':[2.26,2.38], 'description': 'K-band CO'},
	'H': {'index': 1, 'fitrng':[1.52,1.75], 'description': 'full H-band'},
	'H2': {'index': 1, 'fitrng':[1.52,1.59], 'description': 'H-band H2O'},
	'H3': {'index': 1, 'fitrng':[1.61,1.7], 'description': 'H-band CH4'},
	'J': {'index': 2, 'fitrng':[1.15,1.33], 'description': 'full J-band'},
	'J2': {'index': 2, 'fitrng':[1.145,1.185], 'description': 'J-band K I'},
	'J3': {'index': 2, 'fitrng':[1.145,1.23], 'description': 'J-band K I + FeH'},
	'J4': {'index': 2, 'fitrng':[1.15,1.28], 'description': 'J-band 2 K I + FeH + H2O'},
	'J5': {'index': 2, 'fitrng':[1.235,1.28], 'description': 'J-band K I + H2O'},
	'J6': {'index': 2, 'fitrng':[1.26,1.34], 'description': 'J-band red side'},
	'Y': {'index': 3, 'fitrng':[1.10,1.19], 'description': 'full Y-band'},
	'Y1': {'index': 3, 'fitrng':[1.11,1.16], 'description': 'Y-band K I'},
}


# welcome message on load in
print('\n\nWelcome to the NIRESRV spectral fitting code!')
print('This code is designed to fit NIRES and equivalent spectral data of T < 3000 K source to extract RVs and vsinis')
print('You are currently using the BETA version {}; please report errors to aburgasser@ucsd.edu\n'.format(VERSION))
if ERROR_CHECKING==True: print('NOTE: Currently running in error checking mode')



#######################################################
###########   COORDINATE/SETUP FUNCTIONS  #############
#######################################################

# baryRV 
def baryRV(coord,date,location=KECK,verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Computes the barycenteric correction relative to a given location, the star's coordinates, 
	and the observation date

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> import niresrv
	>>> from astropy.coordinates import SkyCoord, EarthLocation
	>>> import astropy.units as u
	>>> from astropy.time import Time
	>>> coord = SkyCoord(214.910283,52.860077,unit=u.degree)
	>>> keck = EarthLocation.from_geodetic(lat=19.8283*u.deg, lon=-155.4783*u.deg, height=4160*u.m)
	>>> date = Time('2024-05-10T10:15:32')
	>>> niresrv.baryRV(coord,date,location=keck)

	−11.106963 km/s
	
	Dependencies
	------------
	astropy.coordinates
	astropy.time
	splat.utilities

	'''
# check that we've been passed a coordiante; if not, try to convert
	if not isinstance(coord,SkyCoord): 
		try: coord0 = splat.properCoordinates(coord)
		except: raise ValueError('Cannot determine coordinates from {}'.format(coord))
	else: coord0 = copy.deepcopy(coord)

# check that we've been passed a time; if not, try to convert
	if not isinstance(date,Time): 
		try: date0 = Time(date)
		except: raise ValueError('Cannot convert {} into a time'.format(date))
	else: date0 = copy.deepcopy(date)

# return the correction in km/s
	return (coord0.radial_velocity_correction(obstime=date0, location=location)).to(u.km/u.s)
		


#######################################################
##################   DATA READING  ####################
#######################################################

def read_nires(file,output='1dspec',name='',funit=DEFAULT_FLUX_UNIT,wunit=DEFAULT_WAVE_UNIT,verbose=ERROR_CHECKING,**kwargs):
	'''
	Purpose
	-------

	Reads in NIRES data file produced by SpeXtool and returns either a single stitched Spectrum object
	or an array of 4 Spectrum objects for each order

	Parameters
	----------

	file : str
		A string that corresponds to the relevant key

	output = '1dspec' : str
		Spectral format to return, either:
		* '1dspec': orders are stitched into a single spectrum, with relative order corrections in overlap regions
		* 'multispec': four separate spectra are returned

	name = '' : str
		Optional name to assign to Spectrum object

	funit = DEFAULT_FLUX_UNIT : astropy.unit
		Default unit for flux array

	wunit = DEFAULT_WAVE_UNIT : astropy.unit
		Default unit for wave array

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	
	Returns either a single Spectrum object for stitched spectrum (output='1dspec') or an array 
	of four Spectrum objects, one for each order (output='multispec') 

	Example
	-------

	>>> import niresrv
	>>> sp = niresrv.read_nires('/Users/adam/spectra/nires/nires_J0415-0935_180901.fits',output='1dspec',name='J0415-0935')
	>>> sp.info()

	1dspec mode: returning stitched spectrum
 	spectrum of J0415-0935

	>>> sp = niresrv.read_nires('/Users/adam/spectra/nires/nires_J0415-0935_180901.fits',output='multispec',name='J0415-0935')
	>>> sp

	[NIRES spectrum of J0415-0935 order 3,
	 NIRES spectrum of J0415-0935 order 4,
	 NIRES spectrum of J0415-0935 order 5,
	 NIRES spectrum of J0415-0935 order 6]
	
	Dependencies
	------------
		
	astropy.io.fits
	splat.core.Spectrum
	splat.core.stitch

	'''
# read fits file
	with fits.open(file, **kwargs) as hdulist:
		header = hdulist[0].header
		meta = {'header': header}
		data = hdulist[0].data

# package into 4 Spectrum objects
	spec = []
	orders = header['ORDERS'].split(',')
	for i in range(len(data[:,0,0])):
		sp = splat.Spectrum(wave=data[i,0,:]*wunit,flux=data[i,1,:]*funit,noise=data[i,2,:]*funit,header=header,instrument='NIRES',name='{} order {}'.format(name,orders[i]))
		spec.append(sp)

# multispec - return array		
	if output=='multispec': 
		if verbose==True: print('multispec mode: returning array of 4 spectrum objects')
		return spec

# 1dspec - return stitched spectrum
	elif output=='1dspec':
		if verbose==True: print('1dspec mode: returning stitched spectrum')
		spc = spec[0]
		for s in spec[1:]: spc = splat.stitch(spc,s,scale=False,header=header)
		spc.name=name
		spc.header = header
		return spc

# otherwise just return data structure		
	else: 
		if verbose==True: print('no output mode selected: returning fits data structure')
		return data


#######################################################
##############   SPECTRAL MANIPULATION  ###############
#######################################################

# downsample
def downsample(sp,wave,verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Downsamples a Spectrum object to a given wavelength grid

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	copy
	numpy
	splat.core
	splat.utilties

	'''		
# check the units of input wave array
	if splat.isUnit(wave): 
		wave = wave.to(sp.wave.unit)
		wv = wave.value
	else: wv = copy.deepcopy(wave)

# trim spectrum to one wave unit ouside input wave array
	wshift = numpy.nanmedian(numpy.roll(wv,-1)-wv)
	sp.trim([wv[0]-wshift,wv[-1]+wshift])

# fill in flux and uncertainty arrays
	flx = [numpy.nan]*len(wave)
	unc = [numpy.nan]*len(wave)
	for i,w in enumerate(wv):
# process end cases
		if i==0: wrng = [w-(wv[1]-w),wv[1]]
		elif i==len(wave)-1: wrng = [wv[i-1],w+(w-wv[i-1])]
		else: wrng = [wv[i-1],wv[i+1]]
		w = numpy.where(numpy.logical_and(sp.wave.value>=wrng[0],sp.wave.value<=wrng[1]))
		cnt = len(sp.flux[w])
# compute median if at least one value is in range	   
		if cnt >= 1:
			flx[i] = numpy.nanmedian(sp.flux.value[w])
			unc[i] = numpy.nanmedian(sp.noise.value[w])/((0.5*(cnt-1))**0.5)
# otherwise expand range	   
		else:
			w = numpy.where(numpy.logical_and(sp.wave.value>=wrng[0]-wshift,sp.wave.value<=wrng[1]+wshift))
			cnt = len(sp.flux[w])
			if cnt >= 1:
				flx[i] = numpy.nanmedian(sp.flux.value[w])
				unc[i] = numpy.nanmedian(sp.noise.value[w])/((0.5*(cnt-1))**0.5)
# returnn new Spectrum object
	return splat.Spectrum(wave=numpy.array(wv)*sp.wave.unit,flux=flx*sp.flux.unit,noise=unc*sp.flux.unit,name=sp.name)


# modelapp
def modelapp(mdl,tell,p,wave):
	'''
	Purpose
	-------
	Applies model parameters to combined atmosphere and telluric models

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	copy
	splat.core.Spectrum

	'''	
	cmdl = copy.deepcopy(mdl)
	ctell = copy.deepcopy(tell)
# RV shift model	
	cmdl.rvShift(p[0]) # rv shift model
# apply vsini to model	
	cmdl.rotate(p[1]) # rotate model
# scale telluric absorption and interpolate to wavelength range
	ctell.flux = [numpy.max([c,0])**p[2] for c in ctell.flux.value]*ctell.flux.unit
	ctell.toWavelengths(cmdl.wave)
# apply telluric absorption with flux offset
	cmdl.flux = (cmdl.flux.value*ctell.flux.value)*cmdl.flux.unit
# apply multiplicative flux offset
	cmdl.flux = (cmdl.flux.value+p[5]*numpy.nanmedian(cmdl.flux.value))*cmdl.flux.unit
# apply instrument broadening
# NOTE: NEED TO CHANGE THIS TO A GAUSSIAN BROADEN
	cmdl.rotate(p[3])
# apply additive wavelength shift - note default assumption that we're shifting in angstroms
	cmdl.wave = (cmdl.wave.value+p[4]/1000.)*cmdl.wave.unit
# map to input wavelength scale
	cmdl.toWavelengths(wave)
	return cmdl

# corrmdl 
def corrmdl(data,mdl,smooth=30,order=5,fxn='poly',verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Computes the continuum correction between model and spectrum

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''		
# duplocate data and mdl so we don't change the originals
	cd = copy.deepcopy(data)
	cd.smooth(smooth)
	cm = copy.deepcopy(mdl)
	cm.smooth(smooth)

# ratio that we'll be fitting   
	ratio = cd.flux.value/cm.flux.value

# fitting options
# NOTE: spline fit is TOO good - does not preserve features of model; we need lower resolution
	if fxn=='spline':
		spl = make_interp_spline(data.wave.value[numpy.isfinite(ratio)==True],ratio[numpy.isfinite(ratio)==True],k=3)
		return mdl.flux.value*spl(data.wave.value)
	elif fxn=='poly':
		fit = numpy.polyfit(data.wave.value[numpy.isfinite(ratio)==True],ratio[numpy.isfinite(ratio)==True],order)
		return mdl.flux.value*numpy.polyval(fit,mdl.wave.value)
	elif fxn=='constant':
		return mdl.flux.value*numpy.nanmedian(ratio)
	else: raise ValueError('Do not recognize function {}; options are poly, spline, or constant'.format(fxn))


#######################################################
################   SPECTRAL FITTING  ##################
#######################################################

# mask_mask
def make_mask(data,mdl,mask=[],std=5,nloop=10,verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Generates a fitting mask by comparing between data and reference model and identifying outliers.
	Mask values of 1 are "good", mask vales of 0 are "bad"

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''
# set up mask			
	if len(mask)!=len(data.wave): msk = numpy.ones(len(data.wave))
	else: msk=numpy.array(mask)

# conduct several nloop loops of identifying and masking outliers
	for i in range(nloop):
		mflx = corrmdl(data,mdl)
		diff = numpy.abs(msk*(data.flux.value-mflx))
		msk[numpy.where(diff>=std*numpy.nanstd(diff))]=0
		if numpy.nansum(msk)==0: msk = numpy.ones(len(data.wave))
	if verbose==True: print('Masked out {:.0f} spectral points'.format(len(msk)-numpy.nansum(msk)))
	return msk


# chi2 
def chi2(data,mdl,mask=[],verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Computes chi2 value between data and model with an optional mask

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''
# set up mask		
	if len(mask)<len(data.wave): msk = numpy.ones(len(data.wave))
	else: msk=numpy.array(mask)

# first correct continuum  
	mflx = corrmdl(data,mdl)

# return straight chi2  
	return numpy.nansum(msk*((data.flux.value-mflx)**2)/(data.noise.value**2))


# prep
def prep(data,mpar,vbary=0.,fitrng=[],mset=DEFAULT_MODEL,adjustnoise=0,snrlimit=3,verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Preps the baseline fitting elements (data, atmosphere model, telluric model) for fitting

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	copy
	numpy
	splat.core
	splat.model

	'''		
# set fitrng to data width as needed	
	if len(fitrng)==0: fitrng=[numpy.nanmin(data.wave.value),numpy.nanmax(data.wave.value)]

# prep data: trim and normalized
	spc = copy.deepcopy(data)
	spc.trim(fitrng)
	spcsm = copy.deepcopy(spc)
	spcsm.smooth(20)
	scl = numpy.nanquantile(spcsm.flux.value,0.95)
	spc.scale(1./scl)

# ansatz for noise scaling if desired
	snr = numpy.nanmedian(spc.flux.value/spc.noise.value)
	if adjustnoise>0 and snr>snrlimit: 
		spc.noise = (spc.noise.value*numpy.nanmax([snrlimit/snr,adjustnoise]))*spc.flux.unit
		if verbose==True: print('Adjusted noise by a factor of {}'.format(adjustnoise))

# prep model: read in, trim, normalize, and shift to barycentric velocity
	if 'model' not in list(mpar.keys()): mpar['model'] = mset
	if 'instrument' not in list(mpar.keys()): mpar['instrument'] = 'RAW'
	mdl = spmdl.loadModel(**mpar)
	mdl.trim([fitrng[0]-0.05,fitrng[1]+0.05])
	mdl.normalize()
	mdl.rvShift(vbary)

# read in telluric model
	tell = spmdl.loadTelluric([fitrng[0]-0.1,fitrng[1]+0.1],output='spectrum')
	return spc,mdl,tell


# initia_model
def initial_model(spc,tell,mset=DEFAULT_MODEL,mask=[],fitrng=[],par_range={},max_models=-1,save_results=False,prefix='',verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Conducts an initial fit over a model grid to identify best model for detailed comparison

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''		
# process mask
	if len(mask)<len(spc.wave): msk = numpy.ones(len(spc.wave))
	else: msk = numpy.array(mask)

# initialize parameters
	if len(fitrng)==0: fitrng = [numpy.nanmin(spc.wave.value),numpy.nanmax(spc.wave.value)]
	vbroad = numpy.nanmedian(spc.wave.value-numpy.roll(spc.wave.value,3))*3.e5/numpy.nanmedian(spc.wave.value)
# CODE FIX: change to dictionary
	p0 = [0,50.,0.7,vbroad,0.,0.]

# read in and constrain model set parameters
# CODE FIX: this only works for continuous variables, need to add discrete
	mpars = pandas.DataFrame(spmdl.loadModelParameters(mset,instrument='RAW')['parameter_sets'])
	if mpars['instrument'].iloc[0]!='RAW':
		raise ValueError('Full resolution version of model not available for {}'.format(mset))
	for l in list(par_range.keys()):
		if l in list(mpars.columns):
			mpars = mpars[mpars[l]>=par_range[l][0]]
			mpars = mpars[mpars[l]<=par_range[l][1]]
# some special cases (other constraints?)
	if 'co' in list(mpars.columns): mpars=mpars[mpars['co']==1.0]

# quit if too many
	if max_models>0 and len(mpars)>max_models:
		raise ValueError('Number of available models ({:.0f}) exceeds limit of {:.0f}; try constraining parameters'.format(len(mpars)),max_models)
	if verbose==True: print('Evaluating {:.0f} models'.format(len(mpars)))

# loop to find best fit (lowest chi2) model
# CODE FIX: also need to work in some initial constraints on RV, alpha - between model fits?
	mpars['chi'] = [numpy.nan]*len(mpars)
	for i in range(len(mpars)):
# read in model, trim & normalize				
		par = dict(mpars.iloc[i])
		mdl0 = spmdl.loadModel(**par)
		mdl0.trim([fitrng[0]-0.05,fitrng[1]+0.05])
		mdl0.normalize()
# apply model parameters
		cmdl = modelapp(mdl0,tell,p0,spc.wave)
# compute chi square
		mpars['chi'].iloc[i] = chi2(spc,cmdl,mask=msk)
		if verbose==True: print(dict(mpars.iloc[i]))

# save results if desired
	par = dict(mpars.iloc[numpy.argmin(mpars['chi'])])
	if save_results==True: 
		mdl0 = spmdl.getModel(**par)
		mdl0.trim([fitrng[0]-0.05,fitrng[1]+0.05])
		mdl0.normalize()
		plot_comparison(spc,mdl0,tell,p0,mask=msk,save_results=save_results,prefix=prefix,verbose=verbose)

# return model parameters
	return(par)


# amoeba_objective
def amoeba_objective(p,spc,mdl,tell,msk,verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Function used by `amoebafit` to determine the quality of fit of a model

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''		
	cmdl = modelapp(mdl,tell,p,spc.wave.value)
	return chi2(spc,cmdl,mask=msk)

# amoebafit
def amoebafit(spc,mdl,tell,mask=[],p0=DEFAULT_PARAMETERS,plow=DEFAULT_PARAMETERS_LOW_LIMITS,
	phigh=DEFAULT_PARAMETERS_HIGH_LIMITS,save_results=False,prefix='amoeba_',verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Conducts an optimization fit of parameters usingthe Nelder-Mean (amoeba) algorithm

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''		
# process mask
	if len(mask)<len(spc.wave): msk = numpy.ones(len(spc.wave))
	else: msk = numpy.array(mask)
	
# fit bounds
	bounds = [(plow[i],phigh[i]) for i in range(len(plow))]

# conduct minimization
	res = minimize(amoeba_objective,p0,args=(spc,mdl,tell,msk),bounds=bounds,method='nelder-mead')

# save results if desired	
	if save_results==True: 
		plot_comparison(spc,mdl,tell,res.x,mask=msk,save_results=save_results,prefix=prefix,verbose=verbose)

# return fit	
	return res.x


# MCMC fit
def rvmcmc(spc,mdl,tell,mask=[],p0=DEFAULT_PARAMETERS,plow=DEFAULT_PARAMETERS_LOW_LIMITS,
	phigh=DEFAULT_PARAMETERS_HIGH_LIMITS,pstep=DEFAULT_PARAMETERS_STEPS,nstep=300,burn=0.25,iterim=50,
	dof_scale=3,rejection_scale=0.5,recover_scale=1.2,save_results=True,prefix='mcmc_',verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Conducts MCMC fit to optimize fit to RV and vsini and other parameters given baseline atmosphere
	and telluric models

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''
# process mask
	if len(mask)<len(spc.wave): msk = numpy.ones(len(spc.wave))
	else: msk = numpy.array(mask)

# degrees of freedom
	dof = int(numpy.nansum(mask)/dof_scale-len(p0))

# initial model fit for baseline
	cmdl = modelapp(mdl,tell,p0,spc.wave.value)
	chis = [chi2(spc,cmdl,mask=msk)]
	pvals = [p0]

# run this MCMC loop
	for i in tqdm(range(nstep)):
# update values using step size, constraining to bounds		
		pnew = numpy.random.normal(pvals[-1],pstep)
		for k in range(len(p0)):
			pnew[k] = numpy.nanmax([pnew[k],plow[k]])
			pnew[k] = numpy.nanmin([pnew[k],phigh[k]])
# generate new model and compare		
		cmdl = modelapp(mdl,tell,pnew,spc.wave)
		chinew = chi2(spc,cmdl,mask=msk)
# recovery - if we've wandered far away return to best fit 	
		if chinew/numpy.nanmin(chis) > recover_scale:
			pv = numpy.array(pvals)
			pvals.append(pv[numpy.argmin(chis)])
			chis.append(numpy.nanmin(chis))
# MCMC switch - using survival function with input rejection scale 	
		elif 2*scipy.stats.f.sf(chinew/chis[-1],dof,dof)>numpy.random.uniform(rejection_scale,1):
			pvals.append(pnew)
			chis.append(chinew)
		else:
			pvals.append(pvals[-1])
			chis.append(chis[-1])
# iterim save
		if i !=0 and iterim>0 and numpy.mod(i,iterim)==0:
			pv = numpy.array(pvals)
			pbest = pv[numpy.argmin(chis)] 
#			if verbose==True: print('Best fit parameters at step {:.0f}: {}'.format(i,pbest))
# CODE FIX: update with defined labels
			if save_results==True: 
				dpfit = pandas.DataFrame({
					'RV (km/s)': [p[0] for p in pv],
					'vsini (km/s)': [p[1] for p in pv],
					'alpha': [p[2] for p in pv],
					'broaden (km/s)': [p[3] for p in pv],
					'shift (Ang)': [p[4] for p in pv],
					'flux offset': [p[5] for p in pv],
					'chi': chis,
				})
				dpfit['dof'] = [dof]*len(dpfit)
# save parameters, chains, comparison, & cornerplot
				dpfit.to_excel('{}parameters.xlsx'.format(prefix),index=False)
				plot_mcmcchains(dpfit,pbest,save_results=save_results,prefix=prefix,verbose=verbose)
				plot_comparison(spc,mdl,tell,pbest,mask=msk,save_results=save_results,prefix=prefix,verbose=verbose)
				plot_corner(dpfit,pbest,dof,save_results=save_results,prefix=prefix,verbose=verbose)

# report results
	pvals = numpy.array(pvals)
	chis = numpy.array(chis)
# best fit
	pbest = pvals[numpy.argmin(chis)] 
	if verbose==True: print('Best fit parameters at step {:.0f}: {}'.format(i,pbest))
# parameters after removing initial burn
# CODE FIX: update with defined labels
	dpfit = pandas.DataFrame({
		'RV (km/s)': [p[0] for p in pvals[int(burn*nstep):]],
		'vsini (km/s)': [p[1] for p in pvals[int(burn*nstep):]],
		'alpha': [p[2] for p in pvals[int(burn*nstep):]],
		'broaden (km/s)': [p[3] for p in pvals[int(burn*nstep):]],
		'shift (Ang)': [p[4] for p in pvals[int(burn*nstep):]],
		'flux offset': [p[5] for p in pvals[int(burn*nstep):]],
		'chi': chis[int(burn*nstep):],
	})
	dpfit['dof'] = [dof]*len(dpfit)
	try:
		dpfit['teff'] = [mdl.teff]*len(dpfit)
		dpfit['logg'] = [mdl.logg]*len(dpfit)
	except: pass

# save parameters, chains, comparison, & cornerplot
	if save_results==True: 
		dpfit.to_excel('{}parameters.xlsx'.format(prefix),index=False)
		plot_mcmcchains(dpfit,pbest,save_results=save_results,prefix=prefix,verbose=verbose)
		plot_comparison(spc,mdl,tell,pbest,mask=msk,save_results=save_results,prefix=prefix,verbose=verbose)
		plot_corner(dpfit,pbest,dof,save_results=save_results,prefix=prefix,verbose=verbose)

# return fit array
	return dpfit



#######################################################
###############   PLOTTING FUNCTIONS  #################
#######################################################

# plot_mcmcchains
def plot_mcmcchains(dpfit,pbest,save_results=False,prefix='',verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Plots the MCMC chains

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''			
# set up figure	
# CODE FIX: plot scale is hard-wired; update
	plt.clf()
	fig = plt.figure(figsize=[12,8])
	for i,l in enumerate(list(dpfit.columns)):	
		ax = plt.subplot(4,3,i+1)
		ax.plot(dpfit[l],'k-')
		if i < len(pbest): ax.plot(numpy.zeros(len(dpfit[l]))+pbest[i],'m--')
		ax.set_ylabel(l)
	plt.tight_layout()
	if save_results==True: fig.savefig('{}chains.pdf'.format(prefix))
	if verbose==True: plt.show()
	return


# plot_comparison
def plot_comparison(spc,mdl,tell,pbest,mask=[],save_results=False,prefix='',verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Plots comparison of source spectrum to model spectrum

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''		
# process mask and mask out bad pixels
	if len(mask)<len(spc.wave): msk = numpy.ones(len(spc.wave))
	else: msk = numpy.array(mask)
	flx = spc.flux.value
	flx[msk!=1]=numpy.nan

# set up figure
	plt.clf()
	xlim = [numpy.nanmin(spc.wave.value),numpy.nanmax(spc.wave.value)]
	fig = plt.figure(figsize=[8,5])
	plt.plot(spc.wave,spc.flux.value,'r-')
	plt.plot(spc.wave,flx,'k-')
	cmdl = modelapp(mdl,tell,pbest,spc.wave)
	chi = chi2(spc,cmdl,mask=msk)
	dof = int(numpy.nansum(msk)/3.-len(pbest))
	palt = copy.deepcopy(pbest)
	palt[2] = 0.
	calt = modelapp(mdl,tell,palt,spc.wave)
	ctmp = copy.deepcopy(mdl)
	ctmp.flux = (numpy.zeros(len(mdl.flux))+1)*mdl.flux.unit
	ctell = modelapp(ctmp,tell,pbest,spc.wave)
	plt.plot(cmdl.wave,ctell.flux.value+1,'g-',alpha=0.8)
	plt.plot(cmdl.wave,corrmdl(spc,calt)+0.75,'b-',alpha=0.8)
	plt.plot(cmdl.wave,corrmdl(spc,cmdl),'m-')
	plt.plot(cmdl.wave,flx-corrmdl(spc,cmdl),'k-')
	plt.plot(cmdl.wave,cmdl.wave.value*0.,'k--')
#	plt.fill_between(spc.wave.value,spc.noise.value**0.5,-1.*spc.noise.value**0.5,facecolor='k',alpha=0.1)
	plt.fill_between(spc.wave.value,spc.noise.value,-1.*spc.noise.value,facecolor='k',alpha=0.25)
	plt.xlabel('Wavelength ($\mu$m)',fontsize=14)
	plt.ylabel('Normalized Flux',fontsize=14)
#	plt.legend(['J0506+0738','Atmosphere Model','Full Model'],fontsize=14)
	plt.text(xlim[1],1.9,'  telluric',color='g',horizontalalignment='left',fontsize=12)
	plt.text(xlim[1],1.45,'  stellar model',color='b',horizontalalignment='left',fontsize=12)
	plt.text(xlim[1],0.8,'  data',horizontalalignment='left',fontsize=12)
	plt.text(xlim[1],0.6,'  full model',color='m', horizontalalignment='left',fontsize=12)
	plt.text(xlim[1],0.4,r'  $\chi^2_r$ = '+'{:.1f}'.format(chi/dof),color='k', horizontalalignment='left',fontsize=12)
	plt.text(xlim[1],0.1,'  uncertainty',color='k',alpha=0.2, horizontalalignment='left',fontsize=12)
	plt.text(xlim[1],-0.1,'  data-model',color='k', horizontalalignment='left',fontsize=12)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xlim(xlim)
	plt.ylim([-0.25,2.2])
	plt.tight_layout()
	if save_results==True: fig.savefig('{}comparison.pdf'.format(prefix))
	if verbose==True: plt.show()
	return

# plot_corner
def plot_corner(dpfit,pbest,dof,use_weights=True,plabels=DEFAULT_PARAMETERS_LABELS,save_results=False,prefix='',verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Makes corner plot for parameters

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''
	dpplot = copy.deepcopy(dpfit)
# chis
	if 'chi' in list(dpplot.columns): 
		chis = numpy.array(dpfit['chi'])
		dpplot.drop(['chi'],axis=1,inplace=True)
	else: chis = numpy.ones(len(dpplot))
	
# weights
	if use_weights==True: weights = 2*stats.f.sf(chis/numpy.nanmin(chis),dof,dof)
	else: weights = numpy.ones(len(chis))

# replace labels
	if len(plabels)==0: plabels = list(dpplot.columns)
	else:
		for i,x in enumerate(list(dpplot.columns)):
			if i<len(plabels): dpplot.rename(columns={x: plabels[i]},inplace=True)

# check for range limits
	for x in list(dpplot.columns):
		if numpy.nanmin(dpplot[x])==numpy.nanmax(dpplot[x]): 
			dpplot.drop([x],axis=1,inplace=True)
	if len(dpplot)<=1:
		if verbose==True: print('Warning: there are no parameters to plot!')
		return
	plabels = list(dpplot.columns)
	truths = []
	if numpy.nanmax(chis)!=numpy.nanmin(chis):
		for x in list(dpplot.columns): truths.append(dpplot[x].iloc[numpy.argmin(chis)])

# generate plot
	plt.clf()
	fig = corner.corner(dpplot,quantiles=[0.16, 0.5, 0.84], labels=plabels, show_titles=True, weights=weights, \
						title_kwargs={"fontsize": 12},smooth=1,truths=truths)
	plt.tight_layout()
	if save_results==True: fig.savefig('{}corner.pdf'.format(prefix))
	if verbose==True: plt.show()
	return


#######################################################
#############   BATCH FITTING FUNCTION  ###############
#######################################################

# fit_sequence
def fit(file,band='K',fitrng=[],mset=DEFAULT_MODEL,initial_fit=True,par_range={},amoeba=True,
	adjustnoise=True,tinit=1000,ginit=5.0,p0=DEFAULT_PARAMETERS,plow=DEFAULT_PARAMETERS_LOW_LIMITS,
	phigh=DEFAULT_PARAMETERS_HIGH_LIMITS,pstep=DEFAULT_PARAMETERS_STEPS,nstep=300,burn=0.25,iterim=50,
	model_mask=True,dof_scale=3,rejection_scale=0.5,recover_scale=1.2,save_results=True,prefix='',
	verbose=ERROR_CHECKING):
	'''
	Purpose
	-------
	Cumulative function that does data preparation, initial model fit, nelder-mead optimization, 
	and MCMC fit

	Parameters
	----------
	TBD

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	TBD

	Example
	-------
	>>> TBD

	Dependencies
	------------
	TBD

	'''		
# initial parameters
	# plabels = ['RV (km/s)','vsini (km/s)','telluric scale','Broadening (km/s)','wave shift (micron)','flux offset']
	# plow = [-300,50,0.1,1.,-0.01,-0.1] # low limits
	# phigh = [300,50,1.1,200.,0.01,0.1] # high limits
	# pstep = [2.,0.,0.03,2.,3.e-5,3.e-4] # step size - note we are setting vsini to be fixed

# read in file and process source information
	if verbose==True: print('\nReading in file {}'.format(file))
	sp = read_nires(file,output='multispec')

# determine fitting band
	if len(fitrng)==0:
		if band in list(FIT_BANDS.keys()):
			spec = sp[FIT_BANDS[band]['index']]
			fitrng = FIT_BANDS[band]['fitrng']
		else: raise ValueError('Cannot fit in {} band; try {}'.format(band,list(FIT_BANDS.keys())))
	if verbose==True: print('Fitting over range {}'.format(fitrng))

# barycentric correction
# CODE FIX: THIS WILL ONLY WORK FOR NIRES
	coordinate = SkyCoord(spec.header['RA'],spec.header['DEC'],unit=u.degree)
	spec.name = splat.designationToShortName(splat.coordinateToDesignation(coordinate))
	date = Time(spec.header['AVE_DATE']+'T'+spec.header['AVE_TIME'])
	vbary = baryRV(coordinate,date)
	if verbose==True: print('Barycentric velocity = {:.2f}'.format(vbary))

# establish broadening
	spc = copy.deepcopy(spec)
	spc.trim(fitrng)
	vbroad = numpy.nanmedian(spc.wave.value-numpy.roll(spc.wave.value,3))*3.e5/numpy.nanmedian(spc.wave.value)
	p0[3] = vbroad
	plow[3] = vbroad*0.75
	phigh[3] = vbroad*1.25
	if verbose==True: print('Baseline instrumental broadening = {:.1f} km/s'.format(vbroad))

# generate initial mask
	spcsm = copy.deepcopy(spc)
	spcsm.smooth(10)
	msk = make_mask(spc,spcsm,verbose=verbose)

	mpar = {'teff':tinit,'logg':ginit,'model':mset,'instrument':'RAW'}
	spc,mdl,tell = prep(spec,mpar,vbary=vbary,fitrng=fitrng,adjustnoise=0,mset=mset,verbose=verbose)

# conduct an initial fit to find best atmosphere model
	if initial_fit==True:
		if verbose==True: print('\nConducting initial fit over models')
		mpar = initial_model(spc,tell,mask=msk,par_range=par_range,mset=mset,save_results=save_results,prefix=prefix+'initial_',verbose=verbose)
		spc,mdl,tell = prep(spec,mpar,vbary=vbary,fitrng=fitrng,adjustnoise=0,mset=mset,verbose=verbose)
		if verbose==True: 
			print('Resulting parameters:')
			for x in list(mpar.keys()): print('\t{}: {}'.format(x,mpar[x]))	

# conduct initial amoeba fit
	if amoeba==True:
		if verbose==True: print('\nConducting Nelder-Mead optimization of fit parameters')
		pamoeba = amoebafit(spc,mdl,tell,mask=msk,p0=p0,plow=plow,phigh=phigh,save_results=save_results,prefix=prefix+'amoeba_',verbose=verbose)
		if verbose==True: 
			print('Nelder-Mead parameters:')
			for i,l in enumerate(list(DEFAULT_PARAMETERS_LABELS)): print('\t{}: {}'.format(l,pamoeba[i]))
		p0 = copy.deepcopy(pamoeba)

# update mask
	cmdl = modelapp(mdl,tell,p0,spc.wave.value)
	if model_mask == True:
		msk = make_mask(spc,cmdl,mask=msk,verbose=verbose)

# adjustnoise if desired
	if adjustnoise==True:
		chi = chi2(spc,cmdl,mask=msk)
		adjustnoise = numpy.nanmax([1,(chi/numpy.nansum(msk)/10)**0.5])
#		if verbose==True: print('Adjusted noise by a factor {}'.format(adjustnoise))
		spc,mdl,tell = prep(spec,mpar,vbary=vbary,fitrng=fitrng,adjustnoise=adjustnoise,mset=mset,verbose=verbose)

# conduct final MCMC
	if verbose==True: print('\nConducting MCMC for fit parameters and uncertainties')
	dpfit = rvmcmc(spc,mdl,tell,mask=msk,p0=p0,plow=plow,phigh=phigh,pstep=pstep,nstep=nstep,burn=0.,
		iterim=iterim,dof_scale=dof_scale,rejection_scale=rejection_scale,recover_scale=recover_scale,
		save_results=save_results,prefix=prefix+'mcmc_',verbose=verbose)

# report best results
	pbest = list(dpfit.iloc[numpy.argmin(dpfit['chi'])])
	if verbose==True: 
		print('Best fit parameters:')
		for i,l in enumerate(list(dpfit.columns)): print('\t{}: {}'.format(l,pbest[i]))

# return
# CODE FIX: PROVIDE MORE OPTIONS?
	return pbest
	
