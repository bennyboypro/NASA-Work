import argparse
from pycbc.inject.inject import _HDFInjectionSet
from pycbc.noise import gaussian
from pycbc.detector import Detector
from pycbc.types import timeseries
import pycbc.inject
import pylab
import pycbc.psd
import pycbc.io
import numpy.random
import os
import h5py

parser = argparse.ArgumentParser()

# ================= Noise parameters ======================

parser.add_argument('--dets', nargs='+',type=str, help="[REQUIRED]Detectors")
parser.add_argument('--len', type=int,            help="[REQUIRED]Length of generated signal in seconds")
parser.add_argument('--psd', type=str,            help="[OPTIONAL]Analytic PSD model", default='aLIGOZeroDetHighPower')
parser.add_argument('--flow', type=int,           help="[OPTIONAL]f_lower value (default=10)", default=10)
parser.add_argument('--srat', type=int,           help="[OPTIONAL]Sample rate (default=4096)", default=4096)

# ================= Signal Parameters =====================

parser.add_argument('--m1', type=float,   help="[REQUIRED]Mass of the first object")
parser.add_argument('--m2', type=float,   help="[REQUIRED]Mass of the second object")
parser.add_argument('--ra', type=float,   help="[REQUIRED]Right ascension of the system, from 0 to 2PI")
parser.add_argument('--dec', type=float,  help="[REQUIRED]Declination of the system, from 0 to PI (North Pole to South Pole")
parser.add_argument('--tc', type=float,   help="[REQUIRED]Coalescence time (reference=center of the earth)")
parser.add_argument('--dist', type=float, help="[REQUIRED]Distance to the system (Mpc)")
parser.add_argument('--apx', type=str,    help="[OPTIONAL]Approximant", default='SEOBNRv4_opt')
parser.add_argument('--sp1z', type=float, help="[OPTIONAL]Spin1z", default=0)
parser.add_argument('--sp2z', type=float, help="[OPTIONAL]Spin2z", default=0)
parser.add_argument('--pol', type=float,  help="[OPTIONAL]Polarization of the system", default=0)
parser.add_argument('--taper', type=str,  help="[OPTIONAL]Tapers the signal. Options: TAPER_END, startend, TAPER_STARTEND, TAPER_NONE, start, TAPER_START, end", default='start')

# ================= File management/User interface parameters ============

parser.add_argument('--fileoutput', nargs='+', type=str,  help="[REQUIRED]Names of the HDF output files")
parser.add_argument('--plotsoutput', nargs='+', type=str, help="[OPTIONAL]If given, assigns the filenames for the plot images")
parser.add_argument('--plotshow', type=bool,              help="[OPTIONAL]=True if you want to display plots (default False)", default=False)
parser.add_argument('--plotzoom', type=float,             help="[OPTIONAL]How far zoomed in the plots are", default=0.05)
parser.add_argument('--script', type=str,                 help="[OPTIONAL]Progress updates for a scripted session", default='Working...')

params = parser.parse_args()


class SingleInjection(_HDFInjectionSet):
    def __init__(self, params):
        self.table = [params]
        self.extra_args = {}
        self.table = pycbc.io.WaveformArray.from_kwargs(**params)

# Define all parameters here
p = {'approximant' : params.apx,
     'mass1' : params.m1,
     'mass2' : params.m2,
     'spin1z' : params.sp1z,
     'spin2z' : params.sp2z,
     'ra' : params.ra,
     'dec' : params.dec,
     'polarization' : params.pol,
     'tc' : params.tc,
     'distance' : params.dist,
     'taper' : params.taper}

my_injection = SingleInjection(p)

# Set noise metadata values
f_lower = params.flow
tsamples = params.srat * params.len
fsamples = tsamples / 2 + 1
df = 1.0 / params.len

i = 0
dets = params.dets
fileoutput = params.fileoutput
plotsoutput = params.plotsoutput

for det in dets:
    print(params.script)
    # Generate PSD
    if det =='V1':
        psd = pycbc.psd.from_string('AdvVirgo', fsamples, df, f_lower)
    else:
        psd = pycbc.psd.from_string(params.psd, fsamples, df, f_lower)

    # Random seed for noise generation
    seed = numpy.random.randint(1, 999)

    # Generate frequency psd
    htilde = pycbc.noise.gaussian.frequency_noise_from_psd(psd, seed=seed)

    # Convert to time psd
    hoft = htilde.to_timeseries()

    # Blank time series same size as noise
    zeros = hoft * 0

    # Insert waveform into blank
    my_injection.apply(zeros, det, params.flow)

    # Insert waveform into noise
    hoft += zeros

    # Save the noise timeseries
    cwd = os.getcwd()

    filename = cwd + '/' + fileoutput[i]

    hoft.save(filename)

    # Optionally view the plots, including qtransform
    if params.plotsoutput != False:
        whitened = hoft.whiten(4,4, low_frequency_cutoff=10)
        bpsd = whitened.highpass_fir(30, 512).lowpass_fir(250, 512)

        times, freqs, power = bpsd.qtransform(.001, logfsteps=100, qrange=(4,8), frange=(20, 512))

        zeros_zoom = zeros.time_slice(p['tc']-params.plotzoom, p['tc']+params.plotzoom)
        hoft_zoom = hoft.time_slice(p['tc']-params.plotzoom, p['tc']+params.plotzoom)
        line = p['tc']

        pylab.figure(figsize=(15,6))

        pylab.subplot(211)
        pylab.plot(hoft_zoom.sample_times, hoft_zoom, label="Signal")
        pylab.plot(zeros_zoom.sample_times, zeros_zoom, label="Waveform")
        pylab.axvline(line, color='black', linestyle='--', label="Coalescence Time")
        pylab.title('{}'.format(det))
        pylab.legend(loc=1)
        pylab.grid()

        pylab.subplot(212)
        pylab.pcolormesh(times, freqs, power**0.5)
        pylab.xlim(p['tc']-params.plotzoom, p['tc']+params.plotzoom)
        pylab.title('{} Spectrogram'.format(det))
        pylab.yscale('log')
        pylab.axvline(line, color='black', linestyle='--')

        t = ('{} mass1:{} mass2:{} DIST:{} RA:{} DEC:{}'.format(det, params.m1, params.m2, params.dist, params.ra, params.dec))
        pylab.suptitle(t)

        filename = cwd + '/' + plotsoutput[i]
        pylab.savefig(filename)
        if params.plotshow:
            pylab.show()

    i += 1