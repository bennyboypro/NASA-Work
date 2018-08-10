import argparse
from pycbc.inject.inject import _HDFInjectionSet
from pycbc.noise import gaussian
from pycbc.detector import Detector
from pycbc.types import timeseries
from pycbc.psd.analytical import AdVMidHighSensitivityP1200087, AdVLateLowSensitivityP1200087
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q, mchirp_from_mass1_mass2
import pycbc.inject
import pylab
import pycbc.psd
import pycbc.io
import numpy.random
import os
import h5py

parser = argparse.ArgumentParser()

# ================= Noise parameters ======================

parser.add_argument('--dets', nargs='+',
                    help="Detectors")
parser.add_argument('--seed', type=float,
                    help="Seed for random noise generation")
parser.add_argument('--len', type=int,
                    help="Length of generated signal in seconds")
parser.add_argument('--psd', type=str,
                    help="Analytic PSD model", default='AdVLateLowSensitivityP1200087')
parser.add_argument('--flow', type=int,
                    help="f_lower value", default=10.)
parser.add_argument('--srat', type=int,
                    help="Sample rate", default=4096)

# ================= Signal Parameters =====================

parser.add_argument('--m1', type=float, default=0,
                    help="Mass of the first object")
parser.add_argument('--m2', type=float, default=0
                    help="Mass of the second object")
parser.add_argument('--mchirp', type=float, default=0,
                    help="Chirp mass of the system")
parser.add_argument('--mchirpscale', type=float, default=1,
                    help="Scale for chirp mass")
parser.add_argument('--q', type=float, default=0,
                    help="Mass ratio of the system")
parser.add_argument('--qscale', type=float, default=1,
                    help="Scale for q")
parser.add_argument('--ra', type=float,
                    help="Right ascension of the system, from 0 to 2PI")
parser.add_argument('--dec', type=float,
                    help="Declination of the system, from 0 to PI (North Pole to South Pole")
parser.add_argument('--tc', type=float,
                    help="Coalescence time (reference=center of the earth)")
parser.add_argument('--dist', type=float,
                    help="Distance to the system (Mpc)")
parser.add_argument('--apx', type=str,
                    help="Approximant", default='SEOBNRv4_opt')
parser.add_argument('--sp1z', type=float,
                    help="Spin1z", default=0)
parser.add_argument('--sp2z', type=float,
                    help="Spin2z", default=0)
parser.add_argument('--pol', type=float,
                    help="Polarization of the system", default=0)
parser.add_argument('--taper', type=str,
                    help="Tapers the signal. Options: TAPER_END, startend, TAPER_STARTEND, TAPER_NONE, start, TAPER_START, end", default='start')

# ================= File management/User interface parameters ============

parser.add_argument('--fileoutput', nargs='+',
                    help="Names of the HDF output files")
parser.add_argument('--plotsoutput', nargs='+',
                    help="If given, assigns the filenames for the plot images", default=False)
parser.add_argument('--plotshow', type=bool,
                    help="=True to interactively display plots (default False)", default=False)
parser.add_argument('--plotzoom', type=float,
                    help="How far zoomed in the plots are", default=0.05)

params = parser.parse_args()

# Define an injection class to inject a signal into noise

class SingleInjection(_HDFInjectionSet):
    def __init__(self, params):
        self.table = [params]
        self.extra_args = {}
        self.table = pycbc.io.WaveformArray.from_kwargs(**params)

# Assign m1 and m2 from mchirp and q

m1 = params.m1/params.scriptscale
m2 = params.m2/params.scriptscale
mchirp = params.mchirp/params.mchirpscale
q = params.q/params.qscale

if m1 == m2 == 0:
    m1 = mass1_from_mchirp_q(mchirp, q)
    m2 = mass2_from_mchirp_q(mchirp, q)

assert(m1 != 0 and m2 != 0), "M1 and M2 OR mchirp and q required."

# Define all signal parameters here
p = {'approximant' : params.apx,
     'mass1' : m1,
     'mass2' : m2,
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
        psd = pycbc.psd.from_string('AdVMidHighSensitivityP1200087', fsamples, df, f_lower)
    else:
        psd = pycbc.psd.from_string(params.psd, fsamples, df, f_lower)

    # Generate frequency psd
    htilde = pycbc.noise.gaussian.frequency_noise_from_psd(psd, seed=params.seed)

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

        zeros_zoom = zeros.time_slice(p['tc']-params.plotzoom, p['tc']+0.025)
        hoft_zoom = hoft.time_slice(p['tc']-params.plotzoom, p['tc']+0.025)
        line = p['tc']

        pylab.figure(figsize=(15,6))

        pylab.subplot(211)
        pylab.plot(hoft_zoom.sample_times, hoft_zoom, label="Signal+Noise")
        pylab.plot(zeros_zoom.sample_times, zeros_zoom, label="Signal", linewidth=2)
        pylab.axvline(line, color='black', linestyle='--', label="Coalescence Time")
        pylab.xlabel('Time (s)')
        pylab.ylabel('Strain $(\Delta L/L)$')
        pylab.xlim(p['tc'] - params.plotzoom, p['tc'] + 0.025)
        pylab.ylim(-1.5e-20, 1.5e-20)
        pylab.title('{} Simulated Data'.format(det))
        pylab.legend(loc='upper left')
        pylab.grid()

        pylab.subplot(212)
        pylab.pcolormesh(times, freqs, power**0.5, cmap='plasma')
        pylab.xlim(p['tc']-params.plotzoom, p['tc']+0.025)
        pylab.title('Spectrogram')
        pylab.xlabel('Time (s)')
        pylab.ylabel('Frequency (Hz)')
        pylab.yscale('log')
        pylab.axvline(line, color='black', linestyle='--')

        pylab.tight_layout()

        filename = cwd + '/' + plotsoutput[i]
        pylab.savefig(filename, dpi=400)
        if params.plotshow:
            pylab.show()

    i += 1