import argparse
import pylab
import pycbc.psd
import pycbc.io
import numpy
import os
from pycbc.types import timeseries, FrequencySeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import highpass, matched_filter_core
from pycbc.waveform import get_waveform_filter
from pycbc.io.live import SingleCoincForGraceDB

parser = argparse.ArgumentParser()

# =============== Filter parameters ==================

parser.add_argument('--m1', type=float,   help="[REQUIRED]Mass of the first object")
parser.add_argument('--m2', type=float,   help="[REQUIRED]Mass of the second object")
parser.add_argument('--apx', type=str,    help="[OPTIONAL]Approximant", default='IMRPhenomD')
parser.add_argument('--sp1z', type=float, help="[OPTIONAL]Spin1z", default=0)
parser.add_argument('--sp2z', type=float, help="[OPTIONAL]Spin2z", default=0)
parser.add_argument('--pol', type=float,  help="[OPTIONAL]Polarization", default=0)
parser.add_argument('--flow', type=float, help="[OPTIONAL]f_lower value", default=10)

# ============= SingleCoinc parameters ===============

parser.add_argument('--ifos', nargs='+', type=str, help="[REQUIRED]Detectors involved in the data generation and collection")


# ========== File management parameters ==============

parser.add_argument('--infile', nargs='+',    help="[REQUIRED]Paths to the signal files")
parser.add_argument('--outfile', nargs='+',   help="[REQUIRED]Names of the output files")
parser.add_argument('--plotsout', nargs='+',  help="[OPTIONAL]If given, assigns the file names for the plot images",
                    default=False)
parser.add_argument('--plotshow', type=bool,  help="[OPTIONAL]=True if you want to display the plots (default=False)",
                    default=False)
parser.add_argument('--plotzoom', type=float, help="[OPTIONAL]How far zoomed in the plots are", default=1)

params = parser.parse_args()

i = 0
infile = params.infile
outfile = params.outfile
plotsout = params.plotsout

for file in infile:

    hoft = pycbc.types.timeseries.load_timeseries(file)
    # =========================
    # = Precondition the data =
    # =========================

    # Remove low frequency noise
    hoft = highpass(hoft, 15.0)

    # Remove 2 seconds from the start and the end to protect against edge effects
    conditioned = hoft.crop(2, 2)

    # Calculate PSD of data
    psd = conditioned.psd(4)
    psd = interpolate(psd, conditioned.delta_f)
    psd = inverse_spectrum_truncation(psd, 4 * conditioned.sample_rate, low_frequency_cutoff=15)

    # Generate a zeros frequency series the same length as the psd
    filter = FrequencySeries(numpy.zeros(len(psd)), 1, dtype=numpy.complex128)

    # Generate the waveform filter with given params
    filter = get_waveform_filter(filter,
                                 approximant=params.apx,
                                 mass1=params.m1,
                                 mass2=params.m2,
                                 sp1z=params.sp1z,
                                 sp2z=params.sp2z,
                                 polarization=params.pol,
                                 delta_f=psd.delta_f,
                                 f_lower=params.flow)

    # Calculate the SNR Time Series
    snr, _, norm = matched_filter_core(filter, conditioned, psd=psd, low_frequency_cutoff=20)

    # Trim SNR Time Series to account for edge effects
    snr = snr.crop(4 + 4, 4)

    # Want real part of SNRTS
    snr = abs(snr)

    # Get current working directory for saving files
    cwd = os.getcwd()

    # Get data of the peak
    peak = abs(snr).numpy().argmax()
    snrp = snr[peak]
    time = snr.sample_times[peak]
    phase = numpy.angle(snrp)

    # Plot the SNRTS
    if params.plotsout != False:
        pylab.figure(figsize=[10, 4])
        pylab.plot(snr.sample_times, abs(snr))
        pylab.ylabel('Signal-to-noise')
        pylab.xlabel('Time (s)')
        pylab.xlim(time - params.plotzoom, time + params.plotzoom)
        plotname = cwd + '/{}'.format(plotsout[i])
        pylab.savefig(plotname)
        if params.plotshow:
            pylab.show()

        # Save the SNR Time Series
        filename = cwd + '/{}'.format(outfile[i])
        snr.save(filename)

        print('Peak signal at {}s with SNR {} with phase {}'.format(time, abs(snrp), phase))

    psds = {'{}'.format(params.ifos)}

i += 1