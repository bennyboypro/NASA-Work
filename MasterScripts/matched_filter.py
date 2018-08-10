import argparse
import pylab
import pycbc.psd
import pycbc.io
import numpy
import os
import h5py
from pycbc.types import timeseries, FrequencySeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import highpass, matched_filter_core, resample_to_delta_t
from pycbc.waveform import get_waveform_filter
from pycbc.io.live import SingleCoincForGraceDB
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q, mchirp_from_mass1_mass2

parser = argparse.ArgumentParser()


parser.add_argument('--m1', type=float,
                    help="Mass of the first object", default=0)
parser.add_argument('--m1scale', type=float,
                    help="scale factor for M1", default=1)
parser.add_argument('--m2', type=float,
                    help="Mass of the second object", default=0)
parser.add_argument('--m2scale', type=float,
                    help="scale factor for M2", default=1)
parser.add_argument('--mchirp', type=float,
                    help="Chirp mass of the system", default=0)
parser.add_argument('--mchirpscale', type=float,
                    help="Scale for chirp mass", default=1)
parser.add_argument('--q', type=float,
                    help="Mass ratio of the system", default=0)
parser.add_argument('--qscale', type=float,
                    help="Scale for q", default=1)
parser.add_argument('--apx', type=str,
                    help="Approximant", default='SEOBNRv4_ROM')
parser.add_argument('--sp1z', type=float,
                    help="Spin1z", default=0)
parser.add_argument('--sp2z', type=float,
                    help="Spin2z", default=0)
parser.add_argument('--pol', type=float,
                    help="Polarization", default=0)
parser.add_argument('--flow', type=float,
                    help="f_lower value", default=10)


parser.add_argument('--ifos', nargs='+',
                    help="Detectors involved in the data generation and collection")
parser.add_argument('--xml', type=str,
                    help="Name of the resulting .xml file")
parser.add_argument('--tc', type=float,
                    help="TC of the fake signal")
parser.add_argument('--threshold', type=float,
                    help="SNR Threshold for detectors", default=5.5)

parser.add_argument('--statfile', type=str,
                    help="h5py statistics file name", default="stats.h5")
parser.add_argument('--dsname', type=str,
                    help="dataset name for h5py file")


parser.add_argument('--infile', nargs='+',
                    help="Names of the signal HDF files")
parser.add_argument('--outfile', nargs='+',
                    help="Names of the output HDF files")
parser.add_argument('--plotsout', nargs='+',
                    help="If given, assigns the file names for the SNR time series image", default=False)
parser.add_argument('--plotzoom', type=float,
                    help="How far zoomed in the plots are", default=0.2)
parser.add_argument('--title', type=str,
                    help="Title of the plot", default="SNR Time Series")

params = parser.parse_args()

i = 0

# Convert m1 and m2 to mchirp and q

m1 = params.m1/params.m1scale
m2 = params.m2/params.m2scale
mchirp = params.mchirp/params.mchirpscale
q = params.q/params.qscale

if m1 == m2 == 0:
    m1 = mass1_from_mchirp_q(mchirp, q)
    m2 = mass2_from_mchirp_q(mchirp, q)

assert(m1 != 0 and m2 != 0), "M1 and M2 OR mchirp and q required."


# Get current working directory for saving files
cwd = os.getcwd()


infile = params.infile
outfile = params.outfile
plotsout = params.plotsout
ifos = params.ifos

avg_flag = 0
snr_peaks = []
flags = []
snrs = {}
psdlist = {}

networksnr = []

psds = {}
trig_ifos = []
results = {}
followup_data = {}

pylab.figure(figsize=[8, 6])
pylab.ylabel('Signal-to-noise')
pylab.xlabel('Time (s)')
pylab.xlim(params.tc - params.plotzoom, params.tc + params.plotzoom)
pylab.ylim([0, 30])

for filename in infile:

    # Load in data files
    file = cwd + '/' + filename
    hoft = pycbc.types.timeseries.load_timeseries(file)

    # =========================
    # = Precondition the data =
    # =========================

    # Remove low frequency noise
    hoft = highpass(hoft, 15.0)

    hoft = resample_to_delta_t(hoft, 1./2048.)

    # Remove 2 seconds from the start and the end to protect against edge effects
    conditioned = hoft.crop(2, 2)

    # Calculate PSD of data
    psd = conditioned.psd(4)
    psd = interpolate(psd, conditioned.delta_f)
    psd = inverse_spectrum_truncation(psd, 4 * conditioned.sample_rate, low_frequency_cutoff=15)

    psdlist[i] = psd



    # Generate a zeros frequency series the same length as the psd
    filter = FrequencySeries(numpy.zeros(len(psd)), 1, dtype=numpy.complex128)

    # Generate the waveform filter with given params
    filter = get_waveform_filter(filter,
                                 approximant=params.apx,
                                 mass1=m1,
                                 mass2=m2,
                                 sp1z=params.sp1z,
                                 sp2z=params.sp2z,
                                 polarization=params.pol,
                                 delta_f=psd.delta_f,
                                 f_lower=params.flow)

    # Calculate the SNR Time Series
    snr, _, norm = matched_filter_core(filter, conditioned, psd=psd, low_frequency_cutoff=20)

    snr = snr * norm

    # Trim SNR Time Series to account for edge effects
    snr = snr.crop(4 + 4, 4)

    snrs[i] = snr

    # Get data of the peak
    peak = abs(snr).numpy().argmax()
    snrp = snr[peak]
    time = snr.sample_times[peak]
    phase = numpy.angle(snrp)

    snrname = cwd + '/' + outfile[i]
    snr.save(snrname)

    # Plot the SNRTS
    pylab.plot(snr.sample_times, abs(snr), label='{} Time:{} Peak:{}'.format(ifos[i], round(time, 3), round(abs(snrp), 3)))

    networksnr.append(abs(snrp))

    # Prepping the data for SingleCoinc
    if abs(snrp) > params.threshold:

        snr_peaks.append(peak)

        psds[ifos[i]] = psd * (pycbc.DYN_RANGE_FAC ** 2.0) #psds are scaled before operating on them, to preserve accuracy with very negative exponentials
        trig_ifos.append(ifos[i])

        base = 'foreground/{}/'.format(ifos[i])
        results['foreground/stat'] = 0
        results['foreground/ifar'] = 1
        results[base+'template_id'] = 0
        results[base+'mass1'] = m1
        results[base+'mass2'] = m2
        results[base+'end_time'] = time # peak time in seconds, can find that within the timeseries object (using epoch)
        results[base+'sigmasq'] = norm
        results[base+'snr'] = abs(snrp)
        results[base+'coa_phase'] = phase

        # add in the SNRTS, odd number and centered, to the additional data
        snr_series = snr[peak - 1024:peak + 1025]
        snr_series = snr_series.astype(numpy.complex64)
        assert (len(snr_series) % 2)

        followup_data[ifos[i]] = {'snr_series': snr_series,  # centered time series
                                  'psd': psd}  # psd of data
    else:
        # Even if the signal is below threshold, we can use the PSD as info in BAYESTAR
        avg_flag = 1

        psdlist[i] = psd * (pycbc.DYN_RANGE_FAC ** 2.0)
        psds[ifos[i]] = psd * (pycbc.DYN_RANGE_FAC ** 2.0)

        flags.append(i)


    i += 1

assert(len(snr_peaks) >= 2)

# Calculate Network SNR, but only from peaks above threshold
runsum = 0
if avg_flag:
    for j in flags:

        avg = numpy.mean(snr_peaks).astype(numpy.int_)

        snr = snrs[j]

        snr_series = snr[avg-1024:avg+1025]
        snr_series = snr_series.astype(numpy.complex64)
        psd = psdlist[j]
        followup_data[ifos[j]] = {'snr_series': snr_series,
                                  'psd' : psd}

        temp = (networksnr[j])**2.0
        runsum += temp

    netsnr = (runsum)**0.5

else:
    k = 0
    for obj in networksnr:
        temp = (networksnr[k])**2.0
        runsum += temp
        k += 1

    netsnr = (runsum)**0.5

# Write Network SNR data to H5 file
with h5py.File(params.statfile, 'a') as hf:
    ds = hf.create_dataset('net_snr/'+params.dsname, data=netsnr)
    ds.attrs['q'] = q
    ds.attrs['mchirp'] = mchirp

# Generate plot

plotname = cwd + '/' + plotsout[0]
pylab.plot([],[], ' ', label="Network SNR: {}".format(round(netsnr,3)))
pylab.axvline(params.tc, color='k', linestyle='--')
pylab.axhline(params.threshold, color='r', linestyle=':')

pylab.legend(loc=1)
pylab.title(params.title)
pylab.grid()
pylab.tight_layout()
pylab.savefig(plotname, dpi=400)


coinc = SingleCoincForGraceDB(trig_ifos, results, psds=psds, low_frequency_cutoff=20, followup_data=followup_data)

xmlname = params.xml

coinc.save(xmlname)
