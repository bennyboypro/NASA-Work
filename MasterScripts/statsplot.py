import h5py
import pylab
import argparse
import numpy

parser = argparse.ArgumentParser()

parser.add_argument('--x', type=str,
                    help="Independent variable")
parser.add_argument('--stat', type=str,
                    help="Which stat to plot (area or sep)")
parser.add_argument('--input', type=str,
                    help="h5 file with stats")
parser.add_argument('--exact', type=float,
                    help="Exact parameter value")
parser.add_argument('--space', type=float,
                    help="Template spacing")
parser.add_argument('--title', type=str,
                    help="Title of the plot")
parser.add_argument('--filename', type=str, default=False,
                    help="Name of the output file")

params = parser.parse_args()

assert(params.stat == 'area' or params.stat == 'sep'), "--stat must be 'area' or 'sep'"

x1 = []
stat = []
x2 = []
nets = []

with h5py.File(params.input, 'r') as hf:

    group = hf['net_snr']
    for dsname in group:
        netsnr = group[dsname][()]

        x2.append(float(group[dsname].attrs[params.x]))
        nets.append(netsnr)

    if params.stat == 'area':
        group = hf['area']
        for dsname in group:
            area = group[dsname][:]

            x1.append(float(group[dsname].attrs[params.x]))
            stat.append(area[0,1])
    if params.stat == 'sep':
        group = hf['sep']
        for dsname in group:
            sep = group[dsname][()]

            x1.append(float(group[dsname].attrs[params.x]))
            stat.append(sep)



x1 = numpy.array(x1)
x2 = numpy.array(x2)

stat = numpy.array(stat)
nets = numpy.array(nets)

max = numpy.max(nets)
norm = nets/max


pylab.figure(figsize=[8,9])

if params.x == 'mchirp':
    pylab.xlabel('Chirp mass used in search ($M_\odot$)')

if params.x == 'q':
    pylab.xlabel('Mass ratio used in search')

pylab.grid()

sort1 = numpy.argsort(x1)
sort2 = numpy.argsort(x2)

f4 = pylab.axvline(params.exact, linestyle='--', color='r', label="Simulated value")
f5 = pylab.axvspan(params.exact-params.space, params.exact+params.space, facecolor='grey', alpha=0.5, label="Template spacing")

f3 = pylab.plot(x2[sort2], norm[sort2], '-o', markersize=2, color='k', label="Normalized SNR")

pylab.legend([f4, f5], [f4.get_label(), f5.get_label()], loc='upper right')

pylab.ylabel('Normalized SNR')

ax2 = pylab.twinx()

if params.stat == 'area':
    label = "90% credible region"
    pylab.ylabel('Sky localization uncertainty ($deg^2$)')

if params.stat == 'sep':
    label = "Angular separation between real and calculated"
    pylab.ylabel('Angular separation (deg)')

f2 = pylab.plot(x1[sort1], stat[sort1], '-o', markersize=2, color='g',label=label)

pylab.title(params.title)

fs = f2+f3
labels = [f.get_label() for f in fs]
ax2.legend(fs, labels, loc='lower right')

if params.filename is not False:
    pylab.savefig(params.filename)

pylab.show()