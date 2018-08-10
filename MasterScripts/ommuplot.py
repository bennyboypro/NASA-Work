import h5py
import pylab
import argparse
import numpy

parser = argparse.ArgumentParser()

parser.add_argument('--stat', type=str,
                    help="Picks area (area) or separation (sep)")
parser.add_argument('--input', nargs='+', type=str,
                    help="h5 file with stats")
parser.add_argument('--title', type=str,
                    help="Title of the plot")
parser.add_argument('--filename', type=str, default=False,
                    help="Name of the output file")

params = parser.parse_args()

assert (params.stat == 'area' or params.stat == 'sep'), "--stat must be 'area' or 'sep'"

pylab.figure(figsize=[8,9])

pylab.xlabel('Fraction of Recovered Optimal SNR ($\mu$)')

pylab.grid()

x = 'mchirp'

for file in params.input:

    x1 = []
    stat = []
    x2 = []
    nets = []

    with h5py.File(file, 'r') as hf:

        if params.stat == 'area':
            group = hf['area']
            for dsname in group:
                area = group[dsname][:]

                x1.append(float(group[dsname].attrs[x]))
                stat.append(area[0,1])

        if params.stat == 'sep':
            group = hf['sep']
            for dsname in group:
                sep = group[dsname][()]

                x1.append(float(group[dsname].attrs[x]))
                stat.append(sep)

        group = hf['net_snr']
        for dsname in group:
            netsnr = group[dsname][()]

            x2.append(float(group[dsname].attrs[x]))
            nets.append(netsnr)

    x1 = numpy.array(x1)
    x2 = numpy.array(x2)

    stat = numpy.array(stat)
    nets = numpy.array(nets)

    max = numpy.max(nets)
    norm = nets/max

    sort1 = numpy.argsort(x1)
    sort2 = numpy.argsort(x2)


    if params.stat == 'area':
        pylab.ylabel('Sky Localization Uncertainty [$deg^2$] ($\Omega$)')
        label = '{} areas'.format(x)

    if params.stat == 'sep':
        pylab.ylabel('Angular separation [deg] ($\Omega$)')
        label = '{} separation'.format(x)

    pylab.plot(numpy.resize(norm[sort2], len(stat)), stat[sort1], 'o', markersize=4, label=label)


    pylab.title(params.title)

    x = 'q'


pylab.legend()

if params.filename is not False:
    pylab.savefig(params.filename)

pylab.show()