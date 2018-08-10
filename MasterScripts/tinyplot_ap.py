import argparse
import numpy as np
import pylab as pl
from astropy.io.fits import Header
from astropy.wcs import WCS
from astropy.visualization.wcsaxes.frame import RectangularFrame
from reproject import reproject_from_healpix
from ligo.skymap.io import read_sky_map
import ligo.skymap.plot
from lalinference.bayestar import postprocess
import healpy as hp
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--fits_file', type=str, help='FITS file to plot')
parser.add_argument('--opt', type=str, help='Optimal FITS file')
parser.add_argument('--output-file', type=str, metavar='FILE',
                    help='If given, save the plot to FILE, otherwise show it interactively')
parser.add_argument('--radec', type=str, nargs='+', default=[0, 0], metavar='RA DEC',
                    help='Center the map at the given spot, in degrees')
parser.add_argument('--size', type=float, default=10,
                    help='Size of the field of view in degrees')
parser.add_argument('--title', type=str, default=None,
                    help='If given, include a title to the plot')

parser.add_argument('--q', type=float, help="mass ratio")
parser.add_argument('--qscale', type=float, default=1, help="scaling factor for q")
parser.add_argument('--mchirp', type=float, help="Chirp mass")
parser.add_argument('--mchirpscale', type=float, default=1, help="scaling factor for mchirp")
parser.add_argument('--dsname', type=str, help='dataset name for h5 file')
parser.add_argument('--statsfile', type=str, help='Stats file')

args = parser.parse_args()

## Plot the Optimal Skymap

optprob, _ = read_sky_map(args.opt, nest=False, distances=False) ###

radec = [float(x) for x in args.radec]

deg_per_pix = 4
header = Header(dict(
    NAXIS=2,
    NAXIS1=360*deg_per_pix, NAXIS2=180*deg_per_pix, # number of pixels
    CRPIX1=180*deg_per_pix, CRPIX2=90*deg_per_pix, # reference pixel
    CRVAL1=radec[0], CRVAL2=radec[1], # physical value at reference pixel
    CDELT1=1./deg_per_pix,
    CDELT2=1./deg_per_pix,
    CTYPE1='RA---AIT',
    CTYPE2='DEC--AIT',
    RADESYS='ICRS'))
wcs = WCS(header)

fig = pl.figure(figsize=[8,6])
ax = pl.axes(projection=wcs, frame_class=RectangularFrame, aspect=1)
world_transform = ax.get_transform('world')
ax.set_xlim((180 - args.size) * deg_per_pix,
            (180 + args.size) * deg_per_pix)
ax.set_ylim((90 - args.size) * deg_per_pix,
            (90 + args.size) * deg_per_pix)


cls = postprocess.find_greedy_credible_levels(optprob)
cls, _ = reproject_from_healpix((cls, 'icrs'), header)
cs = ax.contour(cls, levels=[0.5, 0.9], colors='C0')
ax.plot([],[], '-', color='C0', label='Exact masses')


## Work on the suboptimal skymap

prob, _ = read_sky_map(args.fits_file, nest=False, distances=False)
nside = hp.npix2nside(len(prob))
deg2perpix = hp.nside2pixarea(nside, degrees=True)

radec = [float(x) for x in args.radec]

deg_per_pix = 4
header = Header(dict(
    NAXIS=2,
    NAXIS1=360*deg_per_pix, NAXIS2=180*deg_per_pix, # number of pixels
    CRPIX1=180*deg_per_pix, CRPIX2=90*deg_per_pix, # reference pixel
    CRVAL1=radec[0], CRVAL2=radec[1], # physical value at reference pixel
    CDELT1=1./deg_per_pix,
    CDELT2=1./deg_per_pix,
    CTYPE1='RA---AIT',
    CTYPE2='DEC--AIT',
    RADESYS='ICRS'))
wcs = WCS(header)

cls = postprocess.find_greedy_credible_levels(prob)
clsplt, _ = reproject_from_healpix((cls, 'icrs'), header)
cs = ax.contour(clsplt, levels=[0.5, 0.9], colors='C3')
ax.plot([],[],'-', color='C3', label='Incorrect masses')

## Calculate area of 90% contour

cls = 100 * cls

pp = 90
ii = np.searchsorted(np.sort(cls), 90) * deg2perpix
area = np.vstack((pp, ii)).T

## Calculate angular separation between optimum point and max pixel

maxpix = np.argmax(prob)

ang = hp.pix2ang(nside, maxpix, nest=False, lonlat=True)

ra = np.deg2rad(radec[0])
dec = np.deg2rad(radec[1])
angra = np.deg2rad(ang[0])
angdec = np.deg2rad(ang[1])

sep = np.arccos(np.sin(dec)*np.sin(angdec)+np.cos(dec)*np.cos(angdec)*np.cos(ra-angra))

sep = np.rad2deg(sep)

print(sep)

## Write precision and accuracy to stats file

with h5py.File(args.statsfile, 'a') as hf:
    ds = hf.create_dataset('area/' + args.dsname, data=area)
    ds.attrs['q'] = args.q / args.qscale
    ds.attrs['mchirp'] = args.mchirp / args.mchirpscale

    ds = hf.create_dataset('sep/' + args.dsname, data=sep)
    ds.attrs['q'] = args.q / args.qscale
    ds.attrs['mchirp'] = args.mchirp / args.mchirpscale

ramax = ang[0]-radec[0]+180
decmax = ang[1]-radec[1]+90


if ramax > 360:
    ramax = ramax-360

ax.plot(ramax*deg_per_pix, decmax*deg_per_pix, 'x', color='red')
ax.plot(180 * deg_per_pix, 90 * deg_per_pix, '+', color='black',
        markersize=15, markeredgewidth=1, label="True location")

ax.grid(True, lw=1)
ax.set_xlabel('Right ascension')
ax.set_ylabel('Declination')

ax.plot([],[],' ', label='90%Area:{} $deg^2$'.format(round(area[0][1],3)))
ax.plot([],[],' ', label='Separation:{}$deg$'.format(round(sep,3)))

ax.legend(loc='lower center', bbox_to_anchor=(0.5,0.00), ncol=2)

pl.tight_layout()

if args.title is not None:
    pl.title(args.title)

if args.output_file is None:
    pl.show()
else:
    fig.savefig(args.output_file, dpi=400)