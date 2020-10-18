'''
small script that reades histograms from an archive and saves figures in a public space
ToDo:
[x] Cosmetics (labels etc)
[ ] ratio pad!
  [ ] pseudo data
[ ] uncertainty band
'''


from coffea import hist
import pandas as pd
#import uproot_methods

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Tools.helpers import *
from klepto.archives import dir_archive

def saveFig( fig, ax, rax, path, name, scale='linear', shape=False, y_max=-1 ):
    outdir = os.path.join(path,scale)
    finalizePlotDir(outdir)
    ax.set_yscale(scale)
    ax.set_ylabel('Events')

    y_min = 0.0005 if shape else 0.05
    if y_max>0:
        y_max = 0.1 if shape else 300*y_max
    else:
        y_max = 3e7
    if scale == 'log':
        ax.set_ylim(y_min, y_max)
        #if shape:
        #     ax.yaxis.set_ticks(np.array([10e-4,10e-3,10e-2,10e-1,10e0]))
        #else:
        #    ax.yaxis.set_ticks(np.array([10e-2,10e-1,10e0,10e1,10e2,10e3,10e4,10e5,10e6]))
    else:
        ax.set_ylim(0.0, y_max)


    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for handle, label in zip(handles, labels):
        #print (handle, label)
        try:
            new_labels.append(my_labels[label])
            if not label=='tW_scattering':
                handle.set_color(colors[label])
        except:
            pass

    """if rax:
        plt.subplots_adjust(hspace=0)
        rax.set_ylabel('Obs./Pred.')
        rax.set_ylim(0.5,1.5)"""

    ax.legend(title='',ncol=2,handles=handles, labels=new_labels, frameon=False) 

    fig.text(0., 0.995, '$\\bf{CMS}$', fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )
    fig.text(0.15, 1., '$\\it{Simulation}$', fontsize=14, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )
    fig.text(0.8, 1., '13 TeV', fontsize=14, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )

    fig.savefig(os.path.join(outdir, "{}.pdf".format(name)))
    fig.savefig(os.path.join(outdir, "{}.png".format(name)))
    #ax.clear()

colors = {
    'tW_scattering': '#FF595E',
    'TTW': '#8AC926',
    'TTX': '#FFCA3A',
    'ttbar': '#1982C4',
    'wjets': '#6A4C93',
}
'''
other colors (sets from coolers.com):
#525B76 (gray)
#34623F (hunter green)
#0F7173 (Skobeloff)
'''

my_labels = {
    'tW_scattering': 'tW scattering',
    'TTW': r'$t\bar{t}$W+jets',
    'TTX': r'$t\bar{t}$Z/H',
    'ttbar': r'$t\bar{t}$+jets',
    'wjets': 'W+jets',
    'uncertainty': 'Uncertainty',
    'diboson': 'diboson',
    'DY': 'Drell-Yan'
}

fill_opts = {
    'edgecolor': (0,0,0,0.3),
    'alpha': 1.0
}

error_opts = {
    'label': 'uncertainty',
    'hatch': '///',
    'facecolor': 'none',
    'edgecolor': (0,0,0,.5),
    'linewidth': 0
}

# load the configuration
cfg = loadConfig()

# load the results
cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['simpleProcessor']), serialized=True)
cache.load()

histograms = cache.get('histograms')
output = cache.get('simple_output')
plotDir = os.path.expandvars(cfg['meta']['plots']) + '/10.17pre_withoutSS/'
finalizePlotDir(plotDir)

if not histograms:
    print ("Couldn't find histograms in archive. Quitting.")
    exit()

print ("Plots will appear here:", plotDir )

for name in histograms:
    print (name)
    skip = False
    histogram = output[name]
    if name == 'MET_pt':
        # rebin
        new_met_bins = hist.Bin('pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)
        histogram = histogram.rebin('pt', new_met_bins)
    elif name == 'MT':
        # rebin
        new_met_bins = hist.Bin('pt', r'$M_T \ (GeV)$', 20, 0, 200)
        histogram = histogram.rebin('pt', new_met_bins)
    elif name == 'N_jet':
        # rebin
        new_n_bins = hist.Bin('multiplicity', r'$N_{jet}$', 15, -0.5, 14.5)
        histogram = histogram.rebin('multiplicity', new_n_bins)
    elif name == 'N_b':
        # rebin
        new_n_bins = hist.Bin('multiplicity', r'$N_{b-jet}$', 5, -0.5, 4.5)
        histogram = histogram.rebin('multiplicity', new_n_bins)
    elif name == 'b_nonb_massmax':
        # rebin
        new_mass_bins = hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'b_b_nonb_massmax':
        # rebin
        new_mass_bins = hist.Bin('mass', r'$M(2b, light) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'jet_pair_massmax':
        # rebin
        new_mass_bins = hist.Bin('mass', r'$M(jet pair) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'lepton_jet_pair_massmax':
        # rebin
        new_mass_bins = hist.Bin('mass', r'$M(lepton+jet) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'lepton_bjet_pair_massmax':
        # rebin
        new_mass_bins = hist.Bin('mass', r'$M(lepton+bjet) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    else:
        skip = True
    if not skip:
        y_max = histogram.sum("dataset").values(overflow='over')[()].max()
        y_over = histogram.sum("dataset").values(overflow='over')[()][-1]

        import re
        notdata = re.compile('(?!pseudodata)')

        fig, ax = plt.subplots(1, 1, figsize=(7,7)) #2,1 if you want obs/pred, and change ax to (ax,rax)

        # get axes
        hist.plot1d(histogram[notdata],overlay="dataset", ax=ax, stack=True, overflow='over', clear=False, line_opts=None, fill_opts=fill_opts, error_opts=error_opts, order=['diboson','TTX', 'TTW','ttbar', 'DY']) #error_opts??
        hist.plot1d(histogram['tW_scattering'], overlay="dataset", ax=ax, overflow='over',clear=False, line_opts={'linewidth':3})


        for l in ['linear', 'log']:
            saveFig(fig, ax, None, plotDir, name, scale=l, shape=False, y_max=y_max) #and put arx instead of None to make the ratio plot
        fig.clear()
        #rax.clear()
        #pd_ax.clear()
        ax.clear()

        fig, ax = plt.subplots(1,1,figsize=(7,7))

        hist.plot1d(histogram[notdata],overlay="dataset", density=True, stack=False, overflow='over', ax=ax) # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(fig, ax, None, plotDir, name+'_shape', scale=l, shape=True)
        fig.clear()
        ax.clear()
