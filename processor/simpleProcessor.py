'''
Simple processor using coffea.
[ ] Missing pieces: appropriate sample handling
[ ] Accumulator caching

'''


import os
import time
import glob
import re
from functools import reduce
#from klepto.archives import dir_archive ### result caching not yet implemented

import numpy as np
from tqdm.auto import tqdm
import coffea.processor as processor
from coffea.processor.accumulator import AccumulatorABC
from coffea import hist
import pandas as pd
import uproot_methods

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# This just tells matplotlib not to open any
# interactive windows.
matplotlib.use('Agg')

class exampleProcessor(processor.ProcessorABC):
    """Dummy processor used to demonstrate the processor principle"""
    def __init__(self):
        dataset_axis    = hist.Cat("dataset", "Primary dataset")
        met_axis        = hist.Bin("met", r"$p_{T}^{miss}$ (GeV)", 600, 0, 1000)
        jet_pt_axis     = hist.Bin("jetpt", r"$p_{T}$", 600, 0, 1000)
        jet_eta_axis    = hist.Bin("jeteta", r"$\eta$", 60, -5.5, 5.5)

        self._accumulator = processor.dict_accumulator({
            "met" : hist.Hist("Counts", dataset_axis, met_axis),
            "jet_pt" : hist.Hist("Counts", dataset_axis, jet_pt_axis),
            "jet_pt_fwd" : hist.Hist("Counts", dataset_axis, jet_pt_axis),
            "jet_eta" : hist.Hist("Counts", dataset_axis, jet_eta_axis),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        """
        Processing function. This is where the actual analysis happens.
        """
        output = self.accumulator.identity()
        # We can access the data frame as usual
        # The dataset is written into the data frame
        # outside of this function

        # preselection of events
        selection = df['nLepton']==2
        #df = df[df['nLepton']==2]

        dataset = df["dataset"]

        # And fill the histograms
        output['met'].fill(dataset=dataset,
                            met=df["MET_pt"][selection].flatten())
        output['jet_pt'].fill(dataset=dataset,
                            jetpt=df["Jet_pt"].max().flatten()) # maximum jet pt
        output['jet_eta'].fill(dataset=dataset,
                            jeteta=df["Jet_eta"].flatten())

        ## We can also do arbitrary transformations
        ## E.g.: Sum of MET and the leading jet PTs
        #new_variable = df["MET_pt"] + df["Jet_pt"].max()
        #output['new_variable'].fill(dataset=dataset,
        #                    new_variable=new_variable)

        # To apply selections, simply mask
        # Let's see events with MET > 100
        mask = abs(df["Jet_eta"]) > 2.4

        # And plot the leading jet pt for these events
        output['jet_pt_fwd'].fill(dataset=dataset,
                            jetpt=df["Jet_pt"][mask].max().flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator


def main():
    # Inputs are defined in a dictionary
    # dataset : list of files
    fileset = {
        'tW_scattering': glob.glob("/home/users/dspitzba/ttw_samples/tW_scattering_private_Autumn18/*processed.root"),
        "TTW":           glob.glob("/home/users/dspitzba/ttw_samples/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/*processed.root")
    }

    # Run the processor
    output = processor.run_uproot_job(fileset,
                                  treename='Events',
                                  processor_instance=exampleProcessor(),
                                  executor=processor.futures_executor,
                                  executor_args={'workers': 1, 'function_args': {'flatten': False}},
                                  chunksize=500000,
                                 )

    # Make a few plots
    outdir = "./tmp_plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for name in ["met", "jet_pt", "jet_pt_fwd","jet_eta"]:
        histogram = output[name]
        if name == 'met':
            # rebin
            new_met_bins = hist.Bin('met', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)
            histogram = histogram.rebin('met', new_met_bins)

        ax = hist.plot1d(histogram,overlay="dataset", density=True) # make density plots because we don't care about x-sec differences
        ax.set_yscale('linear') # can be log
        #ax.set_ylim(0,0.1)

        ax.figure.savefig(os.path.join(outdir, "{}.pdf".format(name)))

        ax.clear()


if __name__ == "__main__":
    main()



