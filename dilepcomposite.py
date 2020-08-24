'''
Simple processor using coffea.
[x] weights
[ ] Missing pieces: appropriate sample handling
[x] Accumulator caching
'''


import os
import time
import glob
import re
from functools import reduce
from klepto.archives import dir_archive

import numpy as np
from tqdm.auto import tqdm
import coffea.processor as processor
from coffea.processor.accumulator import AccumulatorABC
from coffea.analysis_objects import JaggedCandidateArray
from coffea import hist
import pandas as pd
import uproot_methods
import awkward


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Tools.helpers import *

# This just tells matplotlib not to open any
# interactive windows.
matplotlib.use('Agg')

class exampleProcessor(processor.ProcessorABC):
    """Dummy processor used to demonstrate the processor principle"""
    def __init__(self):

        # we can use a large number of bins and rebin later
        dataset_axis        = hist.Cat("dataset",   "Primary dataset")
        pt_axis             = hist.Bin("pt",        r"$p_{T}$ (GeV)", 1000, 0, 1000)
        ht_axis             = hist.Bin("ht",        r"$H_{T}$ (GeV)", 500, 0, 5000)
        mass_axis           = hist.Bin("mass",      r"M (GeV)", 1000, 0, 2000)
        eta_axis            = hist.Bin("eta",       r"$\eta$", 60, -5.5, 5.5)
        multiplicity_axis   = hist.Bin("multiplicity",         r"N", 20, -0.5, 19.5)
        norm_axis            = hist.Bin("norm",         r"N", 25, 0, 1)
        
        self._accumulator = processor.dict_accumulator({
            "MET_pt" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "pt_spec_max" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "MT" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "HT" :          hist.Hist("Counts", dataset_axis, ht_axis),
            "ST" :          hist.Hist("Counts", dataset_axis, ht_axis),
            "mbj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mjj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlb_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlb_min" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlj_min" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "N_b" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_ele" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_diele" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_mu" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_dimu" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_spec" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "FWMT1" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT2" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT3" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT4" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT5" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "S" :               hist.Hist("Counts", dataset_axis, norm_axis),
            "S_lep" :           hist.Hist("Counts", dataset_axis, norm_axis),
            'diboson':          processor.defaultdict_accumulator(int),
            'ttbar':            processor.defaultdict_accumulator(int),
            'TTW':              processor.defaultdict_accumulator(int),
            'TTX':              processor.defaultdict_accumulator(int),
            'tW_scattering':    processor.defaultdict_accumulator(int),
            'totalEvents':      processor.defaultdict_accumulator(int),
            'DY':               processor.defaultdict_accumulator(int),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        """
        Processing function. This is where the actual analysis happens.
        """
        output = self.accumulator.identity()
        dataset = df["dataset"]
        cfg = loadConfig()
        # We can access the data frame as usual
        # The dataset is written into the data frame
        # outside of this function

        jet = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt = df['Jet_pt'].content,
            eta = df['Jet_eta'].content,
            phi = df['Jet_phi'].content,
            mass = df['Jet_mass'].content,
            goodjet = df['Jet_isGoodJetAll'].content,
            bjet = df['Jet_isGoodBJet'].content,
            jetId = df['Jet_jetId'].content,
            puId = df['Jet_puId'].content,
        )
        jet     = jet[(jet['goodjet']==1)]
        btag    = jet[jet['bjet']==1]
        light   = jet[(jet['goodjet']==1) & (jet['bjet']==0)]

        ## Leptons
        lepton = JaggedCandidateArray.candidatesfromcounts(
            df['nLepton'],
            pt = df['Lepton_pt'].content,
            eta = df['Lepton_eta'].content,
            phi = df['Lepton_phi'].content,
            mass = df['Lepton_mass'].content,
            pdgId = df['Lepton_pdgId'].content,
        )

        # MET_pt, MT, N_b, N_jets, HT, ST
        selection = ((df['nLepton']==2) & (df['nVetoLepton']==2)) & (df['isSS']==1) & (df['nGoodBTag']==2) & (df['nGoodJet']>=3)
        b_nonb_selection = (jet.counts>3) & (btag.counts==2) & (light.counts>=2) & (df['nLepton']==2) & (df['nVetoLepton']==2) & (df['isSS']==1)
        
        output['MET_pt'].fill(dataset=dataset, pt=df["MET_pt"][selection].flatten(), weight=df['weight'][selection]*cfg['lumi'])
        output['MT'].fill(dataset=dataset, pt=df["MT"][selection].flatten(), weight=df['weight'][selection]*cfg['lumi'])
        output['N_b'].fill(dataset=dataset, multiplicity=df["nGoodBTag"][selection], weight=df['weight'][selection]*cfg['lumi'] )
        output['N_jet'].fill(dataset=dataset, multiplicity=df["nGoodJet"][selection], weight=df['weight'][selection]*cfg['lumi'] )
        
        st = jet[jet['goodjet']==1].pt.sum() + lepton.pt.sum() + df['MET_pt']
        output['ST'].fill(dataset=dataset, ht=st[b_nonb_selection].flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['HT'].fill(dataset=dataset, ht=df['Jet_pt'][b_nonb_selection].sum().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        
        #mbj_max, mjj_max, mlb_max, mlb_min, mlj_max, mlj_min
        b_nonb_pair = btag.cross(light)
        jet_pair = light.choose(2)
        output['mbj_max'].fill(dataset=dataset, mass=b_nonb_pair[b_nonb_selection].mass.max().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['mjj_max'].fill(dataset=dataset, mass=jet_pair[b_nonb_selection].mass.max().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        lepton_bjet_pair = lepton.cross(btag)
        output['mlb_max'].fill(dataset=dataset, mass=lepton_bjet_pair[b_nonb_selection].mass.max().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['mlb_min'].fill(dataset=dataset, mass=lepton_bjet_pair[b_nonb_selection].mass.min().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        lepton_jet_pair = lepton.cross(jet)
        output['mlj_max'].fill(dataset=dataset, mass=lepton_jet_pair[b_nonb_selection].mass.max().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['mlj_min'].fill(dataset=dataset, mass=lepton_jet_pair[b_nonb_selection].mass.min().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])


        # spec
        spectator = jet[(abs(jet.eta)>2.0) & (abs(jet.eta)<4.7) & (jet.pt>25) & (jet['puId']>=7) & (jet['jetId']>=6)] # 40 GeV seemed good. let's try going lower
        output['N_spec'].fill(dataset=dataset, multiplicity=spectator[b_nonb_selection].counts, weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['pt_spec_max'].fill(dataset=dataset, pt=spectator[b_nonb_selection & (spectator.counts>0)].pt.max().flatten(), weight=df['weight'][b_nonb_selection & (spectator.counts>0)]*cfg['lumi'])
        
        ## Muons
        muon = lepton[abs(lepton['pdgId'])==13]
        dimuon = muon.choose(2)
        OSmuon = (dimuon.i0['pdgId'] * dimuon.i1['pdgId'] < 0)
        dimuon = dimuon[OSmuon]

        ## Electrons
        electron = lepton[abs(lepton['pdgId'])==11]
        dielectron = electron.choose(2)
        OSelectron = (dielectron.i0['pdgId'] * dielectron.i1['pdgId'] < 0)
        dielectron = dielectron[OSelectron]

        ## MET
        met_pt  = df["MET_pt"]
        met_phi = df["MET_phi"]

        ## Event classifieres
        
        
        ## define selections (maybe move to a different file at some point)
        trilep      = ((df['nLepton']==3) & (df['nVetoLepton']>=3))
        twoJet      = (jet.counts>2) # those are any two jets
        twoBTag     = (btag.counts>=2)
        twoMuon     = ( muon.counts==2 )
        #Zveto_mu    = ( (dimuon.counts<1) )# | (abs(dimuon.mass - 91)>15) )
        Zveto_mu    = ( ((dimuon.mass-91.)>15).counts<1 )
        Zveto_ele   = ( ((dielectron.mass-91.)>15).counts<1 )
        met         = (met_pt > 35)
        dilep       = ((df['nLepton']==2) & (df['nVetoLepton']>=2))
        fourJet     = (jet.counts>4)
        ss          = (df['isSS']==1)


        ## work on the cutflow
        output['totalEvents']['all'] += len(df['weight'])

        addRowToCutFlow( output, df, cfg, 'skim',        None ) # entry point
        addRowToCutFlow( output, df, cfg, 'dilep',      dilep )
        addRowToCutFlow( output, df, cfg, 'fourJet',      dilep & fourJet )
        addRowToCutFlow( output, df, cfg, 'twoBTag',     dilep & fourJet & twoBTag )
        addRowToCutFlow( output, df, cfg, 'ss',          dilep & fourJet & twoBTag & ss )
        addRowToCutFlow( output, df, cfg, 'met35',       dilep & fourJet & twoBTag & met )
        addRowToCutFlow( output, df, cfg, 'dimuon',      dilep & fourJet & twoBTag & met & twoMuon )
        addRowToCutFlow( output, df, cfg, 'Zveto_mu',    dilep & fourJet & twoBTag & met & Zveto_mu )
        addRowToCutFlow( output, df, cfg, 'Zveto',       dilep & fourJet & twoBTag & met & Zveto_mu & Zveto_ele )

        # preselection of events
        event_selection = dilep & fourJet & twoBTag & met
        
        ## And fill the histograms
        # just the number of electrons and muons
        output['N_ele'].fill(dataset=dataset, multiplicity=electron[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])
        output['N_mu'].fill(dataset=dataset, multiplicity=muon[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])

        # something a bit more tricky
        output['N_diele'].fill(dataset=dataset, multiplicity=dielectron[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])
        output['N_dimu'].fill(dataset=dataset, multiplicity=dimuon[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])

        return output

    def postprocess(self, accumulator):
        return accumulator


def main():

    overwrite = True

    # load the config and the cache
    cfg = loadConfig()

    # Inputs are defined in a dictionary
    # dataset : list of files
    from processor.samples import fileset, fileset_small, fileset_2l

    # histograms
    histograms = []
    histograms += ['N_ele', 'N_mu', 'N_diele', 'N_dimu', 'MET_pt', 'pt_spec_max', 'MT', 'HT', 'ST', 'mbj_max', 'mjj_max', 'mlb_max', 'mlb_min', 'mlj_max', 'mlj_min', 'N_b', 'N_jet', 'N_spec']
    
    # initialize cache
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['singleLep']), serialized=True)
    if not overwrite:
        cache.load()

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and fileset == cache.get('fileset_2l') and cache.get('simple_output'):
        output = cache.get('simple_output')

    else:
        # Run the processor
        output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=exampleProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': 16, 'function_args': {'flatten': False}},
                                      chunksize=100000,
                                     )
        cache['fileset']        = fileset_2l
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()

    # Make a few plots
    outdir = "./tmp_plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return output

if __name__ == "__main__":
    output = main()
