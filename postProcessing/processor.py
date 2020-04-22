'''
Use nanoAODTools processing to skim and slim nanoAOD
Submit one post processing job per nanoAOD file to condor using ProjectMetis
Submission tarball is independent of RootTools/Samples

'''

import os
import glob

from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from ObjectSelection import PhysicsObjects

# argparser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser for cmgPostProcessing")
argParser.add_argument('--logLevel',    action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging" )
argParser.add_argument('--samples',     action='store', nargs='*', type=str, default=['TTZToLLNuNu_ext'], help="List of samples to be post-processed, given as CMG component name" )
argParser.add_argument('--skim',        action='store', nargs='?', type=str, default='dimuon', help="Skim conditions to be applied for post-processing" )
argParser.add_argument('--job',         action='store', type=int, default=0, help="Run only jobs i" )
argParser.add_argument('--nJobs',       action='store', nargs='?', type=int,default=1, help="Maximum number of simultaneous jobs.")
argParser.add_argument('--prepare',     action='store_true', help="Prepare, don't acutally run" )
argParser.add_argument('--overwrite',   action='store_true', help="Overwrite" )
argParser.add_argument('--year',        action='store', default=None, help="Which year? Important for json file.")
argParser.add_argument('--era',         action='store', default="v1", help="Which era/subdirectory?")
options = argParser.parse_args()



outDir = '/home/users/dspitzba/ttw_samples/'

nbSkim = 'nJet>0&&(nElectron+nMuon)>0'

# Load modules
from LumiWeight import LumiWeight

modules = [
#    LumiWeight( 1000., float(sample.normalization) ),
    PhysicsObjects( year=2018 )
    ]

# signal
files = glob.glob('/hadoop/cms/store/user/dspitzba/tW_scattering/tW_scattering/nanoAOD/*.root')
sample_name = 'tW_scattering_private_Autumn18'

# central sample
files = glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/*.root')
sample_name = 'TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1'
files = files[:1]

skimmer = PostProcessor(os.path.join(outDir, sample_name), files, modules=modules, outputbranchsel='keep_and_drop.txt', branchsel='keep_and_drop.txt', cut=nbSkim, haddFileName=os.path.join(outDir, sample_name, sample_name+'_processed.root'), prefetch=False)

merger = PostProcessor(os.path.join(outDir, sample_name), glob.glob( os.path.join(outDir, sample_name) + '*Skim.root' ), haddFileName=os.path.join(outDir, sample_name, 'merged', sample_name+'_merged.root') )

skimmer.run()

merger.run()

