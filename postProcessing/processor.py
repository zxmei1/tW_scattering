'''
Use nanoAODTools processing to skim and slim nanoAOD
Submit one post processing job per nanoAOD file to condor using ProjectMetis
Submission tarball is independent of RootTools/Samples

'''

import os

from RootTools.core.standard import *
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

from Samples.nanoAOD.Fall17_ucsd import *

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

# Logger
import tW_scattering.Tools.logger as logger
import RootTools.core.logger as logger_rt
logger    = logger.get_logger(   options.logLevel, logFile = None)
logger_rt = logger_rt.get_logger(options.logLevel, logFile = None)

logger.info("Searching for sample %s"%options.samples[0])

samples = []
for selectedSample in options.samples:
    for sample in allSamples:
        if selectedSample == sample.name:
            samples.append(sample)
            logger.info("Adding sample %s", sample.name)
            logger.info("Sample has normalization %s", sample.normalization)
            sample.normalization = float(sample.normalization)


if len(samples)==0:
    logger.info( "No samples found. Was looking for %s. Exiting" % options.samples )
    sys.exit(-1)

sample_name = samples[0].name

if len(samples)>1:
    sample_name =  samples[0].name+"_comb"
    logger.info( "Combining samples %s to %s.", ",".join(s.name for s in samples), sample_name )
    sample = Sample.combine(sample_name, samples, maxN = None)
    # Clean up
    for s in samples:
        sample.clear()
    logger.info("Final normalization is %s", sample.normalization)
elif len(samples)==1:
    sample = samples[0]
else:
    raise ValueError( "Need at least one sample. Got %r",samples )

logger.info("Sample contains %s files", len(sample.files))
sample.files = sorted(sample.files) # in order to avoid some random ordered file list, different in each job


outDir = '/home/users/dspitzba/ttw_samples/'

nbSkim = 'nJet>0&&(nElectron+nMuon)>0'

# Load modules
from LumiWeight import LumiWeight

modules = [
    LumiWeight( 1000., float(sample.normalization) ),
#    ObjectSelection( year=2017 )
    ]

len_orig = len(sample.files)
## sort the list of files?
sample = sample.split( n=options.nJobs, nSub=options.job)
logger.info( "fileBasedSplitting: Run over %i/%i files for job %i/%i."%(len(sample.files), len_orig, options.job, options.nJobs))
logger.debug( "fileBasedSplitting: Files to be run over:\n%s", "\n".join(sample.files) )

p = PostProcessor(os.path.join(outDir, sample_name), sample.files, modules=modules, outputbranchsel='keep_and_drop.txt', branchsel='keep_and_drop.txt', cut=nbSkim, haddFileName=os.path.join(outDir, sample_name, sample_name+'_%s_processed.root'%options.job), prefetch=False)

p.run()

