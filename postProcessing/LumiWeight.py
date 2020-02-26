import ROOT
import os
import numpy as np
import math
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module



class LumiWeight(Module):

    def __init__(self, lumiScaleFactor, sampleNormalization, isData=False):
        self.isData              = isData
        self.lumiScaleFactor     = lumiScaleFactor
        self.sampleNormalization = sampleNormalization

    def beginJob(self):
        pass

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        self.out.branch("weight", "F")
        if self.isData:
            self.out.branch("puWeight", "F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass


    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        if not self.isData:
            weight = self.lumiScaleFactor * event.genWeight / self.sampleNormalization
            self.out.fillBranch("weight", weight)
        else:
            # for data
            self.out.fillBranch("weight", 1)
            self.out.fillBranch("puWeight", 1)
        
        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed

lumiWeight = lambda : LumiWeight( 1000 )

