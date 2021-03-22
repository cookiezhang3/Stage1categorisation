#!/usr/bin/env python
from os import system, path, getcwd
from Tools.otherHelpers import submitJob, run
from collections import OrderedDict as od

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-s','--script', default='vbfDataDriven', help='Script to run')
parser.add_option('-l','--local', action="store_true", default=False, help='Run locally instead of on IC batch')
parser.add_option('-d','--dryRun', action="store_true", default=False, help='Print command instead of actually running it')
parser.add_option('-y','--year',default='2017', help='Specify if you want 2016, 2017, or 2018 data, or choose all of them together')
parser.add_option('-a','--additional', default=None, help='Add a json file specifying further options for your script')
(opts,args)=parser.parse_args()

myDir = getcwd()
baseDir = '/vols/cms/kz1617/Stage1categorisation/UltraLegacy/Pass2'
#baseDir = '/vols/cms/es811/Stage1categorisation/Legacy/Pass5'
#baseDir = '/vols/cms/es811/Stage1categorisation/UltraLegacy/Pass2'

if not path.isfile(opts.script):
  raise OSError('Your chosen script does not exist')

lumis = od()
lumis['2016'] = 35.9
lumis['2017'] = 41.5
lumis['2018'] = 59.7
lumis['all']  = 1.
if not opts.year in lumis.keys():
  raise KeyError('Your chosen year is not in the allowed options: %s'%lumis.keys())

#list of model hyper paramters used in training. e.g. ['max_depth:5,eta:0.01']
paramSets = [] 

if __name__=='__main__':
  jobDir = '%s/Jobs/%s/%s' % (myDir, opts.script.replace('.py',''), opts.year)
  if not path.isdir( jobDir ): system('mkdir -p %s'%jobDir)
  trainDir  = '%s/%s/trees'%(baseDir,opts.year)
  theCmd = 'python %s -t %s --intLumi %s'%(opts.script, trainDir, lumis[opts.year])
  if opts.dryRun: print theCmd
  elif opts.local: run(theCmd)
  elif len(paramSets)!=0:
    for params in paramSets:  
      theCmd += ' --trainParams %s '%params
      submitJob( jobDir, theCmd, params=params, dryRun=opts.dryRun)

  else: submitJob( jobDir, theCmd, dryRun=opts.dryRun )
