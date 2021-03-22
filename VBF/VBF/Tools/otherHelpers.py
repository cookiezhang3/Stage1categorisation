import numpy as np
import os

def run(cmd):
  print cmd
  os.system(cmd)

def prettyHist( hist ):
    hist.SetStats(0)
    hist.GetXaxis().SetTitle('Process')
    hist.GetXaxis().SetTickLength(0.)
    hist.GetYaxis().SetTitle('Category')
    hist.GetYaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTickLength(0.)
    hist.SetMinimum(-0.00001)
    hist.SetMaximum(100.)
    
def getRealSigma( hist ):
    sigma = 2.
    if hist.GetEntries() > 0 and hist.Integral()>0.000001:
        hist.Fit('gaus')
        fit = hist.GetFunction('gaus')
        sigma = fit.GetParameter(2)
    return sigma

def getAMS(s, b, breg=3.):
    b = b + breg
    val = 0.
    if b > 0.:
        val = (s + b)*np.log(1. + (s/b))
        val = 2*(val - s)
        val = np.sqrt(val)
    return val

def computeBkg( hist, effSigma ):
    bkgVal = 9999.
    if hist.GetEntries() > 0 and hist.Integral()>0.000001:
        hist.Fit('expo')
        fit = hist.GetFunction('expo')
        bkgVal = fit.Integral(125. - effSigma, 125. + effSigma)
    return bkgVal

def jetPtToClass( jets, pt ):
  return ( (jets>0).astype(int) + (4*((jets>1).astype(int))) + ((jets>0)*(pt>60.)).astype(int) + ((jets>0)*(pt>120.)).astype(int) + ((jets>0)*(pt>200.)).astype(int) )


def submitJob( jobDir, theCmd, params=None, model=None, dryRun=False ):
  outName = '%s/sub__'%jobDir
  if model:
    outName += model.replace('.model','')
  elif params: 
    params = params.split(',')
    for pair in params:
      pair = pair.split(':')
      outName += '%s_%s__'%(pair[0],pair[1])
    outName = outName[:-2]
  else:
    outName += 'None'
  outName += '.sh'
  with open('Tools/submitTemplate.sh') as inFile:
    with open(outName,'w') as outFile:
      for line in inFile.readlines():
        if '!CMD!' in line:
          line = line.replace('!CMD!','"%s"'%theCmd)
        elif '!MYDIR!' in line:
          line = line.replace('!MYDIR!',os.getcwd())
        elif '!NAME!' in line:
          line = line.replace('!NAME!',outName.replace('.sh',''))
        outFile.write(line)
  subCmd = 'qsub -q hep.q -o %s -e %s -l h_vmem=24G %s' %(outName.replace('.sh','.log'), outName.replace('.sh','.err'), outName) 
  #subCmd = 'qsub -q hep.q -o %s -e %s -l h_vmem=12G -l h_rt=8:0:0 %s' %(outName.replace('.sh','.log'), outName.replace('.sh','.err'), outName) 
  print
  print subCmd
  if not dryRun:
    os.system(subCmd)
