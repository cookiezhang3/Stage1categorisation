#usual imports
import numpy as np
import pandas as pd
import xgboost as xg
import uproot as upr
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system, listdir
from Tools.addRowFunctions import truthVBF, vbfWeight, cosThetaStar

from catOptim import CatOptim

#DNN stuff
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential 
from keras.initializers import RandomNormal 
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import Nadam 
from keras.regularizers import l2  
from keras.callbacks import EarlyStopping 
from keras.utils import np_utils 
import h5py

def main(opts):

#setup global variables
  trainDir = opts.trainDir
  if trainDir.endswith('/'): trainDir = trainDir[:-1]
  frameDir = trainDir.replace('trees','frames')
  if opts.trainParams: opts.trainParams = opts.trainParams.split(',')

#define variables to be used
  from Tools.variableDefinitions import allVarsGen, dijetVars, lumiDict,allVarsData

#possible to add new variables here - have done some suggested ones as an example
  newVars = ['gghMVA_leadPhi','gghMVA_leadJEn','gghMVA_subleadPhi','gghMVA_SubleadJEn','gghMVA_SubsubleadJPt','gghMVA_SubsubleadJEn','gghMVA_subsubleadPhi','gghMVA_subsubleadEta']
  allVarsGen += newVars
  dijetVars += newVars
  allVarsData += newVars

#including the full selection
  hdfQueryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_LeadJPt>40.) and (dijet_SubJPt>30.) and (dijet_Mjj>250.)'
  queryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_leadIDMVA>-0.2) and (dipho_subleadIDMVA>-0.2) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_LeadJPt>40.) and (dijet_SubJPt>30.) and (dijet_Mjj>250.)'

#get trees from files, put them in data frames
  procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'vh':'VH.root'}
  theProcs = procFileMap.keys()
  dataFileMap = {'Data':'Data.root'}
  signals = ['ggh','vbf','vh']

#either get existing data frame or create it
  trainTotal = None
  if not opts.dataFrame:
    trainList = []
  #get trees from files, put them in data frames
    if not 'all' in trainDir:
      for proc,fn in procFileMap.iteritems():
        print 'reading in tree from file %s'%fn
        trainFile   = upr.open('%s/%s'%(trainDir,fn))
        if proc in signals: trainTree = trainFile['vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc]
        elif proc in backgrounds: trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
        else: raise Exception('Error did not recognise process %s !'%proc)
        tempFrame = trainTree.pandas.df(allVarsGen).query(queryString)
        tempFrame['proc'] = proc
        trainList.append(tempFrame)
    else:
      for year in lumiDict.keys():
        for proc,fn in procFileMap.iteritems():
          thisDir = trainDir.replace('all',year)
          print 'reading in tree from file %s'%fn
          trainFile   = upr.open('%s/%s'%(thisDir,fn))
          if proc in signals: trainTree = trainFile['vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc]
          elif proc in backgrounds: trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
          else: raise Exception('Error did not recognise process %s !'%proc)
          tempFrame = trainTree.pandas.df(allVarsGen).query(queryString)
          tempFrame['proc'] = proc
          tempFrame.loc[:, 'weight'] = tempFrame['weight'] * lumiDict[year]
          trainList.append(tempFrame)
    print 'got trees and applied selections'

  #create one total frame
    trainTotal = pd.concat(trainList, sort=False)
    del trainList
    del tempFrame
    print 'created total frame'

  #add the target variable and the equalised weight
    trainTotal['truthVBF'] = trainTotal.apply(truthVBF,axis=1)
    trainTotal = trainTotal[trainTotal.truthVBF>-0.5]
    vbfSumW = np.sum(trainTotal[trainTotal.truthVBF==2]['weight'].values)
    gghSumW = np.sum(trainTotal[trainTotal.truthVBF==1]['weight'].values)
    bkgSumW = np.sum(trainTotal[trainTotal.truthVBF==0]['weight'].values)
    trainTotal['vbfWeight'] = trainTotal.apply(vbfWeight, axis=1, args=[vbfSumW,gghSumW,bkgSumW])
    trainTotal['dijet_centrality']=np.exp(-4.*((trainTotal.dijet_Zep/trainTotal.dijet_abs_dEta)**2))

    dataTotal = None
    if not opts.dataFrame:
      dataList = []
    #get the trees, turn them into arrays
      if not 'all' in trainDir:
        for proc,fn in dataFileMap.iteritems():
          dataFile = upr.open('%s/%s'%(trainDir,fn))
          dataTree = dataFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
          tempData = dataTree.pandas.df(allVarsData).query(queryString)
          tempData['proc'] = proc
          dataList.append(tempData)
      else:
        for year in lumiDict.keys():
          for proc,fn in dataFileMap.iteritems():
            thisDir = trainDir.replace('all',year)
            dataFile = upr.open('%s/%s'%(thisDir,fn))
            dataTree = dataFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
            tempData = dataTree.pandas.df(allVarsData).query(queryString)
            tempData['proc'] = proc
            dataList.append(tempData)
      print 'got trees'
    
      dataTotal = pd.concat(dataList)
      
  #add needed variables
    dataTotal['dijet_centrality']=np.exp(-4.*((dataTotal.dijet_Zep/dataTotal.dijet_abs_dEta)**2))
    #save as a pickle file if copied original root files into own dir
    #if not path.isdir(frameDir): 
    #  system('mkdir -p %s'%frameDir)
    #dataTotal.to_pickle('%s/vbfDataTotal.pkl'%frameDir)
    #print 'frame saved as %s/vbfDataTotal.pkl'%frameDir

#define the variables used as input to the classifier
    vbfX  = trainTotal[dijetVars].values
    vbfY  = trainTotal['truthVBF'].values
    vbfP  = trainTotal['HTXSstage1p2bin'].values
    vbfM  = trainTotal['dipho_mass'].values
    vbfFW = trainTotal['weight'].values
    print 'DEBUG {}'.format(vbfFW)
    vbfJ  = trainTotal['dijet_Mjj'].values
    #vbfH  = trainTotal['dipho_pt'].values
    #vbfHJ = trainTotal['dipho_dijet_ptHjj'].values
    #vbfIL = trainTotal['dipho_leadIDMVA'].values
    #vbfIS = trainTotal['dipho_subleadIDMVA'].values
    
    dataX  = dataTotal[dijetVars].values
    dataM  = dataTotal['dipho_mass'].values
    dataFW = np.ones(dataM.shape[0])
    dataJ  = dataTotal['dijet_Mjj'].values
    #dataH  = dataTotal['dipho_pt'].values
    #dataHJ = dataTotal['dipho_dijet_ptHjj'].values
    #dataIL = dataTotal['dipho_leadIDMVA'].values
    #dataIS = dataTotal['dipho_subleadIDMVA'].values

# apply Z scaling (mean -> 0, std -> 1)
    X_scaler = StandardScaler()
    X_scaler.fit(vbfX)
    vbfX = X_scaler.transform(vbfX)
  #  X_scaler.fit(dataX)
    dataX = X_scaler.transform(dataX)

#convert y column with one-hot-encoding
    numOutputs = 3
#   vbfYOH = np_utils.to_categorical(vbfY, num_classes=numOutputs)
#   dataYOH = np_utils.to_categorical(dataY, num_classes=numOutputs)

#load and evaluate VBF three class
    modelDir = trainDir.replace('trees','models')
  #load model
    model = keras.models.load_model('%s/%s'%(modelDir,'DNN.model'))
    vbfPredictions = model.predict(vbfX).reshape(vbfM.shape[0],numOutputs)
    vbfV = vbfPredictions[:,2]
    vbfG = vbfPredictions[:,1]
    print 'some values of the VBF probability %s'%vbfPredictions[0:10,2]
    print 'some values of the ggH probability %s'%vbfPredictions[0:10,1]
    print 'some values of the bkg probability %s'%vbfPredictions[0:10,0]

    dataPredictions = model.predict(dataX).reshape(dataM.shape[0],numOutputs)
    dataV  = dataPredictions[:,2]
    dataG  = dataPredictions[:,1]
    print 'some values of the VBF probability %s'%dataPredictions[0:10,2]
    print 'some values of the ggH probability %s'%dataPredictions[0:10,1]
    print 'some values of the bkg probability %s'%dataPredictions[0:10,0]

#set up optimiser ranges and no. categories to test
    ranges    = [ [0.3,1.], [0,0.7] ]
    names     = ['VBFprob', 'ggHprob']
    print_str = ''
    cats = [1,2,3]

  #do optimisation for each category. Keeping one inclusive category for now
    sigWeights = vbfFW * (vbfY==2) * (vbfJ>350)
    bkgWeights = dataFW *  (dataJ>350)

    for nCats in cats:
        optimiser = CatOptim(sigWeights, vbfM, [vbfV, vbfG], bkgWeights, dataM, [dataV, dataG], nCats, ranges, names)
        optimiser.setOpposite('ggHprob')
        if not 'all' in trainDir: optimiser.optimise(opts.intLumi, opts.nIterations)
        else: optimiser.optimise(1, opts.nIterations) #set lumi to 1 as already scaled when loading in
        print_str += 'Results for {} categories : \n'.format(nCats)
        print_str += optimiser.getPrintableResult()
    print '\n {}'.format(print_str)



if __name__ == "__main__":

  from optparse import OptionParser
  parser = OptionParser()
  parser.add_option('-t','--trainDir', help='Directory for input files')
  parser.add_option('-d','--dataFrame', default=None, help='Name of dataframe if it already exists')
  parser.add_option('-s','--signalFrame', default=None, help='Name of signal dataframe if it already exists')
  parser.add_option('-n','--nIterations', default=3000, help='Number of iterations to run for random significance optimisation')
  parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
  parser.add_option('--useDataDriven', action='store_true', default=False, help='Use the data-driven replacement for backgrounds with non-prompt photons')
  parser.add_option('--trainParams', default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
  (opts,args)=parser.parse_args()
  main(opts)
