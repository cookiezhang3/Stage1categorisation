#usual imports
import numpy as np
import pandas as pd
import xgboost as xg
import uproot as upr
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system, listdir
from Tools.addRowFunctions import truthVBF, vbfWeight, cosThetaStar

#DNN stuff
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential 
from keras.initializers import RandomNormal 
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import Nadam 
from keras.regularizers import l2  
from keras.callbacks import EarlyStopping 
from keras.utils import np_utils 
#import h5py

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Path to dataframe if it already exists')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
parser.add_option('--trainParams', default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
parser.add_option('--useDataDriven', action='store_true', default=False, help='Use the data-driven replacement for backgrounds with non-prompt photons')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')

#define variables to be used
from Tools.variableDefinitions import allVarsGen, dijetVars, lumiDict

#possible to add new variables here - have done some suggested ones as an example
newVars = ['gghMVA_leadPhi','gghMVA_leadJEn','gghMVA_subleadPhi','gghMVA_SubleadJEn','gghMVA_SubsubleadJPt','gghMVA_SubsubleadJEn','gghMVA_subsubleadPhi','gghMVA_subsubleadEta']
allVarsGen += newVars
dijetVars += newVars

#including the full selection
hdfQueryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_LeadJPt>40.) and (dijet_SubJPt>30.) and (dijet_Mjj>250.)'
queryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_leadIDMVA>-0.2) and (dipho_subleadIDMVA>-0.2) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_LeadJPt>40.) and (dijet_SubJPt>30.) and (dijet_Mjj>250.)'

if opts.useDataDriven:
  #define hdf input
  hdfDir = trainDir.replace('trees','hdfs')
  
  hdfList = []
  if hdfDir.count('all'):
    for year in lumiDict.keys():
      tempHdfFrame = pd.read_hdf('%s/VBF_with_DataDriven_%s_MERGEDFF_NORM_NEW.h5'%(hdfDir,year)).query(hdfQueryString)
      tempHdfFrame = tempHdfFrame[tempHdfFrame['sample']=='QCD']
      tempHdfFrame.loc[:, 'weight'] = tempHdfFrame['weight'] * lumiDict[year]
      hdfList.append(tempHdfFrame)
    hdfFrame = pd.concat(hdfList, sort=False)
  else:
    hdfFrame = pd.read_hdf('%s/VBF_with_DataDriven_%s_MERGEDFF_NORM_NEW.h5'%(hdfDir,hdfDir.split('/')[-2]) ).query(hdfQueryString)
    hdfFrame = hdfFrame[hdfFrame['sample']=='QCD']
  
  hdfFrame['proc'] = 'datadriven'

#define input files
procFileMap = {'ggh':'powheg_ggH.root', 'vbf':'powheg_VBF.root', 'vh':'powheg_VH.root',
               'dipho':'Dipho.root'}
theProcs = procFileMap.keys()
signals     = ['ggh','vbf','vh']
backgrounds = ['dipho']
if not opts.useDataDriven:
  procFileMap['gjet_anyfake'] = 'GJet.root'
  backgrounds.append('gjet_anyfake')

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
  if opts.useDataDriven: 
    trainList.append(hdfFrame)
  trainTotal = pd.concat(trainList, sort=False)
  del trainList
  del tempFrame
  if opts.useDataDriven: 
    del hdfFrame
  print 'created total frame'

  #add the target variable and the equalised weight
  trainTotal['truthVBF'] = trainTotal.apply(truthVBF,axis=1)
  trainTotal = trainTotal[trainTotal.truthVBF>-0.5]
  vbfSumW = np.sum(trainTotal[trainTotal.truthVBF==2]['weight'].values)
  gghSumW = np.sum(trainTotal[trainTotal.truthVBF==1]['weight'].values)
  bkgSumW = np.sum(trainTotal[trainTotal.truthVBF==0]['weight'].values)
  trainTotal['vbfWeight'] = trainTotal.apply(vbfWeight, axis=1, args=[vbfSumW,gghSumW,bkgSumW])
  trainTotal['dijet_centrality']=np.exp(-4.*((trainTotal.dijet_Zep/trainTotal.dijet_abs_dEta)**2))

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/vbfDataDriven.pkl'%frameDir)
  print 'frame saved as %s/vbfDataDriven.pkl'%frameDir

#read in dataframe if above steps done before
else:
  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
  print 'Successfully loaded the dataframe'

#set up train set and randomise the inputs
trainFrac = 0.8
theShape = trainTotal.shape[0]
theShuffle = np.random.permutation(theShape)
trainLimit = int(theShape*trainFrac)

#define the values needed for training as numpy arrays
vbfX  = trainTotal[dijetVars].values
vbfY  = trainTotal['truthVBF'].values
vbfTW = trainTotal['vbfWeight'].values
vbfFW = trainTotal['weight'].values
vbfM  = trainTotal['dipho_mass'].values

#do the shuffle
vbfX  = vbfX[theShuffle]
vbfY  = vbfY[theShuffle]
vbfTW = vbfTW[theShuffle]
vbfFW = vbfFW[theShuffle]
vbfM  = vbfM[theShuffle]

#split into train and test
vbfTrainX,  vbfTestX  = np.split( vbfX,  [trainLimit] )
vbfTrainY,  vbfTestY  = np.split( vbfY,  [trainLimit] )
vbfTrainTW, vbfTestTW = np.split( vbfTW, [trainLimit] )
vbfTrainFW, vbfTestFW = np.split( vbfFW, [trainLimit] )
vbfTrainM,  vbfTestM  = np.split( vbfM,  [trainLimit] )

# apply Z scaling (mean -> 0, std -> 1)
X_scaler = StandardScaler()
X_scaler.fit(vbfTrainX)
vbfTrainX = X_scaler.transform(vbfTrainX)
vbfTestX = X_scaler.transform(vbfTestX)

#convert y column with one-hot-encoding (keras only accepts this form)
numOutputs = 3 #three classes
numInputs = vbfTrainX.shape[1]
vbfTrainYOH = np_utils.to_categorical(vbfTrainY, num_classes=numOutputs)
vbfTestYOH  = np_utils.to_categorical(vbfTestY, num_classes=numOutputs)

#build the model
numLayers_rg = [2,3,4]
nodesPerHiddenLayer_rg= [400]

for n in numLayers_rg:
  for k in nodesPerHiddenLayer_rg:
        print 'n ',n
        print 'k ', k
        numLayers = n
        nodesPerHiddenLayer = k
        dropout = 0.2
        batchSize = 64

        model = Sequential()
#        for i, nodes in enumerate([nodesPerHiddenLayer] * numLayers):
        for i, nodes in enumerate([k] * n):
          if i == 0: #first layer     
            model.add(                
            Dense(                    
                     nodes,            
                     kernel_initializer='glorot_normal',
                     activation='relu',
                     kernel_regularizer=l2(1e-5),
                     input_dim=numInputs
                )                 
            )                         
            model.add(Dropout(dropout))
          else: #hidden layers        
            model.add(                
            Dense(                    
                    nodes,            
                    kernel_initializer='glorot_normal',
                    activation='relu',
                    kernel_regularizer=l2(1e-5),
                    )                 
            )                         
            model.add(Dropout(dropout))
                              
#final layer                  
        model.add(                    
                Dense(                
                    numOutputs,      
                    kernel_initializer=RandomNormal(),
                    activation='softmax'
                    )                 
                )                     
                              
        model.compile(                
                loss='categorical_crossentropy',
                optimizer=Nadam(),    
                metrics=['accuracy']  
        )                             
        callbacks = []                
        callbacks.append(EarlyStopping(patience=50))
        model.summary()               
                              
                              
#Fit the model on data
        print('Fitting on the training data')
        history = model.fit(          
            vbfTrainX,           
            vbfTrainYOH,           
            sample_weight=vbfTrainTW, 
            #validation_data=(vbfValidX,vbfValidY, vbfValidTW),
            batch_size=batchSize,     
            epochs=50,              
            shuffle=True
            #callbacks=callbacks # add function to print stuff out there
            )                         
        print('Done') 

        print('Fitting on the testing data')
        history = model.fit(
            vbfTestX,
            vbfTestYOH,
            sample_weight=vbfTestTW,
            #validation_data=(vbfValidX,vbfValidY, vbfValidTW),
            batch_size=batchSize,
            epochs=50,     
            shuffle=True
            #callbacks=callbacks # add function to print stuff out there
            )
        print('Done')

        yProbTrain = model.predict(vbfTrainX).reshape(vbfTrainY.shape[0],numOutputs)
        yProbTest = model.predict(vbfTestX).reshape(vbfTestY.shape[0],numOutputs)
        vbfTrainTrueY = np.where(vbfTrainY==2,1,0)
        vbfTestTrueY = np.where(vbfTestY==2, 1, 0)
        print 'area under roc curve for trianing set, VBF v.s. Rest = %1.3f'%( roc_auc_score(vbfTrainTrueY, yProbTrain[:,2], sample_weight=vbfTrainFW) )
        print 'area under roc curve for test set, VBF v.s. Rest = %1.3f'%( roc_auc_score(vbfTestTrueY, yProbTest[:,2], sample_weight=vbfTestFW) )

#make plots
plotDir = trainDir.replace('trees','plots_DNN')
#plotDir = '%s/%s'%(paramExt)
if not path.isdir(plotDir): 
  system('mkdir -p %s'%plotDir)

#roc_curve vbf
fpr_tr, tpr_tr, thresholds_tr = roc_curve(vbfTrainTrueY, yProbTrain[:,2], sample_weight=vbfTrainFW )
fpr_tst, tpr_tst, thresholds_tst = roc_curve(vbfTestTrueY, yProbTest[:,2], sample_weight=vbfTestFW) 
plt.figure(1)
plt.plot(fpr_tr,tpr_tr,label = r'training set ROC curve (area = %1.3f)'%(roc_auc_score(vbfTrainTrueY, yProbTrain[:,2], sample_weight=vbfTrainFW)) )
plt.plot(fpr_tst,tpr_tst,label = r'test set ROC curve (area = %1.3f)'%( roc_auc_score(vbfTestTrueY, yProbTest[:,2], sample_weight=vbfTestFW)) )
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve original')
plt.legend(loc='best',prop={'size': 12})
plt.savefig('%s/ROC_vbf_original.pdf'%plotDir)
print 'saved as %s/ROC_vbf_original.pdf'%plotDir

