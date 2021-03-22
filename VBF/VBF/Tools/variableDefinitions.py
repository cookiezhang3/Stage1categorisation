## define sets of variables for use across classifiers

lumiDict = {'2016':35.9, '2017':41.5, '2018':59.7}

cosThetaStarVars = ['dipho_leadPhi', 'dipho_subleadPhi', 'gghMVA_leadDeltaPhi', 'gghMVA_subleadDeltaPhi',]

allVarsData = ['dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dipho_pt', 'dipho_leadEta', 'dipho_subleadEta',
               'dijet_leadEta', 'dijet_subleadEta', 'dijet_LeadJPt', 'dijet_SubJPt', 'dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'dipho_dijet_ptHjj', 'dijet_dipho_dphi_trunc',
               'cosThetaStar', 'dipho_cosphi', 'vtxprob', 'sigmarv', 'sigmawv', 'weight', 'dipho_mass', 'dijet_dphi', 'dijet_minDRJetPho', 'dijet_Zep']

allVarsGen  = allVarsData + ['HTXSstage1p2bin', 'dipho_leadPhi', 'dipho_subleadPhi', 'gghMVA_leadDeltaPhi', 'gghMVA_subleadDeltaPhi']

allVarsDataOld = ['dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dipho_pt', 'dipho_leadEta', 'dipho_subleadEta',
               'dijet_leadEta', 'dijet_subleadEta', 'dijet_LeadJPt', 'dijet_SubJPt', 'dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'dipho_dijet_ptHjj', 'dijet_dipho_dphi_trunc',
               'cosThetaStar', 'dipho_cosphi', 'vtxprob', 'sigmarv', 'sigmawv', 'weight', 'dipho_mass', 'dijet_dphi', 'dijet_minDRJetPho', 'dijet_Zep']

allVarsGenOld  = allVarsDataOld + ['HTXSstage1_1_cat']

diphoVars   = ['dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM',
               'dipho_leadEta', 'dipho_subleadEta', 
               'dipho_cosphi', 'vtxprob', 'sigmarv', 'sigmawv']

dijetVars   = ['dipho_lead_ptoM', 'dipho_sublead_ptoM', 
               'dijet_LeadJPt', 'dijet_SubJPt', 'dijet_abs_dEta', 'dijet_Mjj', 
               'dijet_centrality', 'dijet_dphi', 'dijet_minDRJetPho', 'dijet_dipho_dphi_trunc']

vhHadVars   = ['dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_leadEta', 'dijet_subleadEta', 
               'dijet_LeadJPt', 'dijet_SubJPt', 'dijet_Mjj', 'dijet_abs_dEta', 'cosThetaStar']
