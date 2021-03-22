import pandas as pd
import numpy as np
import uproot as upr
import matplotlib
import matplotlib as mpl
mpl.use('Agg') #solve displays environment in windows
import matplotlib.pyplot as plt
import argparse
from variableDefinitions import allVarsGen as all_vars_gen, dijetVars as dijet_vars
from os import path, system

plt.rcParams.update({'font.size':'8'})

def corr_matrix(arr1, arr2, weights):
    '''
    Calculate pearson correlation coefficient for two vectors (features)
    '''

    m1 = np.average(arr1,weights=weights)*np.ones_like(arr1)
    m2 = np.average(arr2,weights=weights)*np.ones_like(arr2)
    cov_11 = float((weights*(arr1-m1)**2).sum()/weights.sum())
    cov_22 = float((weights*(arr2-m2)**2).sum()/weights.sum())
    cov_12 = float((weights*(arr1-m1)*(arr2-m2)).sum()/weights.sum())
    return cov_12/np.sqrt(cov_11*cov_22)


def plot_numbers(ax,mat):
    '''
    Plot correlation coefficient as text labels
    '''
    for i in xrange(mat.shape[0]):
        for j in xrange(mat.shape[1]):
            c = mat[j,i]
            if np.abs(c)>=1:
                ax.text(i,j,'{:.0f}'.format(c),fontdict={'size': 8},va='center',ha='center')

def main(options):
    labels_to_plot = [var.replace('_',' ') for var in dijet_vars]
    vbf_cuts = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_leadIDMVA>-0.2) and (dipho_subleadIDMVA>-0.2) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_LeadJPt>40.) and (dijet_SubJPt>30.) and (dijet_Mjj>250.)'

    #get root files, convert into dataframes
    sig_file = upr.open(options.sigPath)
    sig_tree = sig_file[options.sigTree]
    sig_df   = sig_tree.pandas.df(all_vars_gen).query(vbf_cuts)
    sig_df['dijet_centrality']=np.exp(-4.*((sig_df.dijet_Zep/sig_df.dijet_abs_dEta)**2))

    #get correlations
    sig_corrs = np.array([ [100*corr_matrix(sig_df[var1].values, sig_df[var2].values, sig_df['weight']) for var2 in dijet_vars] for var1 in dijet_vars])
    
    #plot sig correlations
    plt.set_cmap('bwr')
    fig = plt.figure()
    axes= plt.gca()
    mpl.rcParams['figure.figsize'] = [10.0, 8.0]

    mat = axes.matshow(sig_corrs, vmin=-100, vmax=100)
    plot_numbers(axes, sig_corrs)
    axes.set_yticks(np.arange(len(dijet_vars)))
    axes.set_xticks(np.arange(len(dijet_vars)))
    axes.set_xticklabels(labels_to_plot,rotation='vertical')
    axes.set_yticklabels(labels_to_plot)
    axes.xaxis.tick_top()
    cbar = fig.colorbar(mat)
    cbar.set_label(r'Correlation (%)')

    if not path.isdir('/plots'):
        print 'making directory: {}'.format('plots/')
        system('mkdir -p plots/')

    fig.savefig('plots/vbf_sig_crl_matrix.pdf',dpi=300, bbox_inches = "tight") #add "tight" to show full xticks
    print 'saving fig: {}'.format('plots/vbf_sig_crl_matrix.pdf')
    plt.close()

    #NOTE: might be interesting to also look the correlation for the background procs, and/or other signal procs (ggH)

if __name__ == '__main__':
    parser        = argparse.ArgumentParser()
    required_args =  parser.add_argument_group('Required Arguments')
    required_args.add_argument('-S', '--sigPath', action='store', type=str, required=True)
    required_args.add_argument('-s', '--sigTree', action='store', type=str, required=True)
    options=parser.parse_args()
    main(options)


