import csv
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns



def process(path):
        df = pd.read_csv(path)
        df.info()
        df.dropna(how='any', inplace=True)
        df.hist(figsize=(13,13))
        plt.savefig('results/Histogram.png')
        plt.pause(5)
        plt.show(block=False)
        plt.close()
        fig=plt.figure(figsize=(10,5))
        sns.heatmap(df[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].corr(),annot=True)
        fig.savefig('results/Coorlationmatrix.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()
        subdivs = df['SUBDIVISION'].unique()
        num_of_subdivs = subdivs.size
        print('Total # of Subdivs: ' + str(num_of_subdivs))
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        df.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'].plot.bar(color='r',width=0.3,title='Subdivision wise Average Annual Rainfall', fontsize=20)
        plt.xticks(rotation = 90)
        plt.ylabel('Average Annual Rainfall (mm)')
        ax.title.set_fontsize(30)
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        fig.savefig('results/Annualrainfall.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()
        print(df.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[0,1,2]])
        print(df.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[33,34,35]])
        fig = plt.figure(figsize=(16,10))
        ax = fig.add_subplot(111)
        dfg = df.groupby('YEAR').sum()['ANNUAL']
        dfg.plot.line(title='Overall Rainfall in Each Year', fontsize=20)
        #df.groupby('YEAR').sum()['ANNUAL'].plot()
        #plt.xlim(0, 115)
        #plt.xticks(np.linspace(0,115,24,endpoint=True),np.linspace(1900,2015,24,endpoint=True).astype(int))
        #plt.xticks(np.linspace(1901,2015,24,endpoint=True))
        #plt.xticks(rotation = 90)
        plt.ylabel('Overall Rainfall (mm)')
        ax.title.set_fontsize(30)
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        fig.savefig('results/Overallrainfall.png')
        plt.pause(5)
        plt.show(block=False)
        plt.close()
        print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))
        print('Max: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))
        print('Mean: ' + str(dfg.mean()))
        months = df.columns[2:14]
        fig = plt.figure(figsize=(18,10))
        ax = fig.add_subplot(111)
        xlbls = df['SUBDIVISION'].unique()
        xlbls.sort()
        dfg = df.groupby('SUBDIVISION').mean()[months]
        dfg.plot.line(title='Overall Rainfall in Each Month of Year', ax=ax,fontsize=20)
        plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)
        plt.xticks(  rotation = 90)
        plt.ylabel('Rainfall (mm)')
        plt.legend(loc='upper right', fontsize = 'xx-large')
        ax.title.set_fontsize(30)
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        fig.savefig('results/Overallrainfall_each_month.png')
        plt.pause(5)
        plt.show(block=False)
        plt.close()

        dfg = dfg.mean(axis=0)
        print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))
        print('Max: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))
        print('Mean: ' + str(dfg.mean()))
        months = df.columns[2:14]
        
        df[['YEAR', 'Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']].groupby("YEAR").sum().plot(figsize=(10,10))
        #plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)
        plt.xticks(  rotation = 90)
        plt.ylabel('Rainfall (mm)')
        plt.legend(loc='upper right', fontsize = 'x-large')
        ax.title.set_fontsize(30)
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        fig.savefig('results/Seasonal_Variations_of_Rainfall_which_clearly.png')
        plt.pause(5)
        plt.show(block=False)
        plt.close()
        
    
 

      


