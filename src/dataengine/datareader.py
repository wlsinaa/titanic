# Data reading for the main source of titanic information
# Author : Walter
# Date: 19/6/2022

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import sys, os
# base folder should be ""C:\Users\<USER folder>\OneDrive\Desktop\portfolio\kaggle\titanic"
p = os.path.abspath(".")
sys.path.insert(1, p)
import config
class reader():
    
    def __init__(self, path:str):
        self.path = path
    # read csv file and inpurt the data
    def ingress(self):

        try:
            self.df = pd.read_csv(self.path)
        except OSError as err:
            print("OS error: {0}".format(err))
    
    def list_data(self):
        print(self.df)

    def getnarow(self):

        return self.df[self.df.isna().any(axis=1)]

    def countna(self):
        return self.df.isnull().sum()

    def getmissingheapmap(self):
        sns.heatmap(self.df.isnull(), cbar = False)
        plt.show()
    def getsurrateby(self, col):
        return self.df.groupby([col])['Survived'].mean()
if __name__ == "__main__":

    # Test the function
    READER = reader(config.input_data_folder + config.input_data_train_filename)
    READER.ingress()
    READER.list_data()