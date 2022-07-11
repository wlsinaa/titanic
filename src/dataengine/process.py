from dataengine import datareader
import pandas as pd
import sys, os
# base folder should be ""C:\Users\<USER folder>\OneDrive\Desktop\portfolio\kaggle\titanic"
p = os.path.abspath(".")
sys.path.insert(1, p)
import config
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
print(config.input_data_folder + config.input_data_train_filename)
class modifier(datareader.reader):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def insertname(self):

        self.df["first_name"] = self.df["Name"].apply(lambda x: x.split(',')[0].strip())
        self.df["call"] = self.df["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip())
        self.df["last_name"] = self.df["Name"].apply(lambda x: ''.join(x.split(',')[1].split('.')[1:])[1:])

        select = ["Mr","Miss","Mrs","Master","Dr"]
        self.df["call"] = self.df["call"].apply(lambda x: 'Others' if x not in select else x )
        self.df.drop("Name", axis= 1,inplace= True)
        self.df.drop("last_name", axis= 1,inplace= True)
        self.df.drop("first_name", axis= 1,inplace= True)

    def addticket(self):

        self.df["ticket_toget"] = self.df["Ticket"].apply(lambda x: len( self.df[self.df["Ticket"] == x]))
        #self.df["Ticket"].replace("LINE","00000", inplace= True)
        #self.df["ticket_number"] = self.df["Ticket"].apply(lambda x: int(x.split()[-1]))
        self.group("ticket_toget",5)
        self.df["ticket_number"] = self.df["Ticket"].apply(lambda x: x.split()[-1])
        self.df["ticket_header"] = self.df["ticket_number"].apply(lambda x:x[0])
        self.df.drop(["Ticket"],axis = 1, inplace= True)
        self.df.drop(["ticket_number"],axis = 1, inplace= True)

    def fillage(self,use_col: list):
        self.df["Pclass"] = self.df["Pclass"].astype(str)
        dummy_ = pd.get_dummies(self.df[use_col])
        target_df_null = dummy_[self.df["Age"].isnull()]
        target_df = dummy_[self.df["Age"].notnull()]
        self.df["Age"] = self.df["Age"].astype(float)

        train_X = target_df.to_numpy()
        train_Y = self.df[self.df["Age"].notnull()]["Age"].to_numpy()

        neigh = KNeighborsRegressor(n_neighbors=4)
        neigh.fit(train_X, train_Y)
        self.df.loc[self.df[self.df["Age"].isnull()].index, "Age"] = neigh.predict(target_df_null.to_numpy())
    def fillEmbarked(self):
        pass
    def fillCabin(self):

        self.df["Cabin"] = self.df["Cabin"].apply(lambda x: 0 if x is np.nan else 1 )

    def group(self,col: str,threshold: int):
        self.df[col] = self.df[col].apply(pd.to_numeric)
        self.df.loc[self.df[(self.df[col] >= threshold)].index,col] = threshold
        self.df[col] = self.df[col].astype(str)
        self.df.loc[(self.df[col] == str(threshold)),col] = str(threshold) + "+"
    
    def fullproccess(self,col: str):
        self.insertname()
        self.fillage(col)
        self.addticket()
        self.fillCabin()
if __name__ == "__main__":
    print(config.input_data_folder + config.input_data_train_filename)
    m = modifier(config.input_data_folder + config.input_data_train_filename)
    m.ingress()
    m.list_data()