# main module for data prediction
# Author: Walter Sin
# Date: 7/7/2022
from dataengine import datareader
from dataengine import process
from sklearn.neighbors import KNeighborsRegressor
import config
import numpy as np
import pandas as pd
class predictor():
    def __init__(self):
        pass

if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    READER = datareader.reader(config.input_data_folder + config.input_data_train_filename)
    READER.ingress()
    READER.list_data()
    missing_index = np.array(READER.getnarow().index)
    clean_index = np.delete(np.array(READER.df.index),missing_index)
    clean_df = READER.df.iloc[np.delete(READER.df.index,missing_index)]
    miss_df = READER.df.iloc[missing_index]
    #READER.getmissingheapmap()
    print("\n Given known Cabin known")
    print(READER.df[READER.df["Cabin"].notnull()]['Survived'].value_counts())
    stat = READER.df[READER.df["Cabin"].notnull()]['Survived'].value_counts()
    print("Survived rate:" ,stat[1]/stat.sum())
    print("\n Overall")
    print(READER.df['Survived'].value_counts())
    stat = READER.df['Survived'].value_counts()
    print("Survived rate:" ,stat[1]/stat.sum())
    print("\n Given known Cabin unknown")
    print(READER.df[READER.df["Cabin"].isnull()]['Survived'].value_counts())
    stat = READER.df[READER.df["Cabin"].isnull()]['Survived'].value_counts()
    print("Survived rate:" ,stat[1]/stat.sum())

    print("Cabin Count:")
    print(READER.df["Cabin"].value_counts())

    # count the survial rate by the columns respectively
    print(READER.countna())
    print('Survival rate group by  {} '.format("Pclass"))
    print(READER.getsurrateby("Pclass"))
    print('Survival rate group by  {} '.format("Sex"))
    print(READER.getsurrateby("Sex"))
    print('Survival rate group by  {} '.format("SibSp"))
    print(READER.getsurrateby("SibSp"))
    print('Survival rate group by  {} '.format("Parch"))
    print(READER.getsurrateby("Parch"))

                                                            # tackle the missing value of Embarked 
    #print(READER.df[READER.df["Embarked"].isnull()])

    """
         PassengerId  Survived  Pclass                                       Name            Sex   Age     SibSp  Parch Ticket Fare  Cabin   Embarked
            61            62         1       1                        Icard, Miss. Amelie  female  38.0      0      0  113572  80.0   B28      NaN
            829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)  female  62.0      0      0  113572  80.0   B28      NaN

    """
    #Embarked_not_null = READER.df[READER.df["Embarked"].notnull()]
    #print(Embarked_not_null[(Embarked_not_null["Survived"] == 0)  ]["Embarked"].value_counts())
    """
    S    427
    C     75
    Q     47
    """
    #print(Embarked_not_null[(Embarked_not_null["Survived"] == 1)  ]["Embarked"].value_counts())
    """
    S    217
    C     93
    Q     30
    """
    #print(Embarked_not_null[(Embarked_not_null["Survived"] == 0) & (Embarked_not_null["Pclass"] == 1)]["Embarked"].value_counts())
    """
    S    53
    C    26
    Q     1
    """
    #print(Embarked_not_null[(Embarked_not_null["Survived"] == 0) & (Embarked_not_null["Pclass"] == 2)]["Embarked"].value_counts())
    """
    S    88
    C     8
    Q     1
    """
    #print(Embarked_not_null[(Embarked_not_null["Survived"] == 0) & (Embarked_not_null["Pclass"] == 3)]["Embarked"].value_counts())
    """
    S    286
    Q     45
    C     41
    """
    #print(Embarked_not_null[(Embarked_not_null["Survived"] == 1) & (Embarked_not_null["Pclass"] == 1)]["Embarked"].value_counts())
    """
    S    74
    C    59
    Q     1
    """
    #print(Embarked_not_null[(Embarked_not_null["Survived"] == 1) & (Embarked_not_null["Pclass"] == 2)]["Embarked"].value_counts())
    """
    S    76
    C     9
    Q     2
    """
    #print(Embarked_not_null[(Embarked_not_null["Survived"] == 1) & (Embarked_not_null["Pclass"] == 3)]["Embarked"].value_counts())
    """
    S    67
    Q    27
    C    25
    """

    # that means the distribution of "Embarked" affected by Survival = 0
    # Deeper sight into Pclass
    # in both survival = 0 and 1, the distribution of Pclass=3 is significantly different from another two
    # in Pclass = 1, it is match with the phenomanon that more probability on Embarked = Q comparing to Pclass = 2
    # in Pclass = 3, it is match with the phenomanon that more probability on Embarked = C in both Survived = 0, 1
    #target_df = Embarked_not_null[Embarked_not_null["Pclass"] == 1] # process df for t
    #print(target_df[target_df["Sex"] == "female"]['Embarked'].value_counts())
    """
    S    48
    C    43
    Q     1
    """
    #print(target_df[target_df["Sex"] == "male"]['Embarked'].value_counts())
    """
    S    79
    C    42
    Q     1
    """
    # no significant difference, use the original target
    #target_df = target_df[target_df["Sex"] == "female"]
    #colors = ["red","blue","yellow"]
    #fig,ax = plt.subplots()
    #ax2=ax.twinx()
    #for gro, c in zip(target_df["Embarked"].unique(),colors):
    #    x = target_df[(target_df['Embarked'] == gro)]['Age'].to_numpy()
    #    y = target_df[(target_df['Embarked'] == gro)]['Fare'].to_numpy()
    #    ax.scatter(x, y, color = c,label=gro)
    #    ax.legend()
    #x = target_df[(target_df['Embarked'] == "S")]['Age'].to_numpy()
    #ax2.hist(x, bins=10, edgecolor='red',facecolor="None")
    #x = target_df[(target_df['Embarked'] == "C")]['Age'].to_numpy()
    #ax2.hist(x, bins=10, edgecolor='black',facecolor="None")
    # no obvious pattern for Age and Fare
    # sibSP and Parch 
    print("----------------------------------")
    #print(target_df["Embarked"].value_counts())
    #print(target_df[(target_df["SibSp"] == 0) & (target_df["Parch"] == 0)]["Embarked"].value_counts())
    #print(target_df[target_df["SibSp"] == 1 & (target_df["Parch"] == 0)]["Embarked"].value_counts())
    #print(target_df[target_df["Parch"] == 0 & (target_df["SibSp"] == 1)]["Embarked"].value_counts())
    #print(target_df[target_df["Parch"] == 1 & (target_df["SibSp"] == 1)]["Embarked"].value_counts())
    
    # no signifcant difference
    #ch_index = list(READER.df[READER.df["Embarked"].isnull()].index)
    READER.df["Embarked"].fillna("S",inplace= True)

    # tackle Age missing value
    target_df_null = READER.df[READER.df["Age"].isnull()]
    target_df = READER.df[READER.df["Age"].notnull()]
    print(target_df.groupby("Embarked")["Age"].describe())
    """
            count       mean        std   min    25%   50%   75%   max
    Embarked
    C         130.0  30.814769  15.434860  0.42  21.25  29.0  40.0  71.0
    Q          28.0  28.089286  16.915396  2.00  17.50  27.0  34.5  70.5
    S         556.0  29.519335  14.189608  0.67  21.00  28.0  38.0  80.0    
    """
    print(target_df.groupby("SibSp")["Age"].describe())
    """
        count       mean        std   min    25%   50%    75%   max
    SibSp
    0      471.0  31.397558  13.647767  0.42  22.00  29.0  39.00  80.0
    1      183.0  30.089727  14.645033  0.67  20.00  30.0  39.00  70.0
    2       25.0  22.620000  14.679230  0.75  16.00  23.0  28.00  53.0
    3       12.0  13.916667  11.317391  2.00   3.75   9.5  23.25  33.0
    4       18.0   7.055556   4.880601  1.00   3.25   6.5   9.00  17.0
    5        5.0  10.200000   5.805170  1.00   9.00  11.0  14.00  16.0
    """
    print(target_df.groupby("Parch")["Age"].describe())
    """
        count       mean        std    min    25%   50%    75%   max
    Parch
    0      521.0  32.178503  12.570448   5.00  22.00  30.0  39.00  80.0
    1      110.0  24.422000  18.283117   0.42   6.25  23.0  39.00  70.0
    2       68.0  17.216912  13.193924   0.83   5.75  16.5  25.00  58.0
    3        5.0  33.200000  16.709279  16.00  24.00  24.0  48.00  54.0
    4        4.0  44.500000  14.617341  29.00  37.25  42.5  49.75  64.0
    5        5.0  39.200000   1.095445  38.00  39.00  39.0  39.00  41.0
    6        1.0  43.000000        NaN  43.00  43.00  43.0  43.00  43.0
            count       mean        std   min   25%   50%   75%   max
    """
    print(target_df.groupby("Sex")["Age"].describe())
    """
    female  261.0  27.915709  14.110146  0.75  18.0  27.0  37.0  63.0
    male    453.0  30.726645  14.678201  0.42  21.0  29.0  39.0  80.0    
    """
    print(target_df.groupby("Pclass")["Age"].describe())
    # no different
    #pro = process.modifier(target_df.copy())
    #plt.scatter(pro.df["Fare"].to_numpy(),pro.df["Age"].to_numpy())

    pro = process.modifier(READER.df.copy())
    pro.group("SibSp", 3)
    pro.group("Parch", 3)
    pro.fillage(["Pclass","SibSp","Parch","Sex","Embarked"])
    
    # extract ticket relation
    pro.df["ticket_toget"] = pro.df["Ticket"].apply(lambda x: len( pro.df[pro.df["Ticket"] == x]))
    pro.df["Ticket"].replace("LINE","00000", inplace= True)
    pro.df["ticket_number"] = pro.df["Ticket"].apply(lambda x: x.split()[-1])

    pro.df["ticket_header"] = pro.df["ticket_number"].apply(lambda x:x[0])
    #print( "\n",pro.df[["ticket_header", 'Pclass']].groupby(['ticket_header']).value_counts().unstack('Pclass', fill_value=0).reset_index() )

    """
    Pclass ticket_header    1    2    3
    0                  0    0    0    4
    1                  1  192   22   17
    2                  2    4  136   90
    3                  3   14   23  330
    4                  4    0    0   15
    5                  5    4    1    4
    6                  6    2    0   12
    7                  7    0    2   13
    8                  8    0    0    3
    9                  9    0    0    3    
    """
    #print( "\n",pro.df[["ticket_header", 'ticket_toget']].groupby(['ticket_header']).value_counts().unstack('ticket_toget', fill_value=0).reset_index() )
    """
    ticket_toget ticket_header    1   2   3   4  5   6  7
    0                        0    0   0   0   4  0   0  0
    1                        1  111  64  24  20  5   0  7
    2                        2  133  62  18   4  0   6  7
    3                        3  264  50  21   8  5  12  7
    4                        4    9   2   0   4  0   0  0
    5                        5    5   4   0   0  0   0  0
    6                        6    8   2   0   4  0   0  0
    7                        7   13   2   0   0  0   0  0
    8                        8    3   0   0   0  0   0  0
    9                        9    1   2   0   0  0   0  0
    """
    #print(pro.df.groupby("ticket_header")["Fare"].mean())
    """
    ticket_header
    0     0.000000
    1    73.869768
    2    21.611234
    3    15.314336
    4    16.477513
    5    32.261578
    6    21.094350
    7     8.920280
    8    10.431933
    9    13.716667    
    """
    #print(pro.df.groupby("ticket_header")["Survived"].mean())

    """
    0    0.250000
    1    0.606061
    2    0.408696
    3    0.258856
    4    0.133333
    5    0.222222
    6    0.071429
    7    0.266667
    8    0.000000
    9    1.000000    
    """

    pro.df.drop(["Ticket"],axis = 1, inplace= True)
    pro.df.drop(["ticket_number"],axis = 1, inplace= True)
    #sns.displot(data=pro.df[(pro.df["ticket_number"] < 200000) & (pro.df["Sex"] == "male")],bins=20, x="ticket_number", hue="Survived",kde=True)
    #plt.show()
    
    #print(pro.df.groupby("Sex")["ticket_number"].mean())
    """
    female    229671.22293
    male      333623.12305
    """
    #print(pro.df.groupby("Survived")["ticket_number"].mean())
    """
    0    346495.315118
    1    217518.649123
    """
    #print(pro.df["ticket_toget"].value_counts())
    """
    1    547
    2    188
    3     63
    4     44
    7     21
    6     18
    5     10
    """
    #print(pro.df.groupby("ticket_toget")["Survived"].mean())
    """
    1    0.297989
    2    0.574468
    3    0.698413
    4    0.500000
    5    0.000000
    6    0.000000
    7    0.238095    
    """
    #print(pro.df["ticket_toget"].value_counts())
    """
    1    547
    2    188
    3     63
    4     44
    7     21
    6     18
    5     10
    """
    # group ticket_toget
    pro.group("ticket_toget",5)

    # add name information

    # split name and get split name in Name
    pro.df["first_name"] = pro.df["Name"].apply(lambda x: x.split(',')[0].strip())
    pro.df["call"] = pro.df["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    pro.df["last_name"] = pro.df["Name"].apply(lambda x: ''.join(x.split(',')[1].split('.')[1:])[1:])
    
    """
    John                   9
    James                  7
    William                6
    Mary                   6
    Bertha                 4
                        ..
    Nora A                 1
    Howard Hugh "Harry"    1
    Hudson Trevor          1
    Margaret               1
    Karl Howell            1
    """
    #print(pro.df.groupby("call")["Survived"].mean()) # 0.575
    """
    Sex
    female    0.742038
    male      0.188908
    Name: Survived, dtype: float64
    """

    """
    apt            0.000000
    Col             0.500000
    Don             0.000000
    Dr              0.428571
    Jonkheer        0.000000
    Lady            1.000000
    Major           0.500000
    Master          0.575000
    Miss            0.697802
    Mlle            1.000000
    Mme             1.000000
    Mr              0.156673
    Mrs             0.792000
    Ms              1.000000
    Rev             0.000000
    Sir             1.000000
    the Countess    1.000000
    """
    #print(pro.df[pro.df["call"] =="Dr"])
    """
        PassengerId  Survived Pclass                           Name     Sex    Age SibSp  ...      Fare Cabin  Embarked ticket_toget        first_name  call        last_name
    245          246         0      1    Minahan, Dr. William Edward    male  44.00     2  ...   90.0000   C78         Q            2           Minahan    Dr   William Edward     
    317          318         0      2           Moraweck, Dr. Ernest    male  54.00     0  ...   14.0000   NaN         S            1          Moraweck    Dr           Ernest     
    398          399         0      2               Pain, Dr. Alfred    male  23.00     0  ...   10.5000   NaN         S            1              Pain    Dr           Alfred     
    632          633         1      1      Stahelin-Maeglin, Dr. Max    male  32.00     0  ...   30.5000   B50         C            1  Stahelin-Maeglin    Dr              Max     
    660          661         1      1  Frauenthal, Dr. Henry William    male  50.00     2  ...  133.6500   NaN         S            2        Frauenthal    Dr    Henry William     
    766          767         0      1      Brewe, Dr. Arthur Jackson    male  39.25     0  ...   39.6000   NaN         C            1             Brewe    Dr   Arthur Jackson     
    796          797         1      1    Leader, Dr. Alice (Farnham)  female  49.00     0  ...   25.9292   D17         S            1            Leader    Dr  Alice (Farnham)
    """
    #print(pro.df["call"].value_counts())
    """
    Mr              517
    Miss            182
    Mrs             125
    Master           40
    Dr                7
    Rev               6
    Mlle              2
    Major             2
    Col               2
    the Countess      1
    Capt              1
    Ms                1
    Sir               1
    Lady              1
    Mme               1
    Don               1
    Jonkheer          1
    """
    #print(pro.df["first_name"].value_counts())
    """
    Andersson    9
    Sage         7
    Panula       6
    Skoog        6
    Carter       6
                ..
    Hanna        1
    Lewy         1
    Mineff       1
    Haas         1
    Dooley       1
    """
    select = ["Mr","Miss","Mrs","Master","Dr"]
    pro.df["call"] = pro.df["call"].apply(lambda x: 'Others' if x not in select else x )
    pro.df.drop(["Name"],axis= 1, inplace= True)
    # print(pro.df.groupby("call")["Survived"].mean())
    """
    Dr        0.428571 (7)
    Master    0.575000 (40)
    Miss      0.697802 (182)
    Mr        0.156673 (517)
    Mrs       0.792000 (125)
    Others    0.450000 (20)
    """
    # print(pro.df.groupby("Sex")["call"].value_counts())
    """
    female  Miss      182
            Mrs       125
            Others      6
            Dr          1
    male    Mr        517
            Master     40
            Others     14
            Dr          6
    """
    pro.df.drop(["first_name"],axis= 1,inplace= True)
    pro.df.drop(["last_name"],axis= 1,inplace= True)

    # fill cabin
    copy = pro.df[pro.df["Cabin"].notnull()].copy()
    copy["Carbin_alpha"] = copy["Cabin"].apply(lambda x: x[0])
    copy["Carbin_number"] = copy["Cabin"].apply(lambda x: x[1:3])
    #print( "\n",copy[["Pclass", 'Carbin_alpha']].groupby(['Pclass']).value_counts().unstack('Carbin_alpha', fill_value=0).reset_index() )

    """
    Carbin_alpha Pclass   A   B   C   D   E  F  G  T
    0                 1  15  47  59  29  25  0  0  1
    1                 2   0   0   0   4   4  8  0  0
    2                 3   0   0   0   0   3  5  4  0    
    """
    #print( "\n",copy[["Survived", 'Carbin_alpha']].groupby(['Survived']).value_counts().unstack('Carbin_alpha', fill_value=0).reset_index() )
    """
    Carbin_alpha  Survived  A   B   C   D   E  F  G  T
    0                    0  8  12  24   8   8  5  2  1
    1                    1  7  35  35  25  24  8  2  0    
    """
    #print( "\n",copy.groupby("Carbin_alpha")["Survived"].mean() )
    """
    Carbin_alpha
    A    0.466667
    B    0.744681
    C    0.593220
    D    0.757576
    E    0.750000
    F    0.615385
    G    0.500000
    T    0.000000    
    """
    #print( "\n",copy[["Sex", 'Carbin_alpha']].groupby(['Sex']).value_counts().unstack('Carbin_alpha', fill_value=0).reset_index() )

    """
    Carbin_alpha     Sex   A   B   C   D   E  F  G  T
    0             female   1  27  27  18  15  5  4  0
    1               male  14  20  32  15  17  8  0  1 
    """

    #print( "\n",copy[["Sex", 'Carbin_number']].groupby(['Sex']).value_counts().unstack('Carbin_number', fill_value=0).reset_index() )
    """
    Carbin_number     Sex         1   2   3   4   5   6  7  8  9
    0              female  2  1  17  13  17  10   7  10  9  4  7
    1                male  2  3  23  19  14  11  11   6  4  7  7    
    """

    copy.drop(["Carbin_number"],axis= 1,inplace= True)
    #print(pro.df.groupby("Pclass")["Survived"].mean())
    """
    Pclass
    1    0.629630
    2    0.472826
    3    0.242363
    """
    #print(pro.df.groupby("Sex")["Survived"].mean())
    """
    Sex
    female    0.742038
    male      0.188908
    Name: Survived, dtype: float64
    """
    #print(copy.groupby("Sex")["Survived"].mean())
    """
    Sex
    female    0.938144
    male      0.420561
    Name: Survived, dtype: float64    
    """
    pro.df["Cabin"] = pro.df["Cabin"].apply(lambda x: 0 if x is np.nan else 1 )
    print(pro.df)
    # plot carbin_number density plot 
    #print(copy["Carbin_number"].to_numpy())
    #copy = copy[np.invert( list(map(lambda x: len(x) == 0 or x.strip().isalpha() ,copy["Carbin_number"].to_numpy())))]
    #copy["Carbin_number"] = copy["Carbin_number"].astype(int)
    
    #sns.displot(data=copy,x='Carbin_number',hue='Survived', kind="kde", bw_adjust=.25, rug=True)
    #plt.show()
    
    X = datareader.reader(config.input_data_folder + config.input_data_train_filename)
    X.ingress()
    print(X.df[X.df["Embarked"].isna()])
    Y = process.modifier(X.df.copy())
    Y.fullproccess(["Pclass","SibSp","Parch","Sex","Embarked"])

    cols = ["Sex","Pclass","SibSp","Parch","Cabin","Embarked","call","ticket_toget","ticket_header"]
    one_train = pd.get_dummies(Y.df[cols])

    print("------")
    test_ = datareader.reader(config.input_data_folder + config.input_data_test_filename)
    test_.ingress()
    test = process.modifier(test_.df.copy())
    test.fullproccess(["Pclass","SibSp","Parch","Sex","Embarked"])
    one = pd.get_dummies(test.df[cols])
    #one["ticket_header_L"] = 0
    import prince

    mca = prince.MCA(
    n_components=10,
    n_iter=100,
    copy=True,
    check_input=True,
    engine='auto',
    random_state=42
    )
    mca_train = mca.fit(one_train)
    one_train = mca.transform(one_train)

    mca_test = mca.fit(one)
    one = mca.transform(one)

    one_train["Age"] = Y.df["Age"]
    one_train["Fare"] = Y.df["Fare"]
    one["Age"] = test.df["Age"]
    one["Fare"] = test.df["Fare"]
    print(one)
    print(one_train)
    print("-----")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    #one_train.drop(["Sex_male"],axis = 1,inplace = True)
    #one.drop(["Sex_male"],axis = 1, inplace = True)

    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.fit(one_train.to_numpy(), X.df["Survived"].to_numpy())
    #model.feature_importances_
    #plt.barh(one_train.columns, model.feature_importances_)
    #plt.show()
    #print(one_train)
    #print(one)

    #combine = pd.concat([one_train,one])
    #combine = combine[ ( combine["Age"] > 50 )]

    one["Fare"].fillna(47.7, inplace= True )
    prediction = model.predict(one)
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #n_scores = cross_val_score(model, one_train.to_numpy(), X.df["Survived"].to_numpy(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    #print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    s = pd.DataFrame()
    s["PassengerId"] = test.df["PassengerId"]
    s["Survived"] = prediction

    s.to_csv('submit.csv',index=False)