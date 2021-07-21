import pandas as pd # The Pandas module is used for working with tabular data
# The Matplotlib and seaborn module is used for data visualization.
import matplotlib.pyplot as plt
import seaborn as sns
# Use 'train_test_split' to split the data
from sklearn.model_selection import train_test_split 
# the 'sklearn.metrics' module contain metrics that get's the accuacy of our model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Working with tabular data using pandas
def data_manipulation():
    # step1: get the data
    df = pd.read_csv('./datasets/creditcard.csv')

    # step2: get information about the data, use - head(), tail(), shape, info(), columns
    print(df.head())
    print(df.tail())
    print(df.shape)
    print(df.info())
    print(df.columns)
    
    # step3: get statistical information about the data, use - describe()
    df2 = df[['Time', 'Amount', 'Class']] # 'Time' and 'Amount' is features variable -- 'Class' is target variable
    print(df2.describe(include='all'))
    
    # step4: Check for missing values
    # isna() is used for checking if any row in dataset is empty
    # any() is used for sort summary for each columns
    missing_value = df.isna().any() 
    print(missing_value)
    null_cols = pd.DataFrame({'Columns':df.isna().sum().index,'No. Null Values':df.isna().sum().values, 'Percentage':df.isna().sum().values/df.shape[0]})
    print(null_cols)
    
    # step5: Get no. of fraud and not fraud transaction
    nfcount, per_nf, fcount, per_f = fraud_and_notfraud(df) # 'target is a series, it contains one column 'Class'
    print('Number of Not Fraud: ',nfcount)
    print('percentage of total not fraud transaction in the dataset: ',per_nf)
    print('Number of Fraud: ',fcount)
    print('percentage of total fraud transaction in the dataset: ',per_f)
    return df2

def fraud_and_notfraud(df):
    nfcount = 0 # do a count for not fraud
    fcount = 0 # do a count for fraud
    fraudOrNot = df['Class']
    for i in range(len(fraudOrNot)): # get the length of 'fraudOrNot' - series
        if fraudOrNot[i] == 0: # '0' is for not fraud
            nfcount = nfcount + 1
        elif fraudOrNot[i] == 1: # '1' is for fraud
            fcount = fcount + 1
    # get the percentage
    per_nf = (nfcount/len(fraudOrNot))*100
    per_f = (fcount/len(fraudOrNot)*100)
    return nfcount, per_nf, fcount, per_f

def data_visualization(df):
    # Plot countplot() graph for column 'Class'
    plt.title('Bar plot for Fraud VS Genuine transactions')
    sns.countplot(x = 'Class', data = df, palette = 'Blues', edgecolor = 'w')
    plt.show()

    # Plot graph for columns 'Amount' and 'Time'
    x = df['Amount']
    y = df['Time']
    plt.title('Time Vs amount')
    plt.plot(x,y)
    plt.show()

    # Plot graph for distribution of amount
    plt.figure(figsize=(10,8))
    plt.title('Amount Distribution')
    sns.distplot(df['Amount'],color='red')
    plt.show()

    # find outliers using graph
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(df['Amount'],df['Time'])
    ax.set_xlabel('Amount')
    ax.set_ylabel('Time')
    plt.show()

    # Plot a heatmap graph for correlation matrix
    # Correlation metrics help us to understand the core relation between two attributes.
    correlation_matrix = df.corr() # corr() method get all the correlation for all the columns
    plt.figure(figsize=(14,9))
    sns.heatmap(correlation_matrix, vmax = .9, square=True)
    plt.show()

def machine_learning(df):
    # To start with modelling First we need to split the dataset
    # 80% → 80% of the data will use to train the model
    # 20% → 20% to validate the model
    x = df.drop(['Class'], axis=1) # drop the target variable
    y = df['Class'] # 'Class' is the target
    # split the data using 'train_test_split()'
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size= 0.2, random_state=42)

    # create machine learning model using logistic regression
    logisticreg = LogisticRegression(solver='lbfgs')
    logisticreg.fit(xtrain, ytrain) # logisticreg is our model

    # predict the model using predict() 
    y_pred = logisticreg.predict(xtest) # use the model to predict

    # use the score() to get the accuracy of our model
    accuracy = logisticreg.score(xtest, ytest)
    print('Accuracy score of the Logistic regression model: ', accuracy*100,'%')
    
if __name__ == '__main__':
    df = data_manipulation()
    data_visualization(df)
    machine_learning(df)