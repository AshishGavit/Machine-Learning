import pandas as pd # The Pandas module is used for working with tabular data
# The Matplotlib and seaborn module is used for data visualization.
import matplotlib.pyplot as plt
import seaborn as sns

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
    df2 = df[['Time', 'Amount', 'Class']] # 'Time' and 'Amount' is dependent or features variable -- 'Class' is target variable
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

if __name__ == '__main__':
    data_manipulation()