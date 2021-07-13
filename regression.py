# Importing the libraries
import numpy as np # Numeric Python
import pandas as pd # for Data manipulation 
import matplotlib.pyplot as plt # for Data Visualization
import seaborn as sns # Data Visualization - it's build on top of matplotlib

# Importing the dataset and Extracting the Independent and Dependent Variables
companies = pd.read_csv('')
X = companies.iloc[:,:-1].values
y = companies.iloc[:,4].values

companies.head()