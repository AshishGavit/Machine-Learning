Introduction to ML 

Based on the amount of rainfall, how much would be the crop-yield?

crop-field -> Based on Rainfall -> Predict crop yield

- independent and Dependent Variables
- Numerical and categorical

                                 |-- Supervised -------|-- Regression ------|-- Simple Linear Regression
- Machine Learning Algorithms ---|-- Unsupervised      |-- Classification   |-- Multiple Linear Regression
                                 |-- Reinforcement                          |-- Polynomial Linear Regression

- Applications of Linear Regression
1. Economic Growth ( Economic growth of a country or state in the comming quarter, or predict the GDP of a country)
2. Product Price
3. Housing Sales 
4. Score Prediction (Cricket score)

- Understanding Linear Regression 
Linear Regression is a statistical model used to predict the relationship between indepedent and dependent variables.

Regression Equation
y = m*x + c 


     Y|
      |       /                     y ---> Dependent Variable
    y2|------/.                     x ---> independent Variable
      |     / |                                               _____________
      |  m /  |                                              |     y2-y1   |
      |   /   |                     m ---> Slope of the line | m = -----   |
    y1|--/.- -|                                              |     x2-x1   |
      | / |   |                                              |_____________|
      |/  |   |
      |   |   |                     c ---> Coefficient of the line
     c|   |   |
      |___|___|_________
        x1    x2      X


Multiple Linear Regression
---------------------------
Simple Linear Regression - y = m * x + c 
                                    |-------|--------------|----------------> Independent Variable
Multiple Linear Regression - Y = m1*x1 + m2*x2 + .... + mn*xn + c 
                             |   |_______|______________|       |
                      Dependent Variable       |                |
                                            Slopes         Coefficient
