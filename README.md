# creditdefault
Predicting Default Rate of Credit Customers Business Report

Analysis of 11 different feautred engineered data subsets:
 - Recursive Feature Elimination
 - Covariance
 - Correlation
 - Other handpicked attribute variations used for testing and experimentation
 
Algorthims tested:
 - Random Forest
 - KNN
 - MLP (Multi-Layer Perceptron)
 - Support Vector Machhine

The base data set (or control) I ran through the Random Forest algorithm achieved an accuracy of 81.4%.  

The accuracy from a couple of my featured engineered data sets, RFE and correlation, had only slightly better results performing at 81.6% and 81.5% accuracy respectively.  

As I determined in the EDA,  males with lower credit limits are more likely to default. 

In one data set I removed all customer categorical variables (sex, age, marriage, education) and left only credit limit, payment status, bill amount, and the amount paid variables.  And in another data set I did the opposite and removed all credit information and just left customer categorical variables I just mentioned. The best accuracy results I got from these 2 data sets were 81% and 78% respectively.  
  
Instead of continuing to adjust the width of the data set I decided to take a different approach.  
  
  
Searching for drastically different accuracy results between two data sets modeled, I decided to quit adjusting the width of the data set and instead evaluate the length.  The two data sets I produced, I labeled as hiDEF and loDEF.  
  
The hiDEF data set included all 23 attributes but the observations recorded were only males, high school educated with less than $100k credit limit.  These characteristics represented a high likelihood of default, or hiDEF, and included 1202 total observations, 402 of which actually defaulted equaling a percentage of 33.4% (default % from base data set was 22% for reference).  
  
The loDEF data set included only females with a college degree or higher, married and a credit limit over $300k.  This data set included 1343 observations, 170 of which defaulted equaling a percentage of 12.6%.  I began modeling these two data sets and got the varying results I was hoping for to better indicate business intelligence towards identifying target customers.  
  
The loDEF modeled through Random Forest saw accuracy of 88.08%.  The hiDEF modeled through Random Forest saw accuracy of 70.36%.  
  
Data Set     Actual Default %	Predicted Accuracy
loDEF	           12%	              88%
RFE	           22%		      81%
hiDEF	           33%		      70%

The lower the actual default % of the data set, the higher the predicted accuracy. This is a basic assumption in most classification modeling, but it does provide insight into better business practices for Credit One.  If we can recommend customers that fall into the loDEF data set characteristics and have a lower actual likelihood of default, we can better predict the probability of default.  Since we cannot control spending habits and find the underlying reason as to ‘why’ customers default, by setting certain guidelines on customers we recommend (based on the criterion of sex, education, marriage, etc) we can actually control the forecasted accuracy of default.  From my approach I believe this is the best way we can guide our customers to lower default rates and uncover more information about their credit behavior.  





