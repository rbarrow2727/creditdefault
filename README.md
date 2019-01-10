# creditdefault
Predicting Default Rate of Credit Customers

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

Somewhat surprisingly, after trimming the width of the original data set with the 
methods I mentioned above, the accuracy results from the models were all somewhat similar.  
For example, the base data set (or control) I ran through the Random Forest algorithm 
achieved an accuracy of 81.4%.  Some of the parameters I toggled within the algorithm, 
criterion (gini, entropy) and class_weight (balanced, balanced_subsample) affected 
performance but only ever so slightly.  The accuracy from a couple of my featured engineered 
data sets, RFE and correlation, had only slightly better results performing at 81.6% and 81.5% 
accuracy respectively.  I found this odd compared to some previous work I’ve done where the featured 
engineered data sets performed noticeably better.  With this initial test completed, it was time to 
start brainstorming again on how I could improve accuracy and edit the data in a way that has more of 
an effect, positive or negative, on the accuracy of the model.  
	As I determined in the EDA, there were some clear categorical factors that increased the likelihood 
  of default.  For example, in the graphic below you can see how sex and credit limit affect default 
  probability.  The graph illustrates how males with lower credit limits are more likely to default. 
	As I continued to remove specific attributes and tinker with the parameters inside the MLP, KNN, and 
  SVM algorithms I grew more frustrated with the results I was getting.  For example, in one data set I 
  removed all customer categorical variables (sex, age, marriage, education) and left only credit limit, 
  payment status, bill amount, and the amount paid variables.  And in another data set I did the opposite 
  and removed all credit information and just left customer categorical variables I just mentioned.  
  The best accuracy results I got from these 2 data sets were 81% and 78% respectively.  Though 3% difference 
  was a much larger difference than a few tenths, I still did not consider this a very drastic shift.  Instead 
  of continuing to adjust the width of the data set I decided to take a different approach.  
	Searching for drastically different accuracy results between two data sets modeled, I decided to quit adjusting 
  the width of the data set and instead evaluate the length.  The two data sets I produced, I labeled as hiDEF and 
  loDEF.  The hiDEF data set included all 23 attributes but the observations recorded were only males, high school 
  educated with less than $100k credit limit.  These characteristics represented a high likelihood of default, or 
  hiDEF, and included 1202 total observations, 402 of which actually defaulted equaling a percentage of 33.4% 
  (default % from base data set was 22% for reference).  The loDEF data set included only females with a college 
  degree or higher, married and a credit limit over $300k.  This data set included 1343 observations, 170 of which 
  defaulted equaling a percentage of 12.6%.  I began modeling these two data sets and got the varying results I was 
  hoping for to better indicate business intelligence towards identifying target customers.  The loDEF modeled through 
  Random Forest saw accuracy of 88.08%.  The hiDEF modeled through Random Forest saw accuracy of 70.36%.  Because loDEF 
  has a lower percentage of customers defaulting, the modeled performed very well.  In other words, it was much easier 
  for the model to predict because the default vs non-default was so one-sided. Conversely, the hiDEF model had lower 
  accuracy because there was a greater mix of default customers.
Data Set	Actual Default %	Predicted Accuracy
loDEF	12%	88%
RFE	22%	81%
hiDEF	33%	70%
	The table below illustrates my findings.  The lower the actual default % of the data set, the higher the predicted accuracy.  
  This is a basic assumption in most classification modeling, but it does provide insight into better business practices for 
  Credit One.  If we can recommend customers that fall into the loDEF data set characteristics and have a lower actual 
  likelihood of default, we can better predict the probability of default.  Since we cannot control spending habits and 
  find the underlying reason as to ‘why’ customers default, by setting certain guidelines on customers we recommend 
  (based on the criterion of sex, education, marriage, etc) we can actually control the forecasted accuracy of default.  
  From my approach I believe this is the best way we can guide our customers to lower default rates and uncover more 
  information about their credit behavior.  





