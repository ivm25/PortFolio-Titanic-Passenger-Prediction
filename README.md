# Who Survived Titanic?
# Portfolio Project

## Updates on GitHub
***
*Dataset used: Titanic Passenger dataset, publically available on Kaggle*

## Exploratory data exploration, Data Visualization and Passenger Classification:
*Sqlite and Python 3 are the two primary software techniques used for analysis of the dataset. An Object Oriented Programming (OOP) approach is taken to develop a template for the analysis, with a view to productionalize the code and also build a modular coding structure.*

Initial exploration of the training and test datasets is done by writing simple **SQL queries** and thus getting a basic understanding of the data. In the next stage, **pandas library** is used extensively to understand dataset's descriptive statistics and also visualise the results. A deep dive into the data is undertaken by visualising data patterns using **seaborn and plotly libraries.** 

**Basic Feaure Engineering (ratios and products)** is undertaken within the train and test datasets to derive new features that are more likely to survive, for example families, fare per person or single traveller. 

*The last step of the analysis includes developing models to predict accurate classification of the test dataset features into either a survived or not survived category.*

Developed a Baseline **Logistic Regression Model** with an average accuracy of 79%. Improved the accuracy further by implementing tree based and boosting models. **Highest average accuracy** of 82% is obtained for **Gradient boosting** algorithm, assessed by a **5-fold cross validation** and hyper parameter tuning. 


***


## Kaggle Leaderboard:
The final prediction results using the gradient boosting model helped in reataining leaderboard position within top 60% of the total submissions. 
