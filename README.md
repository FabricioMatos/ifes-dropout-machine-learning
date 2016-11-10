# IFES Dropout Machine Learning Study
Student’s dropout prediction for the Brazilian Federal Institute of Education, Science and Technology of Espirito Santo using Machine Learning with Python.

# The Problem

Since the Brazilian federal anual budget for each Federal Institute of Education depends on the number of active students, the high levels of dropouts is a very big concern.

Our objectives for this ‘IFES Pilot Project' study is build a preliminary Machine Learning Model to process student's data available after the 1st year/semester in order to predict dropouts with high accuracy.

This pilot study is just the first step of a much bigger and complex data science project.

# Initial Results

For a pilot project, the results were very motivating. 

The algorithm with the best response was a Logistic Regression with a stronger regularization strength (C=0.1).

The prediction accuracy of the finalized model for the test set was very good (Avg. precision = 85%). 

# The Analyses Process

- EDA1 (Exploratory Data Analyses #1): understand the data available, clean, scale and standardize the data. Also run different binary classifier algorithms and check the preliminary results;
- EDA2: dropout inappropriate features identified in EDA1 and run the same algorithms again;
- EDA3: explore feature reduction techniques in order to check if they can improve the results;
- EDA4: tune the 3 best algorithms and choose the best one;
- Train the chosen model, save it, run the predictions for unseen test sets and evaluate the results accuracy;
- Report the results and suggest the next steps.
