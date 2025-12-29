# Load the model
model = pickle.load(open("model.sav", "rb"))

# Prepare the input data
data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
         inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
         inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]

new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                     'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                     'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                     'PaymentMethod', 'tenure'])

# Group tenure in bins of 12 months
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
new_df['tenure_group'] = pd.cut(new_df.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
new_df.drop(columns=['tenure'], axis=1, inplace=True)

# Get dummy variables
new_df_dummies = pd.get_dummies(new_df, drop_first=True)

# Ensure the columns match the model's training data
model_columns = model.feature_names_in_  # Get feature names from the trained model
new_df_dummies = new_df_dummies.reindex(columns=model_columns, fill_value=0)

# Make predictions
single = model.predict(new_df_dummies)
probability = model.predict_proba(new_df_dummies)[:, 1]

# Interpret results
if single == 1:
    o1 = "This customer is likely to be churned!!"
    o2 = "Confidence: {}".format(probability * 100)
else:
    o1 = "This customer is likely to continue!!"
    o2 = "Confidence: {}".format(probability * 100)

return render_template('home.html', output1=o1, output2=o2, 
                       query1=request.form['query1'], 
                       query2=request.form['query2'],
                       query3=request.form['query3'],
                       query4=request.form['query4'],
                       query5=request.form['query5'], 
                       query6=request.form['query6'], 
                       query7=request.form['query7'], 
                       query8=request.form['query8'], 
                       query9=request.form['query9'], 
                       query10=request.form['query10'], 
                       query11=request.form['query11'], 
                       query12=request.form['query12'], 
                       query13=request.form['query13'], 
                       query14=request.form['query14'], 
                       query15=request.form['query15'], 
                       query16=request.form['query16'], 
                       query17=request.form['query17'],
                       query18=request.form['query18'], 
                       query19=request.form['query19'])