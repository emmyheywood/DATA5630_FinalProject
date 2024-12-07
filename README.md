# Time Series Forcasting Application

### .py File 
The app is contained in forecasting_app.py and run with streamlit. It requires the user to upload a time series dataset and can handle data with missing values. It is able to run ETS, Arima, Random Forest and Decisioin Tree models and compare results. Allows users to adjust relevant parameters. 

### .csv Files
cleaned_ocean_data.csv and ocean-acidification-ph-1998-2020 were used to test the app. 
ocean-acidification-munida-state-1998-2020.csv is the original dataset. 

### .ipynb Files
Notebooks are included for the data_cleaning process and the decision tree and random forest model development. 

### To Run
Install appropriate dependencies (included in .yml file) and enter "streamlit run forecasting_app" into terminal to open the app in a browser. 
