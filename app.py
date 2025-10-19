from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

data = pd.read_csv('House_Pricing.csv')
print(data.columns)


data = data.dropna()


X = data[['Flat Area (in Sqft)', 'No of Bedrooms', 'No of Bathrooms', 'No of Floors', 'Age of House (in Years)']]
y = data['Sale Price']



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open('model_final.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('house_pricing.html')    

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
       flat_area = float(request.form['flat_area'])
       bedrooms = float(request.form['bedrooms'])
       bathrooms = float(request.form['bathrooms'])
       floors = float(request.form['floors'])
       age = float(request.form['age'])

       model = pickle.load(open('model_final.pkl','rb'))
       scaler = pickle.load(open('scaler.pkl','rb'))

       input_data = [[flat_area, bedrooms, bathrooms, floors, age]]
       input_scaled = scaler.transform(input_data)
       prediction = model.predict(input_scaled)[0]

       return render_template('house_pricing.html', prediction_text=f'Predicted House Price: ₹{prediction:,.2f}')

    except Exception as e:
        return render_template('house_pricing.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
