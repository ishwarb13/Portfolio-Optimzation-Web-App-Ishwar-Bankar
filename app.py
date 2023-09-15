from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from portfolio_optimizer import optimize_portfolio  # Import the function
from stock_pred import get_stock_predictions


app = Flask(__name__)

# Load the linear regression model
# with open('static/models/lin_reg_model.pkl', 'rb') as file:
#     lin_reg_model = pickle.load(file)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/stock_prediction', methods=['GET', 'POST'])
def stock_prediction():
    stocks = ['ABT', 'IBM', 'ORCL', 'INTC','MO','NVS','PFE','TMUS','KO','XOM', "CL=F",   # Crude Oil (WTI)
    "BZ=F",   
    "NG=F",   
    "SI=F",   
    "HG=F",   
    "ZC=F",   
    "CT=F",   
    "LE=F",   
    "KC=F",   
    ]
    
    prediction = 0
    actual_values = 0

    if request.method == 'POST':
        selected_stock = request.form['stock_select']
        actual_values, prediction = get_stock_predictions(selected_stock)

    return render_template('stock_prediction.html', stocks=stocks, prediction=prediction, actual_values=actual_values)


@app.route('/portfolio_recommendation', methods=["GET", "POST"])
def portfolio_recommendation():
    allocated_amounts = {}
    annRisk = 0
    annRet = 0
    
    if request.method == "POST":
        investment = float(request.form.get("investment"))
        xOptimalArray, assetLabels, annRisk, annRet = optimize_portfolio()

        allocated_amounts = {asset: investment * weight for asset, weight in zip(assetLabels, xOptimalArray[0])}

    return render_template('portfolio_recommendation.html', allocated_amounts=allocated_amounts, annRisk=annRisk, annRet=annRet)

if __name__ == '__main__':
    app.run(debug=True)
