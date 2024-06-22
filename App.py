from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import ast

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Mendapatkan data transaksi dari form
        transactions = request.form.getlist('transaction')
        min_support = float(request.form['min_support'])
        
        # Parsing input transaksi
        dataset = [trans.strip().split(',') for trans in transactions if trans.strip()]
        
        # Create the TransactionEncoder and transform the dataset
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Perform FP-Growth
        frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
        
        return render_template('result.html', transactions=transactions, frequent_itemsets=frequent_itemsets)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=15001)
