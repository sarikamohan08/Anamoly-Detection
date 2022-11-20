from tabula import read_pdf
import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self):
        pass

    def preprocess(self, path):
        dataframe = read_pdf(path, pages='all')
        return dataframe

    def clean(self, dataframe):
        df = pd.DataFrame(dataframe[0].values, columns=['Line No.','Posting Date', 'Due Date', 'Document date', 'Document No.', 'Transaction No.', 'Debit', 'Credit'])
        if len(dataframe) > 1:
            for i in range(1, len(dataframe)):
                df = df.append(pd.DataFrame(dataframe[i].values, columns=['Line No.','Posting Date', 'Due Date', 'Document date', 'Document No.', 'Transaction No.', 'Debit', 'Credit']),ignore_index=True,)
                df = df.append(pd.DataFrame([dataframe[i].columns.to_list()], columns=['Line No.','Posting Date', 'Due Date', 'Document date', 'Document No.', 'Transaction No.', 'Debit', 'Credit']),ignore_index=True,)
        for i in range(1, len(dataframe)):
            df = df.append(pd.DataFrame(dataframe[i].values, columns=['Line No.','Posting Date', 'Due Date', 'Document date', 'Document No.', 'Transaction No.', 'Debit', 'Credit']),ignore_index=True,)
            df = df.append(pd.DataFrame([dataframe[i].columns.to_list()], columns=['Line No.','Posting Date', 'Due Date', 'Document date', 'Document No.', 'Transaction No.', 'Debit', 'Credit']),ignore_index=True,)
        df['Posting Date'] = pd.to_datetime(df['Posting Date'], format='%d-%m-%Y')
        df.replace("Unnamed: 0", np.nan, inplace=True)
        df.replace(np.nan,0, inplace=True)
        df['Line No.'] = df['Line No.'].astype(int)
        df['Transaction No.'] = df['Transaction No.'].astype(int)
        df['Debit'] = df['Debit'].astype(float)
        df['Credit'] = df['Credit'].astype(float)
        df['Debit'] = df['Debit'].apply(lambda x: -x if x > 0 else x)
        df = df.groupby(['Transaction No.'], as_index=False).agg({'Debit': 'sum', 'Credit': 'sum'})
        df['Balance'] = df['Debit'] + df['Credit']
        return df
    
    def run(self, path, save_path):
        df = self.preprocess(path)
        df = self.clean(df)
        self.save(df, save_path)

    def save(self, df, path):
        df.to_csv(path, index=False)
        

if __name__ == '__main__':
    path = 'data/dataset.pdf'
    preprocessor = Preprocess()
    dataframe = preprocessor.preprocess(path)
    df = preprocessor.clean(dataframe)
    preprocessor.save(df, 'data/preprocessed.csv')

    # save as pickle 
    # df.to_pickle('data/preprocessed.pkl')


# Path: train_test.py