
import pandas
import os

filename = "testTrain.csv"
csv_delimiter = ","

def read_csv():
    df = pandas.read_csv(filename, sep=csv_delimiter)
    return df

columns = ['listing_id_overlap', 'comments', 'review_scores_rating']	

def labelCreation():
    df_final = pandas.DataFrame([],columns=columns)
    for i in range(df_cleaned.shape[0]):
        rowVal = df_cleaned.loc[i]
        rating = 0

        if rowVal['review_scores_rating'] > 50:
            rowVal['review_scores_rating'] = 1
        else:
            rowVal['review_scores_rating'] = 0
        df_final.append(rowVal)

    if os.path.isfile("testTrainNew.csv") == False:
        df_final[['listing_id_overlap','comments','review_scores_rating']].dropna().to_csv("testTrainNew.csv")
    else:
        with open('testTrainNew.csv', 'a') as f:
            df_final[['listing_id_overlap','comments','review_scores_rating']].dropna().to_csv(f, header=False)

df = read_csv()
df_cleaned = df.dropna()
labelCreation()