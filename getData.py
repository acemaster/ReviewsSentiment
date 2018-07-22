'''
Json structure of final data
{
	id: {
		avaliable_dates: [{
			date:
			avaliable:
			price:
		}],
		listings: {
			get columns from dataset
		},
		reviews: {

		}
	}
}
'''
import pandas
import os


csv_delimiter = ","

def read_csv(filename):
	df = pandas.read_csv(filename, sep=csv_delimiter)
	return df

def getData(root):
	file_cal = root+"/calendar.csv"
	file_list = root+"/listings.csv"
	# file_neighbourhoods = root+"/neighborhoods.csv"
	file_reviews = root+"/reviews.csv"
	# Reading calendar csv
	df_cal = read_csv(file_cal)
	df_cal_filled = df_cal.fillna("$0")
	df_list = read_csv(file_list)
	# df_neighbourhoods = read_csv(file_neighbourhoods)
	df_reviews = read_csv(file_reviews)
	df_list_index = df_list.set_index('id')
	df_train_data = read_csv("train.csv")
	#Data joined and ready
	df_join_cal = df_cal_filled.join(df_list_index,on="listing_id",lsuffix="_overlap")
	df_join_reviews = df_reviews.join(df_list_index,on="listing_id",lsuffix="_overlap")
	#Creating training data only for reviews
	df_join_train = df_train_data.join(df_join_reviews,on="listing_id",lsuffix="_overlap")
	print df_join_train.head(10)
	print os.path.isfile("testTrain.csv")
	if os.path.isfile("testTrain.csv") == False:
		df_join_train[['listing_id_overlap','comments','review_scores_rating']].dropna().to_csv("testTrain.csv")
	else:
		with open('testTrain.csv', 'a') as f:
			df_join_train[['listing_id_overlap','comments','review_scores_rating']].dropna().to_csv(f, header=False)


	# Write all this to a file

os.remove("testTrain.csv")
for root, dirs, files in os.walk("contestdata"): 
	if "/" in root:
		print root
		getData(root)