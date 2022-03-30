import os
from numpy import mean, std
import pandas as pd
import  argparse
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_squared_error as mse

BAL_AUTH_CODES = ["MISO", "SWPP", "PJM", "ERCOT", "CAISO", "PGE", "SOCO"]

THIS_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(THIS_DIR, "..", "OUTPUT")

def mbe(y_true: list, y_pred: list) -> float:
	""" Mean Bias Error """	
	return np.mean(y_pred) - np.mean(y_true)

def load_data(data_file: str) -> pd.DataFrame:
	return pd.read_csv(data_file)

def main():

	summary = pd.DataFrame(index=BAL_AUTH_CODES)

	for bal_auth in BAL_AUTH_CODES:

		# filenames
		temp_file = dict()
		temp_file["WEIGHTED"] = os.path.join(OUTPUT_DIR, f"{bal_auth}-temperature-2020-pop.csv")
		temp_file["UNWEIGHTED"] = os.path.join(OUTPUT_DIR, f"{bal_auth}-temperature-unweighted.csv")
		temp_file["CENTROID"] = os.path.join(OUTPUT_DIR, f"{bal_auth}-temperature-centroid.csv")

		# load data
		temp = dict()
		temp["WEIGHTED"] = load_data(temp_file["WEIGHTED"])
		temp["UNWEIGHTED"] = load_data(temp_file["UNWEIGHTED"])
		temp["CENTROID"] = load_data(temp_file["CENTROID"])

		for label in ["WEIGHTED", "UNWEIGHTED", "CENTROID"]:
			summary.loc[bal_auth, f"{label} MEAN"] = mean(temp[label])
			summary.loc[bal_auth, f"{label} STD"] = std(temp[label])
		
		for label in ["UNWEIGHTED", "CENTROID"]
			summary.loc[bal_auth, f"WEIGHTED-{label} MAPE"] = mape(temp["WEIGHTED"])
			summary.loc[bal_auth, f"WEIGHTED-{label} MSE"] =
			summary.loc[bal_auth, f"WEIGHTED-{label} MBE"] = 

if __name__ == "__main__":
	main()