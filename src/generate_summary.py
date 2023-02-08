import os
from numpy import mean, std, percentile
import pandas as pd
import  argparse
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_squared_error as mse

BAL_AUTH_CODES = ["MISO", "SWPP", "PJM", "ERCO", "CISO", "PGE", "SOCO"]

THIS_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(THIS_DIR, "..", "OUTPUT")
SUMMARY_FILE = os.path.join(THIS_DIR, "..", "summary.csv")

LABEL_SUFFIX = {"WEIGHTED" : "2020-pop",
				"UNWEIGHTED" : "unweighted",
				"CENTROID" : "centroid"}

def mbe(y_true: list, y_pred: list) -> float:
	""" Mean Bias Error """	
	return mean(y_pred) - mean(y_true)

def load_data(data_file: str) -> pd.DataFrame:
	return pd.read_csv(data_file, index_col=0, parse_dates=True)

def main():

	summary = pd.DataFrame()

	for bal_auth in BAL_AUTH_CODES:

		# load data
		temp = dict()

		for label in LABEL_SUFFIX:
			# filename
			temp_file = os.path.join(OUTPUT_DIR, f"{bal_auth}-temperature-{LABEL_SUFFIX[label]}.csv")
			temp[label] = load_data(temp_file)

		# extreme hours
		pct_25, pct_75 = percentile(temp["WEIGHTED"].values, [25, 75])
		top_25_pct = temp["WEIGHTED"].index[temp["WEIGHTED"]["Temperature (K)"] >= pct_75]
		bottom_25_pct = temp["WEIGHTED"].index[temp["WEIGHTED"]["Temperature (K)"] <= pct_25]

		for label in LABEL_SUFFIX:
			temp[f"{label}-T25"] = temp[label].loc[top_25_pct]
			temp[f"{label}-B25"] = temp[label].loc[bottom_25_pct]

		# gather stats
		bal_auth_summary = pd.Series(name=bal_auth, dtype=float)

		# individual
		for tag in ["", "-T25", "-B25"]:
			for label in ["WEIGHTED", "UNWEIGHTED", "CENTROID"]:
				bal_auth_summary.loc[f"{label}{tag} MEAN"] = mean(temp[f"{label}{tag}"].values)
				bal_auth_summary.loc[f"{label}{tag} STD"] = std(temp[f"{label}{tag}"].values)
		
			# comparative
			for label in ["UNWEIGHTED", "CENTROID"]:
				bal_auth_summary.loc[f"WEIGHTED-{label}{tag} MAPE"] = mape(temp[f"WEIGHTED{tag}"].values, temp[f"{label}{tag}"].values)
				bal_auth_summary.loc[f"WEIGHTED-{label}{tag} MSE"] = mse(temp[f"WEIGHTED{tag}"].values, temp[f"{label}{tag}"].values)
				bal_auth_summary.loc[f"WEIGHTED-{label}{tag} MBE"] = mbe(temp[f"WEIGHTED{tag}"].values, temp[f"{label}{tag}"].values)

		summary = pd.concat((summary, bal_auth_summary),axis=1)
	summary.T.to_csv(SUMMARY_FILE)

if __name__ == "__main__":
	main()