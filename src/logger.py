import numpy as np
import pandas as pd


class Logger:
	def __init__(self, output_dir):
		self._output_log_dir = self._output_dir + "out.csv"

	def save_output(self, out):
		out_df = pd.DataFrame(dict)
		out_df.to_csv(self._output_log_dir)
