import ioniq.core as ct
from ioniq.io import EDHReader
from ioniq.parsers import SpeedyStatSplit, SpikeParser
import ioniq.datatypes as dt
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy import signal
import tqdm


def main():
	"""
	Call all the functions
	"""
	mpl.rcParams['pdf.fonttype'] = 42
	# matplotlib.rcParams['ps.fonttype'] = 42
	plt.rcParams["pdf.use14corefonts"] = False
	plt.rcParams["font.family"] = ["Arial"]
	plt.rcParams["ps.useafm"] = True
	plt.rcParams["figure.dpi"] = 150

	sfm = dt.SessionFileManager()
	#glob.glob("ioniq/test_data/Fixed/**/MBP_P1/**/*dec_pretreatAR20*_*_CH00*/*.edh", recursive=True)
	datafiles = glob.glob("/Users/dinaraboyko/grad_school/cloned_repo/ioniq/test_data/Fixed/MBP_Permutations/MBP_P1/**/*dec_pretreatAR20*_*_CH00*/*.edh", recursive=True)
	#datafiles = []
	#datafiles += glob.glob("../../test_data/Fixed/**/*dec_pretreatAR20*_1[0-9]_CH00*/*.edh", recursive=True)
	ignored_files = np.loadtxt("../../test_data/ignored_files.csv", delimiter=",", dtype=str).transpose()
	#print(ignored_files)
	for ig_file in ignored_files[0]:
		if ig_file in datafiles:
			datafiles.remove(ig_file)

	#datafiles = [u"".join([u"\\\\?\\", os.path.join(os.getcwd(), filename)]) for filename in datafiles]
	# datanames=glob.glob("")
	print(*datafiles, sep='\n')

	class Prefilter:
		def __init__(self, sos):
			self.sos = sos

		def __call__(self, trace):
			trace[:] = signal.sosfiltfilt(self.sos, trace, axis=0)
			return

	def select(tracefile, n_remove=1000, rank="vstep", newrank="vstepgap"):
		for v in tracefile.traverse_to_rank(rank):
			#print(v)
			try:
				v.add_child(ct.MetaSegment(start=v.start + n_remove, end=v.end, rank=newrank, parent=v))
			except:
				pass

	sampling_freq = 200000
	cutoff_freq = 10000
	order = 2

	sos = signal.butter(order, cutoff_freq, btype="lowpass", analog=False, output='sos', fs=sampling_freq)

	lp_filter = Prefilter(sos)

	for datafile in datafiles:
		metadata, current, voltage = EDHReader().read(datafile, voltage_compress=True, downsample=10, prefilter=lp_filter)

		tf = dt.TraceFile(current, voltage, parent=sfm, metadata=metadata, unique_features={"sampling_freq": metadata["eff_sampling_freq"]})
		select(tf, n_remove=100)
	# print(len(sfm.children))
	for i, tf in enumerate(sfm.children):
		plt.figure(122, figsize=(7, 3))
		for child in tf.traverse_to_rank("vstepgap"):
			if -0.25 < child.get_feature("voltage") < 0.25:
				# print(f'child.get_feature(voltage) {child.get_feature("voltage")}')
				plt.plot(child.t, child.current, lw=0.3, c='k')
		plt.ylim(-5e-10, 5e-10)
		plt.title(tf.get_feature('metadata')["HeaderFile"].split("\\")[-2])
	plt.show()


if __name__ == "__main__":
	main()
