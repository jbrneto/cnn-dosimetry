from data import InstancePreprocessor

in_dir = 'C:/Users/joao/Desktop/tese/datasets/Pancreatic/Pancreatic-Normalized'
out_dir = 'C:/Users/joao/Desktop/tese/datasets/Pancreatic/Pancreatic-Preprocessed2'
cts = 'DI'
struct = 'ROI'
struct_name = 'ROI'
dose = 'Doses'

instance = InstancePreprocessor(in_dir,
	cts_filter=cts, structs_filter=struct, struct_name=struct_name, oars_filter=None, doses_filter=dose,
	batch_size=2
)

instance.calculate()
instance.preprocess(out_dir, data_augmentation=0)