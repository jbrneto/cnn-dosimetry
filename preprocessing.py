from data import InstancePreprocessor

in_dir = 'C:/Users/joao/Desktop/tese/datasets/Pancreatic/Pancreatic-Normalized'
out_dir = 'C:/Users/joao/Desktop/tese/datasets/Pancreatic/Pancreatic-Preprocessed2'
cts = 'Novo'#'DI'
struct = None#'ROI'
struct_name = None#'ROI'
dose = 'Doses'

instance = InstancePreprocessor(in_dir,
	cts_filter=cts, structs_filter=struct, struct_name=struct_name, oars_filter=None, doses_filter=dose,
	batch_size=1#2
)

instance.calculate()
instance.preprocess(out_dir, data_augmentation=0)