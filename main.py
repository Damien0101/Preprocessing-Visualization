
from preprocessing.categorical_to_num_dict import CategoricalNumDict
from preprocessing.preprocessing import Preprocessing
from preprocessing.df_splitter import DataFrameSplitter
from preprocessing.clean_data import CleanData
from utils.save_read import read_to_df



def main():
    """
    Main function to create an instance of DataProcessor and execute the processing steps.
    """
    data_path = "data/final_dataset.json"
    zip_path = "data/zipcode-belgium.json"
    save_path = "data/clean_dataset.csv"
    
    p = CleanData(data_path, zip_path, save_path)
    p.process()
    
    
    splitter = DataFrameSplitter()
    # splitter.split_and_save(p.get_data())
    
    pre = Preprocessing()
    dicts = CategoricalNumDict()
    
    pre.transform_categorical(p.get_data(), "PEB", dicts)
    pre.transform_categorical(p.get_data(), "FloodingZone", dicts)
    
    # pre.transform(p.get_data())

    # print(p.get_data().head())
    # print(p.get_column())
    # print(p.get_summary_stats())
    

if __name__ == "__main__":
<<<<<<< HEAD
    temp = csv()
    temp.temp()
    #main()
=======
    main()
>>>>>>> parent of 723a5e1 (pouetpouet)
