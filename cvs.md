import pandas as pd  
import numpy as np  
from sklearn.impute import IterativeImputer  
from sklearn.preprocessing import OneHotEncoder, StandardScaler  

def missing_value_ratio_per_row(data, threshold=0.5):  
    """  
    统计每一行缺失值占比，并返回缺失值占比超过阈值的行索引  
    """  
    if isinstance(data, np.ndarray):  
        data = pd.DataFrame(data)  

    missing_ratios = data.isnull().sum(axis=1) / data.shape[1]  
    high_missing_rows = missing_ratios[missing_ratios > threshold].index  
    return missing_ratios, high_missing_rows  

def clean_and_save_data(data, high_missing_rows, file_path):  
    """  
    删除指定索引的行，并保存结果到指定文件路径  
    """  
    if isinstance(data, np.ndarray):  
        data = pd.DataFrame(data)  

    data_cleaned = data.drop(index=high_missing_rows).reset_index(drop=True)  
    data_cleaned.to_csv(file_path, index=False)  
    print(f"清洗后的数据已保存到: {file_path}")  

def get_numeric_column_indices(df: pd.DataFrame):  
    """获取所有数值类型的列（int, float）"""  
    return df.select_dtypes(include=['number']).columns.tolist()  

def encode_and_save(df: pd.DataFrame, excluded_columns: list, save_path: str):  
    """  
    对 DataFrame 中除指定列以外的列进行 OneHot 编码，并保存结果  
    """  
    df_copy = df.copy()  
    columns_to_encode = [col for col in df.columns if col not in excluded_columns]  

    encoder = OneHotEncoder(sparse=False, drop='first')  
    encoded_data = encoder.fit_transform(df_copy[columns_to_encode])  
    encoded_col_names = encoder.get_feature_names_out(columns_to_encode)  
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_col_names)  

    final_df = pd.concat([df_copy[excluded_columns].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)  
    final_df.to_csv(save_path, index=False)  
    print(f"数据独热编码已保存到: {save_path}")  
    return final_df  

def process_data_pipeline(input_file, output_file, threshold=0.5):  
    """  
    数据处理管道：读取数据，处理缺失值，编码，标准化，最后保存结果。  
    """  
    try:    
        data = pd.read_csv(input_file)  
        print(f"成功读取输入文件: {input_file}")  

        missing_ratios, high_missing_rows = missing_value_ratio_per_row(data, threshold)  
        clean_and_save_data(data, high_missing_rows, 'temp_cleaned_data.csv')  

        data = pd.read_csv('temp_cleaned_data.csv')  
        numeric_columns = get_numeric_column_indices(data)  

        encoded_data = encode_and_save(data, numeric_columns, 'temp_encoded_data.csv')  
        data = pd.read_csv('temp_encoded_data.csv')  

        # 使用插值法填补缺失值  
        miss_fill = IterativeImputer()  
        data_filled_numpy = miss_fill.fit_transform(data)  
        data_filled = pd.DataFrame(data_filled_numpy, columns=data.columns)  

        # 标准化数据  
        scaler = StandardScaler()  
        data_standardized_numpy = scaler.fit_transform(data_filled)  
        data_standardized = pd.DataFrame(data_standardized_numpy, columns=data.columns)  

        # 保存最终结果  
        data_standardized.to_csv(output_file, index=False)  
        print(f"处理后的数据已保存到: {output_file}")  

    except Exception as e:  
        print(f"处理中发生错误: {e}")  

# 调用处理管道  
process_data_pipeline('data/equity-post-HCT-survival-predictions/train.csv',   
                      'data/equity-post-HCT-survival-predictions/train_processed.csv')