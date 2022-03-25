import os
import sys
import time
import pandas as pd
import numpy as np
from pickle import dump
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pandas_profiling import ProfileReport
 
ROOT_FOLDER = 'data_preprocessing'
 
class DataPreprocessing:
   
    def __init__(self, input_path, output_path ,cur_path, file_name, target_col = None, categorical_cols = None,
                 numerical_cols = None):
 
        self.input_path = input_path
        self.output_path = output_path
        self.cur_path = cur_path
        self.file_name = file_name
       
        self.target_col = target_col
        self.categorical_cols = [col for col in categorical_cols if col != '']
        self.numerical_cols = [col for col in numerical_cols if col != '']    
        
        self.sub_path = None
 
 
     #데이터 컬럼 읽기
    def read_data(self, cols) :       
        file_extension = self.input_path.split('.')[-1]
       
        if file_extension == 'csv':
            
            data = pd.read_csv(self.input_path).loc[:, cols]
 
        else :
           
            raise Exception('Check Input Data Extention')
            
        return data.values, data
   
    #컬럼 타입 적용
    def dataframe_type_apply(self, data) :
       
        if numerical_cols  : data[self.numerical_cols] = data[self.numerical_cols].astype(np.float64)
           
        if categorical_cols : data[self.categorical_cols] = data[self.categorical_cols].astype('str')
 
    #중복제거
    def deduplicate_data(self,data,is_remove_duplicate):       
       
        if is_remove_duplicate :
            
            data.drop_duplicates(ignore_index = True, inplace = True)
           
 
    def missing_numerical_value(self,data,method,val):
       
        #숫자 결측값 처리
       
        method = method.upper()
       
        if method == 'INTERPOLATE':
                
            if val not in ['linear', 'time', 'index', 'values', 'nearest',
                          'zero', 'slinear', 'quadratic', 'cubic', 'barycentric',
                          'krogh', 'polynomial', 'spline', 'piecewise_polynomial',
                          'from_derivatives', 'pchip', 'akima', 'None']:
                raise
               
            data.loc[:, self.numerical_cols] = data.loc[:, self.numerical_cols].interpolate(interpolate_method)
 
        elif method == 'FILL_NA':
           
            if val not in ['backfill', 'bfill', 'pad', 'ffill', 'None']:
               
                raise
 
            data.loc[:, self.numerical_cols] = data.loc[:, self.numerical_cols].fillna(method=val)
   
        
        elif method == 'ROW_DROP':
           
            data.dropna(axis=0, subset=self.numerical_cols, inplace = True)
      
        
        
    def missing_categorical_value(self,data,method,val):
   
        #문자 결측값 처리
       
            method = method.upper()
           
            if method == 'MODE' :
               
                data[self.categorical_cols] = data[self.categorical_cols].apply(lambda x : x.fillna(x.mode()[0]))
               
            elif method == 'ROW_DROP' :
               
                data.dropna(axis=0, subset=self.categorical_cols, inplace = True)
               
    #스케일러
    def target_transform(self,data):     
 
        data.dropna(inplace=True, axis=0, subset = self.target_col)
         
        if data[self.target_col].dtypes is 'object':
       
            encoder = LabelEncoder()
            data.loc[:, self.target_col] = encoder.fit_transform(data[self.target_col])
           
            path = self.sub_path + '/' +'target_encoder.pkl'
            dump(encoder, open(path, 'wb'))
           
    def numerical_scaler_preprocessing(self, data, _scaler):
       
        _scaler = _scaler.upper()
       
        if not self.numerical_cols:
            
            return
       
        if _scaler == 'MINMAX_SCALER' : scaler = MinMaxScaler()
            
        elif _scaler == 'STANDARD_SCALER' : scaler = StandardScaler()
        
        data.loc[:, self.numerical_cols] = scaler.fit_transform(data.loc[:, self.numerical_cols])
         
        path = self.sub_path + '/' + 'scaler.pkl'
        dump(scaler, open(path, 'wb')) 
                
    def categorical_encoder_preprocessing(self, data, _encoder):
       
        _encoder = _encoder.upper()
        
      
        if not self.categorical_cols:
            return
       
        if _encoder == 'LABEL_ENCODER' : encoder = LabelEncoder()
           
        elif _encoder == 'ONEHOT_ENCODER' : encoder = OneHotEncoder()
           
        data.loc[:, self.categorical_cols] = encoder.fit_transform(data.loc[:, self.categorical_cols])
       
        path = self.sub_path + '/' + 'encoder.pkl'
        dump(encoder, open(path, 'wb'))
       
        
    def makeDirs(self, path):
        
        now = time.strftime("%y%m%d_%Hh%Mm%Ss")
        file_name = path.split('.')[0].split('/')[-1]
        dir_name = file_name + '_' + now
        dir_path = '/'.join(path.split('/')[:-1]) + '/' + dir_name
       
        if not os.path.exists(dir_path):
           
            os.makedirs(dir_path)
       
        else :  print('Directory path is not available.')
 
        return dir_path  
    
    def default_profile(self,data,file_name) :
       
        profile = data.profile_report(minimal=True)
       
        profile.to_file(self.sub_path + '/' + file_name)
   
    def preprocess(self, is_remove_duplicate, f_missing_m, f_missing_v, 
                   c_missing_m, c_missing_v, _scaler,_encoder):
       
        # 폴더 생성
       
        root_path = self.cur_path+'/'+ROOT_FOLDER
       
        if not os.path.exists(root_path) :
           
            os.makedirs(root_path)
 
        self.sub_path = self.makeDirs(root_path+'/'+self.file_name)
       
        #데이터 입력
        
        _, data = self.read_data(self.numerical_cols + self.categorical_cols + self.target_col)
       
        self.default_profile(data,'original_profile')
        
        #데이터 변환
        self.dataframe_type_apply(data)
       
        #target col 처리
        if self.target_col : self.target_transform(data)
           
        #중복 제거
        self.deduplicate_data(data, is_remove_duplicate)
           
        #결측치 제거       
        if self.numerical_cols :
            self.missing_numerical_value(data,f_missing_m,f_missing_v)
       
        if self.categorical_cols :
            self.missing_categorical_value(data,c_missing_m,c_missing_v)
       
        #스케일링
        if self.numerical_cols :
            self.numerical_scaler_preprocessing(data,_scaler)
            
        if self.categorical_cols :
            self.categorical_encoder_preprocessing(data,_encoder) 
        
          
        #파일 저장
        np.savez_compressed(self.sub_path + '/' + file_name +'_'+ 'preprocessed.npz', data=data[self.numerical_cols+self.categorical_cols],
                        columns=[self.numerical_cols+self.categorical_cols], target_name=self.target_col,
                            target=data[target_col])
 
        data.to_csv(self.sub_path + '/' + file_name +'_'+ 'preprocessed.csv', sep=',', index=False)
       
        self.default_profile(data,'preprocessed_profile')
                         
        return data
 
   
    ###이전 코드 사용하지 않는 유틸 함수###
   
    #파일 유무 확인
#     def is_exist(self, file_path):
       
#         path = file_path.split('.')
#         extension = path[-1]
#         identifier = 1
 
#         while os.path.exists(file_path):
#             file_path = path[0] + str(identifier) + '.' + extension
#             identifier += 1
 
#         if extension == 'npz':
#             return file_path[:-4]
 
#         return file_path 
    
#     def olsDescribe(self, data, feature_cols, target_col):
#         polynomial = target_col.pop() + '~' + '+'.join(feature_cols)
#         model = ols(formula=polynomial, data=data).fit()
 
#         path = self.is_exist(self.cur_path + '/' + self.file_name + '_ols_describe.html')
#         file = open(path, 'w')
#         file.write(model.summary().as_html())
#         file.close()
       
#     def pearsonCorrelation(self, data):
#         corr = np.array(data.corr())
#         z_text = np.around(corr, decimals=2)
#         fig = ff.create_annotated_heatmap(corr, y = list(preprocessed_data.columns), x = list(preprocessed_data.columns),
#                                           annotation_text=z_text, colorscale='plasma')
#         path = self.is_exist(self.cur_path + '/' + self.file_name + '_corr_heatmap.html')
#         fig.write_html(path)
       
#     def change_np_to_pandas(path):
 
#     load_datas = np.load(path,allow_pickle=True)
#     data = load_datas['data']
#     target = load_datas['target']
#     columns = load_datas['columns']
#     target_name = load_datas['target_name']
#     col_data = pd.DataFrame(data,columns=columns)
#     load_datas.close()
#     return col_data, target,columns,target_name
 
#     def change_np_to_pandas_unsupervised(path):
#         load_datas = np.load(path,allow_pickle=True)
#         data = load_datas['data']
#         columns = load_datas['columns']
#         col_data = pd.DataFrame(data,columns=columns)
#         load_datas.close()
#         return col_data,columns
   
#     def timedata_preprocessing(data,column):
#         data['date'] = pd.to_datetime(data[column])
#         data['year'] = data['date'].dt.year
#         data['month'] = data['date'].dt.month
#         data['day'] = data['date'].dt.day
#         return data
 
   
#     def change_excel_to_pandas(path,target_name):
#         if path.split('.')[-1] == 'csv':
#             data = pd.read_csv(path)
#             columns = data.columns
#             target_name = target_name
#             target = data[target_name]
#             return data,columns,target_name,target
#         elif path.split('.')[-1] == 'tsv':
#             data = pd.read_csv(path)
#             columns = data.columns
#             target_name = target_name
#             target = data[target_name]
#             return data,columns,target_name,target
 
#     def change_excel_to_pandas_unsupervised(path):
#         if path.split('.')[-1] == 'csv':
#             data = pd.read_csv(path)
#             columns = data.columns
#             return data,columns
#         elif path.split('.')[-1] == 'tsv':
#             data = pd.read_csv(path)
#             columns = data.columns
#             return data,columns
 
    #####

if __name__ ==  '__main__' :   
    if len(sys.argv) != 13:
        raise

    else :
        empty_list = ['None', '']

        input_path = sys.argv[1]
        output_path = sys.argv[2]

        target_col = [''] if sys.argv[3] in empty_list else sys.argv[3].split(',')        
        numerical_cols = [''] if sys.argv[4] in empty_list else sys.argv[4].split(',')
        categorical_cols = [''] if sys.argv[5] in empty_list else sys.argv[5].split(',')

        is_remove_duplicate = sys.argv[6]

        numerical_missing_method = sys.argv[7]
        numerical_missing_val = sys.argv[8]

        categorical_missing_method = sys.argv[9]
        categorical_missing_val = sys.argv[10]

        numerical_scaler = sys.argv[11]
        categorical_encoder = sys.argv[12]

    file_name = input_path.split('.')[0].split('/')[-1]

    processor = DataPreprocessing(input_path=input_path, output_path=output_path ,cur_path=output_path, file_name=file_name,
                                  target_col=target_col, categorical_cols=categorical_cols,
                                  numerical_cols=numerical_cols)

    preprocessed_data = processor.preprocess(is_remove_duplicate, numerical_missing_method,numerical_missing_val,
                                             categorical_missing_method,categorical_missing_val, numerical_scaler, categorical_encoder)