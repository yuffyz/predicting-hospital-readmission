# -*- coding: utf-8 -*-

#!pip install category_encoders

# library imports
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
#import category_encoders as ce # categorical encoding

# setting display options for displaying dataframe
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

data = pd.read_csv("diabetic_data.csv")
df_clean = pd.DataFrame()

# Changing 3 class labels to two main classes
# 0 : 'NO' and '>30'
# 1 : '<30'
data = data.replace({'readmitted': 'NO'}, 0)
data = data.replace({'readmitted': '>30'}, 0)
data = data.replace({'readmitted': '<30'}, 1)

# dropping column with most amount of missing values
# Keeping encounter_id and patient_nbr for reference
cols = ['weight', 'payer_code', 'medical_specialty']
data.drop(cols, axis=1, inplace=True)

# Mode imputation is used for Missing Values in Race Column
max_race_count = max(data.race.value_counts())
max_race = data.race.value_counts()[data.race.value_counts()==max_race_count].index[0]
data.race.replace('?', max_race, inplace=True)

# Beginning building of our clean dataframe
df_clean['encounter_id'] = data.encounter_id
df_clean['patient_id'] = data.patient_nbr

# race (one hot encode them)
df_clean['race_white'] = [1 if x=='Caucasian' else 0 for x in data.race]
df_clean['race_aa'] = [1 if x=='AfricanAmerican' else 0 for x in data.race]
df_clean['race_hispanic'] = [1 if x=='Hispanic' else 0 for x in data.race]
#df_clean['race_other'] = [1 if x=='Other' else 0 for x in data.race]
df_clean['race_asian'] = [1 if x=='Asian' else 0 for x in data.race]

# gender (move unknown into Female based on Mode Imputation)
#gender_code = {'Female':0,'Male':1,'Unknown/Invalid':0}
#data.gender = [gender_code[x] for x in data.gender]
#df_clean['is_female'] = [1 if x=='Female' else 0 for x in data.gender]
df_clean['is_male'] = [1 if x == 'Male' else 0 for x in data.gender]

# age (combine ages into three bins)
youth = ['[0-10)','[10-20)']
adult = ['[20-30)','[30-40)','[40-50)']
elderly = ['[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
age_codes = []
for x in data.age:
    if x in youth:
        age_codes.append(0)
    elif x in adult:
        age_codes.append(1)
    else:
        age_codes.append(2)
data.age = age_codes
#df_clean['age_youth'] = [1 if x==0 else 0 for x in data.age]
#df_clean['age_adult'] = [1 if x==1 else 0 for x in data.age]
#df_clean['age_elderly'] = [1 if x==2 else 0 for x in data.age]
df_clean['age'] = data.age

# admission type id, one hot encode, combine several categories into one.
df_clean['admission_emergency_urgent'] = [1 if x in (1,2) else 0 for x in data.admission_type_id]
df_clean['admission_elective'] = [1 if x==3 else 0 for x in data.admission_type_id]
df_clean['admission_newborn'] = [1 if x==4 else 0 for x in data.admission_type_id]
df_clean['admission_trauma'] = [1 if x==7 else 0 for x in data.admission_type_id]
df_clean['admission_unknown'] = [1 if x in (5,6,8) else 0 for x in data.admission_type_id]

"""Collapsed Multiple discharge_disposition_ids into fewer categories:<br>
discharge_disposition_unknown: 18, 25, 26<br>
discharge_disposition_expired: 11, 19, 20, 21<br>
discharge_disposition_home_other_facility: 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 16, 22, 23, 24, 27, 28, 29, 30<br>
discharge_disposition_current_hospital_patient: 9, 12, 15, 17
"""

# discharge disposition id, one hot encode
df_clean['discharge_diposition_unknown'] = [1 if x in (18, 25, 26) else 0 for x in data.discharge_disposition_id]
df_clean['discharge_diposition_expired'] = [1 if x in (11, 19, 20, 21) else 0 for x in data.discharge_disposition_id]
df_clean['discharge_diposition_home_other_facility'] = [1 if x in (1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 16, 22, 23, 24, 27, 28, 29, 30) else 0 for x in data.discharge_disposition_id]
df_clean['discharge_diposition_current_hospital_patient'] = [1 if x in (9, 12, 15, 17) else 0 for x in data.discharge_disposition_id]

# admission_source_id, one hot encode
# Collapsing multiple categories into one - if all 0, assumed "Other" Category
df_clean['admission_source_emergency_room'] = [1 if x==7 else 0 for x in data.admission_source_id]
df_clean['admission_source_physician_referral'] = [1 if x==1 else 0 for x in data.admission_source_id]
df_clean['admission_source_transfer_hospital_health_care_facility_clinic'] = [1 if x in (4,6,2) else 0 for x in data.admission_source_id]

# time_in_hospital
for i, row in data.time_in_hospital.iteritems():
    if row <= 4:
        data.at[i, 'time_in_hospital'] = 0
    if (row > 4) & (row <=9):
        data.at[i, 'time_in_hospital'] = 1
    if (row > 9):
        data.at[i, 'time_in_hospital'] = 2
df_clean['time_in_hospital'] = data['time_in_hospital']

# num_lab_procedures
for i, row in data.num_lab_procedures.iteritems():
    if row <= 30:
        data.at[i, 'num_lab_procedures'] = 0
    if (row > 30) & (row <=60):
        data.at[i, 'num_lab_procedures'] = 1
    if (row > 60):
        data.at[i, 'num_lab_procedures'] = 2
df_clean['num_lab_procedures'] = data['num_lab_procedures']

# num_procedures
for i, row in data.num_procedures.iteritems():
    if row <= 2:
        data.at[i, 'num_procedures'] = 0
    if (row > 2) & (row <= 4):
        data.at[i, 'num_procedures'] = 1
    if (row > 4):
        data.at[i, 'num_procedures'] = 2
df_clean['num_procedures'] = data['num_procedures']

# num_medications
for i, row in data.num_medications.iteritems():
    if row <= 20:
        data.at[i, 'num_medications'] = 0
    if (row > 20) & (row <= 30):
        data.at[i, 'num_medications'] = 1
    if (row > 30):
        data.at[i, 'num_medications'] = 2
df_clean['num_medications'] = data['num_medications']

# 'number_outpatient', 'number_emergency', 'number_inpatient'
# since they are visits, and may have large effect, want to keep it as is.
#df_clean['number_outpatient'] = data['number_outpatient']
#df_clean['number_emergency'] = data['number_emergency']
#df_clean['number_inpatient'] = data['number_inpatient']

# alternative coding #
df_clean['number_outpatient'] = [0 if x==0 else 1 for x in data['number_outpatient']]
df_clean['number_emergency'] = [0 if x==0 else 1 for x in data['number_emergency']]
inpatient_code = {0:0,1:1}
df_clean['number_inpatient'] = [inpatient_code[x] if x in inpatient_code.keys() else 2 for x in data['number_inpatient']]


"""Cleaning and Categorizing diag_1 column<br>
Ignoring diag_2 and diag_3 for now, since they have more missing values
"""

col = data['diag_1']
data.insert(data.columns.get_loc('diag_1') + 1, 'numeric_diag_1', col, allow_duplicates=True)

# temporarily inputting -1 in numeric_diag columns for codes with 'V' or 'E'
data.loc[data['numeric_diag_1'].str.contains("V|E")==True, ['numeric_diag_1']] = '-1'

# replace '?' with NaN in numeric_diag columns
data = data.replace({'numeric_diag_1': '?'}, np.nan)

#converting numeric_diag columns to numeric datatype
data['numeric_diag_1'] = pd.to_numeric(data.numeric_diag_1)

# creating duplicate numeric_diag_1 column as category_diag_1
col = data['numeric_diag_1']
data.insert(data.columns.get_loc('numeric_diag_1') + 1, 'category_diag_1', col, allow_duplicates=True)

"""**ICD-9 Code Categories: https://icd.codes/icd9cm**

1) 001-139 : Infectious And Parasitic Diseases **= Infection**<br>
2) 140-239 : Neoplasms ** = Neoplasms**<br>
3) 240-279 : Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders ** = Endocrine**<br>
4) 280-289 : Diseases Of The Blood And Blood-Forming Organs ** = Blood**<br>
5) 290-319 : Mental Disorders ** = Mental**<br>
6) 320-389 : Diseases Of The Nervous System And Sense Organs ** = Nervous**<br>
7) 390-459 : Diseases Of The Circulatory System ** = Circulatory**<br>
8) 460-519 : Diseases Of The Respiratory System ** = Respiratory**<br>
9) 520-579 : Diseases Of The Digestive System ** = Digestive**<br>
10) 580-629 : Diseases Of The Genitourinary System ** = Genitourinary**<br>
11) 630-679 : Complications Of Pregnancy, Childbirth, And The Puerperium ** = Pregnancy_Childbirth**<br>
12) 680-709 : Diseases Of The Skin And Subcutaneous Tissue ** = Skin**<br>
13) 710-739 : Diseases Of The Musculoskeletal System And Connective Tissue ** = Musculoskeletal**<br>
14) 740-759 : Congenital Anomalies ** = Congenital**<br>
15) 760-779 : Certain Conditions Originating In The Perinatal Period ** = Perinatal_Condition**<br>
16) 780-799 : Symptoms, Signs, And Ill-Defined Conditions ** = Symptoms**<br>
17) 800-999 : Injury And Poisoning ** = Injury_Poisoning**<br>
18) V01-V91 : Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services ** = Additional_Factors**<br>
19) E000-E999 : Supplementary Classification Of External Causes Of Injury And Poisoning ** = External_Cause**
"""

# Encoding the categories for value range in ICD-9 Diagnosis Codes
def encode_diagnosis_categories(diag_col_name, data):
    temp_data = data
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(0, 140, inclusive=False), 
                                    1001, data[diag_col_name]) #'Infection', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(139, 240, inclusive=False), 
                                    1002, data[diag_col_name]) #'Neoplasms', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(239, 280, inclusive=False), 
                                    1003, data[diag_col_name]) #'Endocrine', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(279, 290, inclusive=False), 
                                    1004, data[diag_col_name]) #'Blood', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(289, 320, inclusive=False), 
                                    1005, data[diag_col_name]) #'Mental', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(319, 390, inclusive=False), 
                                    1006, data[diag_col_name]) #'Nervous', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(389, 460, inclusive=False), 
                                    1007, data[diag_col_name]) #'Circulatory', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(459, 520, inclusive=False), 
                                    1008, data[diag_col_name]) #'Respiratory', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(519, 580, inclusive=False), 
                                    1009, data[diag_col_name]) #'Digestive', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(579, 630, inclusive=False), 
                                    1010, data[diag_col_name]) #'Genitourinary', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(629, 680, inclusive=False), 
                                    1011, data[diag_col_name]) #'Pregnancy_Chidlbirth', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(679, 710, inclusive=False), 
                                    1012, data[diag_col_name]) #'Skin', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(709, 740, inclusive=False), 
                                    1013, data[diag_col_name]) #'Musculoskeletal', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(739, 760, inclusive=False), 
                                    1014, data[diag_col_name]) #'Congenital', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(759, 780, inclusive=False), 
                                    1015, data[diag_col_name]) #'Perinatal_Condition', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(779, 800, inclusive=False), 
                                    1016, data[diag_col_name]) #'Symptoms', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name].between(799, 1000, inclusive=False), 
                                    1017, data[diag_col_name]) #'Injury_Poisoning', data[diag_col_name])
    temp_data[diag_col_name] = np.where(temp_data[diag_col_name] == -1, 
                                    1018, data[diag_col_name]) #'Additional_Factors/External_Cause', data[diag_col_name])
    return temp_data

cols = ['category_diag_1']
data = encode_diagnosis_categories(cols[0], data)
num_vals = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 
            1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018]
string_vals = ['Infection', 'Neoplasms', 'Endocrine', 'Blood', 'Mental', 'Nervous', 'Circulatory', 'Respiratory', 'Digestive', 'Genitourinary',
              'Pregnancy_Childbirth', 'Skin', 'Musculoskeletal', 'Congenital', 'Perinatal_Condition', 'Symptoms', 'Injury_Poisoning', 'Additional_Factors/External_Cause']
for i in range(len(num_vals)):
    data = data.replace({cols[0]: num_vals[i]}, string_vals[i]) #data.replace({col: 1001}, 'Infection')
data = data.replace({cols[0]: np.nan}, "Unknown")

# Collapsing Multiple diag_1 categories into main categories
diag_categories = ['Circulatory', 'Endocrine', 'Respiratory', 'Digestive', 
                   'Symptoms', 'Injury_Poisoning', 'Genitourinary', 'Musculoskeletal', 'Neoplasms', 'Other']
def one_hot_encode(string_value):
    return([1 if x==string_value else 0 for x in data.category_diag_1])
df_clean['diag1_circulatory'] = one_hot_encode('Circulatory')
df_clean['diag1_endocrine'] = one_hot_encode('Endocrine')
df_clean['diag1_respiratory'] = one_hot_encode('Respiratory')
df_clean['diag1_digestive'] = one_hot_encode('Digestive')
df_clean['diag1_symptoms'] = one_hot_encode('Symptoms')
df_clean['diag1_injury_poisoning'] = one_hot_encode('Injury_Poisoning')
df_clean['diag1_genitourinary'] = one_hot_encode('Genitourinary')
df_clean['diag1_musculoskeletal'] = one_hot_encode('Musculoskeletal')
df_clean['diag1_neoplasms'] = one_hot_encode('Neoplasms')
df_clean['diag1_other'] = [1 if x not in diag_categories else 0 for x in data.category_diag_1]

# number_diagnoses
for i, row in data.number_diagnoses.iteritems():
    if row <= 4:
        data.at[i, 'number_diagnoses'] = 0
    if (row > 4) & (row <= 6):
        data.at[i, 'number_diagnoses'] = 1
    if (row > 6) & (row <= 8):
        data.at[i, 'number_diagnoses'] = 2
    if (row > 8):
        data.at[i, 'number_diagnoses'] = 3
df_clean['number_diagnoses'] = data['number_diagnoses']

# max_glu_serum
max_glu_serum_code = {'None':0,'Norm':1,'>200':2,'>300':3}
data.max_glu_serum = [max_glu_serum_code[x] for x in data.max_glu_serum]
df_clean['max_glu_serum'] = data.max_glu_serum

# A1Cresult
A1Cresult_code = {'None':0,'Norm':1,'>7':2,'>8':3}
data.A1Cresult = [A1Cresult_code[x] for x in data.A1Cresult]
df_clean['A1Cresult'] = data.A1Cresult

# For the 23 Medicine Columns - Creating a two new column 
# num_medicine_change and num_total_medicine
medicine_col_names = data.columns[23:46]
medicine_df = data[medicine_col_names]
num_medicine_change_series = medicine_df.isin(['Up', 'Down']).sum(1)
num_total_medicine_series = medicine_df.isin(['Up', 'Down', 'Steady']).sum(1)

data.insert(data.columns.get_loc('metformin-pioglitazone') + 1, 'num_total_medicine', num_total_medicine_series, allow_duplicates=True)

data.insert(data.columns.get_loc('num_total_medicine') + 1, 'num_medicine_change', num_medicine_change_series, allow_duplicates=True)

# Encoding the Medicines Columns - only considering columns that have considerable amount of important data
# 14 medicine columns for now
#metformin_code = {'No':0,'Steady':2,'Up':3,'Down':1}
metformin_code = {'No':0,'Steady':1,'Up':2,'Down':3}

data.metformin = [metformin_code[x] for x in data.metformin]
df_clean['metformin'] = data.metformin
data.repaglinide = [metformin_code[x] for x in data.repaglinide]
df_clean['repaglinide'] = data.repaglinide
data.nateglinide = [metformin_code[x] for x in data.nateglinide]
df_clean['nateglinide'] = data.nateglinide
data.chlorpropamide = [metformin_code[x] for x in data.chlorpropamide]
df_clean['chlorpropamide'] = data.chlorpropamide
data.glimepiride = [metformin_code[x] for x in data.glimepiride]
df_clean['glimepiride'] = data.glimepiride
data.glipizide = [metformin_code[x] for x in data.glipizide]
df_clean['glipizide'] = data.glipizide
data.glyburide = [metformin_code[x] for x in data.glyburide]
df_clean['glyburide'] = data.glyburide
data.pioglitazone = [metformin_code[x] for x in data.pioglitazone]
df_clean['pioglitazone'] = data.pioglitazone
data.rosiglitazone = [metformin_code[x] for x in data.rosiglitazone]
df_clean['rosiglitazone'] = data.rosiglitazone
data.acarbose = [metformin_code[x] for x in data.acarbose]
df_clean['acarbose'] = data.acarbose 
data.miglitol = [metformin_code[x] for x in data.miglitol]
df_clean['miglitol'] = data.miglitol
data.tolazamide = [metformin_code[x] for x in data.tolazamide]
df_clean['tolazamide'] = data.tolazamide
data.insulin = [metformin_code[x] for x in data.insulin]
df_clean['insulin'] = data.insulin
data['glyburide-metformin'] = [metformin_code[x] for x in data['glyburide-metformin']]
df_clean['glyburide_metformin'] = data['glyburide-metformin']

# adding two new columns to clean dataframe - num_medicine_change and num_total_medicine
df_clean['num_medicine_change'] = data['num_medicine_change']
df_clean['num_total_medicine'] = data['num_total_medicine']

change_code = {'No':0,'Ch':1}
data.change = [change_code[x] for x in data.change]
df_clean['change_in_medications'] = data.change

diabetesMed_code = {'No':0,'Yes':1}
data.diabetesMed = [diabetesMed_code[x] for x in data.diabetesMed]
df_clean['diabetesMed'] = data.diabetesMed

# adding new column called category_diag_1
# this has the string value of diag_1 category
diag_categories = ['Circulatory', 'Endocrine', 'Respiratory', 'Digestive', 
                   'Symptoms', 'Injury_Poisoning', 'Genitourinary', 'Musculoskeletal', 'Neoplasms']
data.loc[~data.category_diag_1.isin(diag_categories), 'category_diag_1'] = 'Other'

col = data['category_diag_1']
df_clean.insert(df_clean.columns.get_loc('number_inpatient') + 1, 'category_diag_1', col, allow_duplicates=True)

df_clean['readmitted'] = data['readmitted']

print(len(df_clean.columns)-1, "features after cleaning the input data!")

#print("Sample preview of Cleaned Data:\n")
#print(df_clean.head(3))

df_clean.to_csv("clean_diabetic_data.csv", index=False)
print("\nCleaned data frame is exported to the current directory\n")