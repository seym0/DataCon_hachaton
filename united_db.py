import pandas as pd
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def db1():
  data = pd.read_csv("https://raw.githubusercontent.com/dataconHack/hackathon/main/data.csv")

  data.loc[(data.ZOI_drug_NP =='32+'),'ZOI_drug_NP'] = '32'
  data.loc[(data.ZOI_drug_NP =='17+2'),'ZOI_drug_NP'] = '19'
  data.loc[(data.ZOI_drug =='32+'),'ZOI_drug'] = '32'
  data.loc[(data.ZOI_NP =='50+'),'ZOI_NP'] = '50'

  data = data.drop_duplicates()
  data = data.drop(columns=['Unnamed: 0.1','Unnamed: 0','ZOI_NP', 'NP size_avg','fold_increase_in_antibacterial_activity (%)' ])
  data = data.dropna(subset=['ZOI_drug_NP'])

  data['ZOI_drug'] = data['ZOI_drug'].astype('float64')
  data['ZOI_drug_NP'] = data['ZOI_drug_NP'].astype('float64')

  for i in range(len(data.NP_concentration)):
    if '/' not in str(data.NP_concentration.loc[i]):
      data.NP_concentration.loc[i] = float(data.NP_concentration.loc[i])
    else:
      data = data.drop(labels = [i],axis = 0)

  data['NP_concentration'] = data['NP_concentration'].astype('float64')

  data = data.fillna({'ZOI_drug' : data.ZOI_drug.mean()})
  data = data.fillna({'Drug_dose' : data.Drug_dose.mean()})
  data = data.fillna({'NP_concentration' : data.NP_concentration.mean()})

  data.loc[(data.Bacteria =='Enterobacter cloacae '),'Bacteria'] = 'Enterobacter cloacae'
  data.loc[(data.Bacteria =='Actinobacillus pleuropneumoniae '),'Bacteria'] = 'Actinobacillus pleuropneumoniae'
  data.loc[(data.Bacteria =='Salmonella  typhi'),'Bacteria'] = 'Salmonella typhimurium'
  data.loc[(data.Bacteria =='Salmonella typhimurium '),'Bacteria'] = 'Salmonella typhimurium'
  data.loc[(data.Bacteria =='Acinetobacter baumanii'),'Bacteria'] = 'Acinetobacter baumannii'

  data = data[~((data.Bacteria=='Candida albicans') | (data.Bacteria=='Candida saitoana'))]
  data = data[~((data.Bacteria=='Salmonella Paratyphi') | (data.Bacteria=='Bacillus spp.        ') | (data.Bacteria=='Candida glochares') | (data.Bacteria=='Candida glabrata'))]

  data.to_csv(str(os.getcwd() + "/db1.csv"))

def db2():
  bacterial = pd.read_csv("https://raw.githubusercontent.com/dataconHack/hackathon/main/bacterial_descriptors.csv")

  bacterial = bacterial[~(bacterial['kingdom'] =='Fungi')]
  bacterial = bacterial[(bacterial.isolated_from!='blood') & (bacterial.Bacteria !='Salmonella typhimurium')]
  bacterial.loc[(bacterial.Bacteria =='Salmonella typhimurium '),'Bacteria'] = 'Salmonella typhimurium'
  bacterial = bacterial.drop(columns=['Tax_id','kingdom', 'subkingdom', 'clade','min_Incub_period, h', 'max_Incub_period, h'])

  bacterial.to_csv(str(os.getcwd() + "/db2.csv"))

def db3():
  drug = pd.read_csv("https://raw.githubusercontent.com/dataconHack/hackathon/main/drug_descriptors.csv")

  drug = drug.drop(columns=['prefered_name','chemID','Unnamed: 0'])
  drug.loc[ len(drug.index )] = ['Neomycin', 'CC1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)N)O)N)O)N.C[C@H]3[C@H]([C@H](N)[C@@H](C)O3)O[C@@H]4[C@H](O[C@H]([C@@H]([C@H]4O)O)N5C=CC(=O)NC5=O)N']

  descriptor_list = Descriptors.descList
  descriptors = []

  for descriptor in descriptor_list:
      descriptors.append(descriptor[0])

  def get_descriptor_values(mol, descriptors):
    calc = MolecularDescriptorCalculator(descriptors)
    ds = calc.CalcDescriptors(mol)
    return ds[0]

  for i in descriptors:
    drug[i] = pd.Series(np.array([get_descriptor_values(Chem.MolFromSmiles(j), [i]) for j in drug["smiles"]]), index=drug.index)

  drug = drug.rename(columns = {'drug':'Drug'})

  for i in list(drug.columns[drug.isna().any()]):
    drug = drug.fillna({i : drug[i].mean()})

  def fingerprints():
    result = []
    for i in drug.smiles:
      result.append(list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i), 2, nBits=512)))
    result = np.array(result)
    result = result.T

    ind=0
    for i in result:
      drug[f'finger_print{str(ind)}'] = i
      ind+=1
  fingerprints()

  drug.to_csv(str(os.getcwd() + "/db3.csv"))


def general_db():
  db_1 = pd.read_csv(str(os.getcwd()) + "/db1.csv")
  db_2 = pd.read_csv(str(os.getcwd()) + "/db2.csv")
  db_3 = pd.read_csv(str(os.getcwd()) + "/db3.csv")

  general_data = pd.merge(db_1, db_2, on='Bacteria')
  general_data = pd.merge(general_data, db_3, on='Drug')
  general_data.to_csv(str(os.getcwd() +'/general_data.csv'))


def final_cleaning():
  data = pd.read_csv(str(os.getcwd() + '/general_data.csv'))

  data = data.drop(columns=['Unnamed: 0'])

  descriptor_list = Descriptors.descList
  descriptors = []

  for descriptor in descriptor_list:
    descriptors.append(descriptor[0])

  corr_matrix = data[descriptors].corr().abs()
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
  to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

  data.drop(to_drop, axis=1, inplace=True)

  cat_cols = list(data.select_dtypes(include=['object', 'category']).columns)
  #cat_cols = ['Bacteria', 'NP_Synthesis', 'Drug', 'Drug_class_drug_bank', 'shape', 'method', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'gram', 'isolated_from', 'smiles']

  for col in cat_cols:
    data[col] = data[col].astype('category').cat.codes

  for i in list(data.head(0)):
    if len(data[i].unique()) == 1:
      data.drop([i], axis=1, inplace=True)

  data = data.drop(['Bacteria', 'smiles'], axis=1)
  data.drop(['species', 'class'], axis=1, inplace=True)
  data.drop(['Unnamed: 0.1', 'Unnamed: 0_x', 'Unnamed: 0_y'], axis=1, inplace=True)
  data.to_csv(str(os.getcwd()) + "/final.csv")


db1()
print("the database1 has been processed")
db2()
print("the database2 has been processed")
db3()
print("the database3 has been processed")
general_db()
print("the general_database has been processed")
final_cleaning()
print("the final_database has been processed")