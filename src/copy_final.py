import shutil
import os
from tqdm import tqdm

workdir = os.path.dirname(os.getcwd())
source_dir = '/project2/lhansen/pf_ms3/'
destination_dir = workdir + '/output/'

N = 100_000
T = 283

for i in tqdm(range(1,131)):
    print(i)
    case = 'actual data, seed = ' + str(i) + ', T = ' + str(T) + ', N = ' + str(N)
    casedir = destination_dir + case  + '/'
    try:
        os.mkdir(casedir)
        shutil.copy(source_dir + case  + '/X_0.pkl', casedir)
        shutil.copy(source_dir + case  + '/w_50.pkl', casedir)
        shutil.copy(source_dir + case  + '/X_50.pkl', casedir)
        shutil.copy(source_dir + case  + '/w_100.pkl', casedir)
        shutil.copy(source_dir + case  + '/X_100.pkl', casedir)
        shutil.copy(source_dir + case  + '/w_150.pkl', casedir)
        shutil.copy(source_dir + case  + '/X_150.pkl', casedir)
        shutil.copy(source_dir + case  + '/w_200.pkl', casedir)
        shutil.copy(source_dir + case  + '/X_200.pkl', casedir)
        shutil.copy(source_dir + case  + '/w_250.pkl', casedir)
        shutil.copy(source_dir + case  + '/X_250.pkl', casedir)
        shutil.copy(source_dir + case  + '/w_282.pkl', casedir)
        shutil.copy(source_dir + case  + '/X_282.pkl', casedir)
    except:
        pass
