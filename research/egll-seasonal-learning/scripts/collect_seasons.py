"""Собрать погожие дни EGLL, повторно используя готовые суточные архивы"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,pandas as pd
from collect_days import BASE,PROJECT,analyze
from prepare_examples import prepare

def collect(season):
 c=pd.read_csv(BASE/'data/weather_candidates_2025.csv');c=c.loc[c.season.eq(season)].copy()
 c['cached']=c.date.map(lambda d:(PROJECT/'research/lfpg-seasonal-learning/data'/d/'archive/download_manifest.json').exists())
 c=c.sort_values(['cached','east_fair_fraction','date'],ascending=[False,False,True])
 ready=[]
 for date in c.date:
  folder=BASE/'data'/date
  try:
   if not (folder/'examples.parquet').exists():
    if not (folder/'day_result.json').exists():analyze(date)
    result=prepare(date)
   else:
    f=pd.read_parquet(folder/'examples.parquet');result={'date':date,'qualified_09l':int(f.event_id.nunique()),'examples':len(f)}
   if result['qualified_09l']>=10 and result['examples']>=300:ready.append(result)
   (BASE/'data'/f'{season}_collection.json').write_text(json.dumps({'season':season,'ready':ready},indent=2))
   print('PROGRESS',season,len(ready),'/5',date,result,flush=True)
   if len(ready)==5:return ready
  except Exception as e:
   print('DAY_ERROR',season,date,repr(e),flush=True)
 raise RuntimeError('Не хватает дней '+season)

if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=3) as pool: list(pool.map(collect,['spring','summer','autumn']))
 from train_models import run
 run()
 import build_days_notebook
 import build_training_notebook
 print('EGLL_DONE',flush=True)
