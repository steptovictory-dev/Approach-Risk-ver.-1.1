"""Найти погожие дневные интервалы LFPG в запрошенных и зимних месяцах"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
import re
import urllib.request
import pandas as pd
import numpy as np
from metar import Metar

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data'
DATA.mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
YEAR=int(sys.argv[1]) if len(sys.argv)>1 else 2025
MONTHS=[1,2,3,7,10,12]

def fetch_month(month):
    path=DATA/f'metar_{YEAR}_{month:02}.csv'
    if not path.exists():
        following=pd.Timestamp(YEAR,month,1)+pd.offsets.MonthBegin(1)
        url=('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=LFPG&data=metar'
            f'&year1={YEAR}&month1={month}&day1=1&year2={following.year}&month2={following.month}&day2=1'
            '&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=3&report_type=4')
        with urllib.request.urlopen(url,timeout=120) as response:
            path.write_bytes(response.read())
    frame=pd.read_csv(path)
    frame['utc']=pd.to_datetime(frame.valid,utc=True).astype('datetime64[ns, UTC]')
    rows=[]
    for record in frame.itertuples():
        body=re.split(r'\s(?:TEMPO|BECMG|RMK)\s',record.metar)[0]
        row={'utc':record.utc,'metar':record.metar,'fair':False,'wind':np.nan,'direction':np.nan,
             'gust':np.nan,'visibility_m':np.nan,'ceiling_ft':np.nan,'temperature_c':np.nan}
        try:
            obs=Metar.Metar(body,month=month,year=YEAR,strict=True)
            wind=obs.wind_speed.value('KT') if obs.wind_speed else None
            gust=obs.wind_gust.value('KT') if obs.wind_gust else wind
            direction=obs.wind_dir.value() if obs.wind_dir else np.nan
            visibility=obs.vis.value('M') if obs.vis else None
            temperature=obs.temp.value('C') if obs.temp else None
            unknown_ceiling=any(cloud in ['BKN','OVC','VV'] and level is None for cloud,level,*rest in obs.sky)
            ceilings=[level.value('FT') for cloud,level,*rest in obs.sky
                      if cloud in ['BKN','OVC','VV'] and level is not None]
            ceiling=min(ceilings) if ceilings else np.inf
            cloud_known=('CAVOK' in body or bool(obs.sky) or bool(re.search(r'\b(?:NSC|NCD|CLR|SKC)\b',body)))
            adverse=bool(obs.weather) or bool(re.search(r'\b(?:WS|RE\w+)\b',body))
            fair=(wind is not None and gust is not None and visibility is not None
                and temperature is not None and wind<=15 and gust<=25 and visibility>=10000
                and -10<=temperature<=30 and ceiling>=3000 and cloud_known and not unknown_ceiling and not adverse)
            row.update({'fair':fair,'wind':wind,'direction':direction,'gust':gust,
                'visibility_m':visibility,'ceiling_ft':ceiling,'temperature_c':temperature})
        except Exception:
            pass
        rows.append(row)
    parsed=pd.DataFrame(rows).sort_values('utc').drop_duplicates('utc',keep='last')
    parsed.to_parquet(DATA/f'weather_{YEAR}_{month:02}.parquet',index=False)
    summaries=[]
    for day in pd.date_range(f'{YEAR}-{month:02}-01',periods=pd.Period(f'{YEAR}-{month:02}').days_in_month):
        start=(day+pd.Timedelta(hours=8)).tz_localize('Europe/Paris')
        end=(day+pd.Timedelta(hours=18)).tz_localize('Europe/Paris')
        clock=pd.DataFrame({'utc':pd.date_range(start.tz_convert('UTC'),end.tz_convert('UTC'),freq='5min',inclusive='left').astype('datetime64[ns, UTC]')})
        join=pd.merge_asof(clock,parsed.rename(columns={'utc':'observed'}),left_on='utc',right_on='observed',direction='backward',tolerance=pd.Timedelta(minutes=60))
        good=join.fair.fillna(False).astype(bool)
        east=join.direction.between(30,150)
        summaries.append({'date':day.strftime('%Y-%m-%d'),'month':month,'fair_fraction':good.mean(),
            'east_fair_fraction':(east & good).mean(),'wind_max_kt':join.loc[good,'wind'].max(),
            'temperature_min_c':join.loc[good,'temperature_c'].min(),
            'temperature_max_c':join.loc[good,'temperature_c'].max()})
    summary=pd.DataFrame(summaries)
    return summary.loc[summary.fair_fraction.eq(1)].sort_values(['east_fair_fraction','date'],ascending=[False,True])

with ThreadPoolExecutor(max_workers=4) as pool:
    outputs=list(pool.map(fetch_month,MONTHS))
candidates=pd.concat(outputs,ignore_index=True)
candidates.to_csv(DATA/f'weather_candidates_{YEAR}.csv',index=False)
print(candidates.to_string(index=False),flush=True)
print('Кандидатов по месяцам:',candidates.groupby('month').size().to_dict(),flush=True)
