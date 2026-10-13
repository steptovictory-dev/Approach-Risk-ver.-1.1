"""Получить справочную геометрию EDDF отдельно от других аэропортов"""
from pathlib import Path
import io
import json
import sys
import urllib.request
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'scripts'))
from lfpg_approach_audit import runway_geometry


def run():
    output = PROJECT / 'airports/EDDF/reports'
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in ['airports', 'runways']:
        url = f'https://davidmegginson.github.io/ourairports-data/{name}.csv'
        with urllib.request.urlopen(url, timeout=120) as response:
            frame = pd.read_csv(io.BytesIO(response.read()), low_memory=False)
        local = frame.loc[frame['ident' if name == 'airports' else 'airport_ident'].eq('EDDF')].copy()
        local.to_csv(output / f'03_{name}_reference.csv', index=False)
        sources[name] = url
        if name == 'runways':
            active = local.loc[local.closed.eq(0) & local.he_ident.notna()]
            geometry = runway_geometry(active)
            assert {'07L', '25R', '07C', '25C', '07R', '25L'}.issubset(set(geometry.runway))
            geometry.to_csv(output / '03_runway_geometry.csv', index=False)
        else:
            assert int(local.elevation_ft.iloc[0]) == 364
    (output / '03_geometry_sources.json').write_text(json.dumps({
        'airport': 'EDDF', 'runway': '07L', 'sources': sources,
        'runway_confirmation': 'https://www.fraport.com/en/sustainability/dialog-with-neighbors/noise-and-air/flight-operations/runway-system-and-operating-hours.html',
        'research_proxy': True, 'historical_operational_applicability_verified': False}, indent=2) + '\n')


if __name__ == '__main__':
    run()
