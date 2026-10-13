"""Проверить погожие дни по фактическим заходам EDDF 07L"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import sys
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from archive_scan import scan
from lfpg_approach_audit import find_events
from lfpg_approach_audit import attach_qnh
from training_helpers import select_cohort

from download_fast import download


def analyze(date):
    output = BASE / 'data' / date
    if not (output / 'region_points.parquet').exists():
        cached = next((p for study in ['lfpg-seasonal-learning', 'egll-seasonal-learning'] if (p := PROJECT / 'research' / study / 'data' / date / 'archive' / 'download_manifest.json').exists()), PROJECT / 'research/lfpg-seasonal-learning/data' / date / 'archive' / 'download_manifest.json').parent
        manifest = cached / 'download_manifest.json'
        if manifest.exists():
            folder, entries = cached, json.loads(manifest.read_text(encoding='utf-8'))['files']
        else:
            folder, entries = download(date)
        scan(folder, BASE, date, manifest_entries=entries)
    points = pd.read_parquet(output / 'region_points.parquet')
    day = pd.Timestamp(date)
    start = (day + pd.Timedelta(hours=8)).tz_localize('Europe/Berlin').timestamp()
    end = (day + pd.Timedelta(hours=18)).tz_localize('Europe/Berlin').timestamp()
    points = points.loc[points.time.ge(start) & points.time.lt(end)].copy()
    points, _ = attach_qnh(points, BASE / 'data' / f'metar_{day.year}_{day.month:02}.csv', 364,
                           delay_minutes=15)
    geometry = pd.read_csv(PROJECT / 'airports/EDDF/reports/03_runway_geometry.csv')
    events, event_points = find_events(points, geometry)
    protocol = json.loads((PROJECT / 'configs/eddf_07l_training.json').read_text(encoding='utf-8'))
    eligible = events.direction.eq('arrival') & events.runway.eq('07L') & events.cross300_observed
    eligible &= events.cross300_gap_s.le(protocol['maximum_trace_gap_s'])
    eligible &= events.precontrol_max_gap_s.le(protocol['maximum_trace_gap_s'])
    eligible &= events.precontrol_span_s.ge(protocol['minimum_precontrol_span_s'])
    eligible &= ~events.parallel_ambiguity
    protocol['cohort_size'] = int(eligible.sum())
    selected = select_cohort(events, protocol)
    selected['split'] = 'external'
    selected.to_csv(output / 'selected_07l.csv', index=False)
    event_points.loc[event_points.event_id.isin(selected.event_id)].to_parquet(
        output / 'selected_07l_points.parquet', index=False)
    arrivals = events.loc[events.direction.eq('arrival')]
    result = {'date': date, 'arrivals_by_runway': arrivals.runway.value_counts().to_dict(),
              'eligible_07l': len(selected), 'enough_approaches': len(selected) >= 10}
    (output / 'day_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('DAY_RESULT', json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('dates', nargs='+')
    parser.add_argument('--workers', type=int, default=1, choices=[1, 2, 3])
    arguments = parser.parse_args()
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(analyze, arguments.dates))
    print('CHECKED_DAYS', json.dumps(results), flush=True)
