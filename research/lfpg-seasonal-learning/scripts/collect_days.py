"""Проверить погожие дни по фактическим заходам LFPG 09L"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import sys
import urllib.request
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from lfpg_archive_pilot import scan
from lfpg_approach_audit import attach_qnh, find_events
from lfpg_training import select_cohort


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'Approach-Risk-Research', **(headers or {})}), timeout=120)


def download(date):
    folder = BASE / 'data' / date / 'archive'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'download_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if all(Path(item['path']).exists() and Path(item['path']).stat().st_size == item['bytes']
               for item in saved['files']):
            return folder, saved['files']
    tag = 'v' + date.replace('-', '.') + '-planes-readsb-prod-0'
    release_url = f'https://api.github.com/repos/adsblol/globe_history_{date[:4]}/releases/tags/{tag}'
    with request(release_url) as response:
        release = json.load(response)
    assets = sorted([a for a in release['assets'] if '.tar' in a['name']], key=lambda a: a['name'])
    assert assets, 'В суточном выпуске нет архива'

    def part(asset):
        path = folder / asset['name']
        if not path.exists() or path.stat().st_size != asset['size']:
            offset = path.stat().st_size if path.exists() else 0
            if offset >= asset['size']:
                offset = 0
            with request(asset['browser_download_url'],
                         {'Range': f'bytes={offset}-'} if offset else None) as response:
                resume = offset > 0 and response.status == 206
                if resume:
                    assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                copied = offset if resume else 0
                next_message = copied + 512 * 1024 * 1024
                if resume:
                    print('RESUME', date, path.suffix, round(offset / 1e9, 2), 'GB', flush=True)
                stream = path.open('ab' if resume else 'wb')
                try:
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
                        copied += len(chunk)
                        if copied >= next_message:
                            print(date, path.suffix, round(copied / 1e9, 2), 'GB', flush=True)
                            next_message += 512 * 1024 * 1024
                finally:
                    stream.close()
            assert path.stat().st_size == asset['size']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if asset.get('digest'):
            assert 'sha256:' + digest == asset['digest']
        return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest,
                'url': asset['browser_download_url']}

    with ThreadPoolExecutor(max_workers=2) as pool:
        entries = list(pool.map(part, assets))
    manifest.write_text(json.dumps({'release_url': release_url, 'files': entries}, indent=2),
                        encoding='utf-8')
    return folder, entries


def analyze(date):
    output = BASE / 'data' / date
    if not (output / 'region_points.parquet').exists():
        folder, entries = download(date)
        scan(folder, BASE, date, manifest_entries=entries)
    points = pd.read_parquet(output / 'region_points.parquet')
    day = pd.Timestamp(date)
    start = (day + pd.Timedelta(hours=8)).tz_localize('Europe/Paris').timestamp()
    end = (day + pd.Timedelta(hours=18)).tz_localize('Europe/Paris').timestamp()
    points = points.loc[points.time.ge(start) & points.time.lt(end)].copy()
    points, _ = attach_qnh(points, BASE / 'data' / f'metar_{day.year}_{day.month:02}.csv', 392,
                           delay_minutes=15)
    geometry = pd.read_csv(PROJECT / 'airports/LFPG/reports/03_runway_geometry.csv')
    events, event_points = find_events(points, geometry)
    protocol = json.loads((PROJECT / 'configs/lfpg_09l_training.json').read_text(encoding='utf-8'))
    eligible = events.direction.eq('arrival') & events.runway.eq('09L') & events.cross300_observed
    eligible &= events.cross300_gap_s.le(protocol['maximum_trace_gap_s'])
    eligible &= events.precontrol_max_gap_s.le(protocol['maximum_trace_gap_s'])
    eligible &= events.precontrol_span_s.ge(protocol['minimum_precontrol_span_s'])
    eligible &= ~events.parallel_ambiguity
    protocol['cohort_size'] = int(eligible.sum())
    selected = select_cohort(events, protocol)
    selected['split'] = 'external'
    selected.to_csv(output / 'selected_09l.csv', index=False)
    event_points.loc[event_points.event_id.isin(selected.event_id)].to_parquet(
        output / 'selected_09l_points.parquet', index=False)
    arrivals = events.loc[events.direction.eq('arrival')]
    result = {'date': date, 'arrivals_by_runway': arrivals.runway.value_counts().to_dict(),
              'eligible_09l': len(selected), 'enough_approaches': len(selected) >= 15}
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
