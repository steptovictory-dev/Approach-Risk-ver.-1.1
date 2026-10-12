"""Получить открытые источники геометрии с фиксацией происхождения"""
from pathlib import Path
import hashlib
import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / 'airports/LFPG'
DATA = BASE / 'data/geometry'
DATA.mkdir(parents=True, exist_ok=True)
records = []

def fetch(url, filename):
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Approach-Risk-Research'})
        with urllib.request.urlopen(request, timeout=35) as response:
            payload = response.read(20 * 1024 * 1024 + 1)
            if len(payload) > 20 * 1024 * 1024:
                raise ValueError('Превышен предел небольшого справочного файла')
            status = response.status
        (DATA / filename).write_bytes(payload)
        records.append({'url': url, 'file': filename, 'status': status,
                        'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
        print(filename, status, len(payload), flush=True)
        return payload
    except (urllib.error.URLError, TimeoutError) as error:
        records.append({'url': url, 'file': filename, 'status': str(error)})
        print(filename, str(error), flush=True)
        return None

# Проверяем официальный цикл, действовавший на дату пилота
for prefix in ['media/dvd', 'dvd']:
    url = f'https://www.sia.aviation-civile.gouv.fr/{prefix}/eAIP_26_DEC_2024/FRANCE/AIRAC-2024-12-26/html/eAIP/FR-AD-2.LFPG-fr-FR.html'
    payload = fetch(url, 'sia_2024-12-26_' + prefix.replace('/', '_') + '.html')
    if payload and b'LFPG' in payload:
        break

# История публичного справочника не является подтверждением действовавшей процедуры
url = 'https://github.com/davidmegginson/ourairports-data/commits/main/runways.csv?until=2025-01-15T00%3A00%3A00Z'
history_path = DATA / 'ourairports_history.html'
history = history_path.read_bytes() if history_path.is_file() else fetch(url, 'ourairports_history.html')
commit = None
commit_date = None
if history:
    # Отбираем по точному времени коммита, а не по дню в заголовке страницы
    match = re.search(rb'<script type="application/json" data-target="react-app.embeddedData">(.*?)</script>', history, re.S)
    if match:
        embedded = json.loads(match.group(1))
        cutoff = datetime(2025, 1, 15, tzinfo=timezone.utc)
        commits = [item for group in embedded['payload']['commitsRefRoute']['commitGroups']
                   for item in group['commits']]
        eligible = [item for item in commits if datetime.fromisoformat(item['committedDate']) <= cutoff]
        if eligible:
            selected = max(eligible, key=lambda item: datetime.fromisoformat(item['committedDate']))
            commit, commit_date = selected['oid'], selected['committedDate']
reference = commit or 'main'
for filename in ['runways.csv', 'airports.csv']:
    fetch(f'https://raw.githubusercontent.com/davidmegginson/ourairports-data/{reference}/{filename}', filename)
metar_url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=LFPG&data=metar&year1=2025&month1=1&day1=14&year2=2025&month2=1&day2=16&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=3&report_type=4'
fetch(metar_url, 'lfpg_metar_2025-01-14_16.csv')
report = {'checked_on': '2026-10-12', 'ourairports_reference': reference,
          'ourairports_commit_date': commit_date,
          'historical_commit_found': bool(commit), 'official_historical_validity_verified': False,
          'records': records}
(BASE / 'reports/03_geometry_sources.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
