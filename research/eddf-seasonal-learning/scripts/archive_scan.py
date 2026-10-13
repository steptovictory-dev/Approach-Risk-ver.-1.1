"""Извлечение траекторий района EDDF из суточного архива ADSB.lol"""
from pathlib import Path
import gzip
import hashlib
import io
import json
import tarfile
import pandas as pd
import orjson

class Parts(io.RawIOBase):
    # Читаем части TAR как один поток с ограниченным размером чтения
    def __init__(self, paths):
        self.streams = [path.open('rb') for path in paths]
        self.index = 0
    def readable(self):
        return True
    def read(self, size=-1):
        if size < 0:
            raise ValueError('Необходимо ограниченное чтение')
        result = bytearray()
        while len(result) < size and self.index < len(self.streams):
            chunk = self.streams[self.index].read(size - len(result))
            if chunk:
                result.extend(chunk)
            else:
                self.index += 1
        return bytes(result)
    def close(self):
        for stream in self.streams:
            stream.close()
        super().close()

def scan(old_root, base, date='2025-01-15', manifest_entries=None):
    output = base / 'data' / date
    raw = output / 'raw_candidates'
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = old_root / 'docs/sources/download_manifest.json'
    if manifest_entries is None:
        original = json.loads(manifest_path.read_text(encoding='utf-8'))
        entries = [item for item in original['files'] if f'usa/{date}/' in item['path']]
        assert len(entries) == 2
    else:
        entries = manifest_entries
        assert entries
    paths = [old_root / item['path'] for item in entries]
    verified = []
    for path, item in zip(paths, entries):
        # Проверяем полные части архива до пространственного отбора
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert path.stat().st_size == item['bytes']
        assert digest == item['sha256'], 'Изменился исходный архив'
        verified.append({'path': item['path'], 'bytes': path.stat().st_size,
                         'sha256': digest, 'url': item['url']})
    rows, traces = [], []
    checked = 0
    with Parts(paths) as stream, tarfile.open(fileobj=stream, mode='r|', bufsize=1024 * 1024) as archive:
        for member in archive:
            if not member.isfile() or '/traces/' not in '/' + member.name or 'trace_full_' not in member.name:
                continue
            assert member.size <= 20 * 1024 * 1024, 'Необычно большой объект'
            content = archive.extractfile(member).read()
            payload = gzip.decompress(content) if content.startswith(b'\x1f\x8b') else content
            try:
                data = orjson.loads(payload)
            except orjson.JSONDecodeError:
                data = json.loads(payload)
            checked += 1
            points = data.get('trace', [])
            # Прямоугольник служит поиском, а не правилом принадлежности аэропорту
            local = [point for point in points if len(point) >= 8
                     and isinstance(point[1], (int, float))
                     and isinstance(point[2], (int, float))
                     and 49.75 <= point[1] <= 50.3 and 8.15 <= point[2] <= 9.05]
            low = [point for point in local if isinstance(point[3], (int, float))
                   and point[3] <= 10000]
            if low:
                # Сохраняем исходный объект целиком и все точки в области поиска
                identifier = hashlib.sha256(member.name.encode()).hexdigest()[:16]
                raw_name = identifier + '.json.gz'
                if content.startswith(b'\x1f\x8b'):
                    (raw / raw_name).write_bytes(content)
                else:
                    (raw / raw_name).write_bytes(gzip.compress(content, mtime=0))
                traces.append({'trace_id': identifier, 'icao24': data.get('icao'),
                               'typecode': data.get('t'), 'member': member.name,
                               'member_sha256': hashlib.sha256(content).hexdigest(),
                               'raw_copy': f'raw_candidates/{raw_name}',
                               'region_rows': len(local), 'low_rows': len(low)})
                for point in local:
                    flags = int(point[6] or 0)
                    rows.append({'trace_id': identifier, 'icao24': data.get('icao'),
                        'time': data['timestamp'] + point[0], 'lat': point[1], 'lon': point[2],
                        'altitude_raw': point[3], 'altitude_ft': point[3] if isinstance(point[3], (int, float)) else None,
                        'ground': point[3] == 'ground', 'speed_knots': point[4],
                        'track_deg': point[5], 'flags': flags, 'stale_position': bool(flags & 1),
                        'geometric_altitude': bool(flags & 8), 'geometric_rate': bool(flags & 4),
                        'vertical_rate_fpm': point[7], 'position_source': point[9] if len(point) > 9 else None,
                        'geoaltitude_ft': point[10] if len(point) > 10 else None,
                        'ias_knots': point[12] if len(point) > 12 else None})
            if checked % 5000 == 0:
                print(f'Проверено {checked} треков; низких кандидатов {len(traces)}', flush=True)
    assert rows, 'Низкие пространственные кандидаты не найдены'
    frame = pd.DataFrame(rows)
    # Смешанное поле сохраняем строкой; числовая высота хранится отдельно
    frame['altitude_raw'] = frame.altitude_raw.map(lambda value: json.dumps(value))
    frame.to_parquet(output / 'region_points.parquet', index=False)
    pd.DataFrame(traces).to_csv(output / 'candidate_traces.csv', index=False)
    report = {'date': date, 'full_traces_checked': checked,
              'candidate_trace_files': len(traces), 'region_rows': len(frame),
              'region_ground_rows': int(frame.ground.sum()),
              'numeric_altitude_min_ft': float(frame.altitude_ft.min()),
              'bbox': [49.75, 50.3, 8.15, 9.05], 'candidate_max_altitude_ft': 10000,
              'airport_assignment_verified': False, 'goaround_labels_created': False,
              'license': 'ODbL-1.0', 'archives': verified}
    (output / 'pilot_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'archives'}, ensure_ascii=False), flush=True)
    return report
