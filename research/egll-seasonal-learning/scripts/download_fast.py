"""Загрузка больших архивов по проверенным диапазонам HTTP"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import time
import urllib.request

BASE = Path(__file__).resolve().parents[1]


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'Approach-Risk-Research', **(headers or {})}), timeout=120)


def download(date):
    folder = BASE / 'data' / date / 'archive'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'download_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if all(Path(f['path']).exists() and Path(f['path']).stat().st_size == f['bytes']
               for f in saved['files']):
            return folder, saved['files']
    tag = 'v' + date.replace('-', '.') + '-planes-readsb-prod-0'
    url = f'https://api.github.com/repos/adsblol/globe_history_{date[:4]}/releases/tags/{tag}'
    with request(url) as response:
        release = json.load(response)
    assets = sorted([a for a in release['assets'] if '.tar' in a['name']], key=lambda a: a['name'])
    assert assets

    def part(asset):
        path = folder / asset['name']
        size = asset['size']
        offset = path.stat().st_size if path.exists() else 0
        assert offset <= size
        block = 128 * 1024 * 1024
        ranges = [(start, min(start + block, size) - 1) for start in range(offset, size, block)]

        def chunk(bounds):
            start, end = bounds
            target = folder / f'{asset["name"]}.range_{start}_{end}'
            for attempt in range(5):
                copied = target.stat().st_size if target.exists() else 0
                assert copied <= end - start + 1
                if copied == end - start + 1:
                    break
                try:
                    begin = start + copied
                    with request(asset['browser_download_url'], {'Range': f'bytes={begin}-{end}'}) as response:
                        assert response.status == 206, 'Сервер не подтвердил частичную загрузку'
                        assert response.headers['Content-Range'] == f'bytes {begin}-{end}/{size}'
                        with target.open('ab') as stream:
                            shutil.copyfileobj(response, stream, length=1024 * 1024)
                    assert target.stat().st_size == end - start + 1
                    break
                except (OSError, TimeoutError):
                    if attempt == 4:
                        raise
                    time.sleep(2)
            assert target.stat().st_size == end - start + 1
            print('DOWNLOADED_RANGE', date, path.suffix, start, end, flush=True)
            return target

        with ThreadPoolExecutor(max_workers=2) as pool:
            chunks = list(pool.map(chunk, ranges))
        with path.open('ab') as output:
            for file in chunks:
                with file.open('rb') as stream:
                    shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
                output.flush()
        assert path.stat().st_size == size
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if asset.get('digest'):
            assert 'sha256:' + digest == asset['digest']
        for file in chunks:
            file.unlink()
        return {'path': str(path), 'bytes': size, 'sha256': digest, 'url': asset['browser_download_url']}

    with ThreadPoolExecutor(max_workers=2) as pool:
        entries = list(pool.map(part, assets))
    manifest.write_text(json.dumps({'release_url': url, 'files': entries}, indent=2), encoding='utf-8')
    return folder, entries


if __name__ == '__main__':
    import sys
    for date in sys.argv[1:]:
        download(date)
