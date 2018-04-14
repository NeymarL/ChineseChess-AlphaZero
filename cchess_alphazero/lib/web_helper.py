import requests
import os
from logging import getLogger
from urllib.request import urlopen
from tqdm import tqdm

logger = getLogger(__name__)

def upload_file(url, path, filename=None, data=None, rm=False):
    filename = filename if filename is not None else 'file'
    files = {'file': (filename, open(path, 'rb'), 'application/json')}
    success = False
    for i in range(3):
        try:
            r = requests.post(url, files=files, data=data)
            if r.status_code != 200:
                logger.error(f"Error occurs when upload {filename}: {r.text}")
            else:
                success = True
                break
        except Exception as e:
            logger.error(f"Error occurs when upload {filename}: {e}")
    if rm:
        os.remove(path)
    return r.json() if success else None

def http_request(url, post=False, data=None):
    success = False
    try:
        if post:
            r = requests.post(url, data=data)
        else:
            r = requests.get(url)
        if r.status_code != 200:
            logger.error(f"Error occurs when request {url}: {r.text}")
        else:
            success = True
    except Exception as e:
        logger.error(f"Error occurs when request {url}: {e}")
    return r.json() if success else None

def download_file(url, save_path):
    if os.path.exists(save_path):
        os.remove(save_path)
    file_size = int(urlopen(url).info().get('Content-Length', -1))
    if os.path.exists(save_path):
        first_byte = os.path.getsize(save_path)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(save_path, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return True


