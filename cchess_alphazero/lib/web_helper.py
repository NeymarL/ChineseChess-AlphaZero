import requests
import os
from logging import getLogger

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
    success = False
    for i in range(3):
        try:
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                logger.error(f"Error occurs when download from {url}: {r.text}")
            else:
                success = True
                break
        except Exception as e:
            logger.error(f"Error occurs when download from {url}: {e}")
    
    if not success:
        return False
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
    return True


