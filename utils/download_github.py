
import os
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

def download_file_from_github(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Reference: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


os.makedirs('./checkpoints', exist_ok=True)
download_file_from_github('https://github.com/csxmli2016/MARCONetPlusPlus/releases/download/v1/bsrgan_bg.pth', model_dir='./checkpoints')
download_file_from_github('https://github.com/csxmli2016/MARCONetPlusPlus/releases/download/v1/net_prior_860000.pth', model_dir='./checkpoints')
download_file_from_github('https://github.com/csxmli2016/MARCONetPlusPlus/releases/download/v1/net_sr_860000.pth', model_dir='./checkpoints')
download_file_from_github('https://github.com/csxmli2016/MARCONetPlusPlus/releases/download/v1/net_w_encoder_860000.pth', model_dir='./checkpoints')
download_file_from_github('https://github.com/csxmli2016/MARCONetPlusPlus/releases/download/v1/yolo11m_short_character.pt', model_dir='./checkpoints')
