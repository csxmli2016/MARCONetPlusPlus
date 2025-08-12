from basicsr.utils.download_util import load_file_from_url

os.makedirs('./checkpoints', exist_ok=True)

load_file_from_url('https://github.com/csxmli2016/MARCONetPlusPlus/releases/download/v1/bsrgan_bg.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONetPlusPlus/releases/download/v1/net_prior_860000.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONetPlusPlus/releases/download/v1/net_sr_860000.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONetPlusPlus/releases/download/v1/net_w_encoder_860000.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONetPlusPlus/releases/download/v1/yolo11m_short_character.pt', model_dir='./checkpoints')
