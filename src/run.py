import os
import sys
import time
import yaml
import logging
from pathlib import Path
from datetime import datetime
from pipeline import Pipeline

# ====== SETUP LOG HANDLER ====== #
def setup_logger(log_level) -> None:
    Path('Logs').mkdir(exist_ok=True, parents=True)
    # Create logger
    logger = logging.getLogger('Mosaic-Pipeline')
    # Setup output formating
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    # Setup log lovel
    logger.setLevel(logging.getLevelName(log_level))
    # Setup file handler
    file_name = datetime.now().strftime("%Y%m%dT%H%M%S-Mosaic")
    file_handler = logging.FileHandler("{0}/{1}.log".format('Logs', file_name))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Setup stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

# ====== READ CONFIGS ====== #
def read_config() -> dict:
    # Load the file
    with open('config.yml', 'r') as config_file:
        configs = yaml.safe_load(config_file)
    with Path('/external_config.yaml').expanduser().open('r') as f:
        c = yaml.safe_load(f)
        configs['input_dataset'] = c['Exp1-Mosaic']['input_dataset']
        configs['NFCS_regions'] = c['Exp1-Mosaic']['NFCS_regions']
    # Add environment variables
    configs['ctx_id'] = int(os.environ['CTX_ID'])

    return configs

# ====== MAIN ====== #
if __name__=='__main__':
    start = time.time()
    configs = read_config()
    # === Setup the logger params
    logger = setup_logger(configs['log_level'])
    print("CONFIGS: ", configs)
    logger.info(f'[DATA] Input arguments: {configs}')
    # === Create the pipeline
    logger.info('[CHECKPOINT] Creating pipeline')
    pipeline = Pipeline(**configs)
    # === Run the detection
    logger.info('[CHECKPOINT] Starting detection')
    if configs['process_detection']: pipeline.detection()
    else: logger.warning('[WARNING] Skipping detection step')
    logger.info('[CHECKPOINT] Finished detection')
    # === Run the mosaic
    logger.info('[CHECKPOINT] Starting mosaic')
    if configs['process_mosaic']: pipeline.mosaic()
    else: logger.warning('[WARNING] Skipping mosaic step')
    logger.info('[CHECKPOINT] Finished mosaic')
    # === Join nfcs regions
    if configs['NFCS_regions']:
        logger.info('[CHECKPOINT] Joining nfcs regions')
        pipeline.join_regions()
        logger.info('[CHECKPOINT] Finished nfcs regions')
    # === End logs
    logger.info(f'[END] Finished after {(time.time()-start):.2f} seconds')