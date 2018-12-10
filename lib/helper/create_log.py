from ..config import DIR_OUTPUT
from ..config import LOG_NAME


def create_log():
    log_path = os.path.join(DIR_OUTPUT, LOG_NAME)
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(filename=log_path,
                        format='%(asctime) 8s--- %(message)s',
                        datefmt='%a, %d %b %H:%M:%S',
                        level=logging.INFO)

