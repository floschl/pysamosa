import datetime
import logging
import os


class DeltaTimeFormatter(logging.Formatter):
    def format(self, record):
        now = datetime.datetime.now()
        self.last_timestamp = (
            now if "last_timestamp" not in dir(self) else self.last_timestamp
        )
        delta_last = now - self.last_timestamp
        self.last_timestamp = now
        hours, _ = divmod(delta_last.seconds, 3600)
        minutes, seconds = divmod(delta_last.seconds, 60)
        record.delta_last = f"{hours:02d}:{minutes // 60:02d}:{seconds:02d}"

        duration = datetime.datetime.utcfromtimestamp(record.relativeCreated / 1000)
        record.delta_start = duration.strftime("%H:%M:%S")

        return super().format(record)


format_str = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s (to_start: %(delta_start)s, to_last: %(delta_last)s)"


def set_root_logger(log_level=logging.DEBUG):
    for lh in logging.root.handlers:
        logging.root.removeHandler(lh)

    logging.root.setLevel(log_level)

    # add stream handler
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(DeltaTimeFormatter(format_str))
    logging.root.addHandler(sh)

    logging.getLogger("matplotlib.font_manager").disabled = True


def generate_and_add_file_logger(log_filename):
    # add file handler
    # create new logfile
    os.umask(0)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    if os.path.exists(log_filename):
        os.remove(log_filename)

    fh = logging.FileHandler(log_filename)
    fh.setFormatter(DeltaTimeFormatter(format_str))
    fh.setLevel(logging.DEBUG)

    logging.root.addHandler(fh)

    return fh
