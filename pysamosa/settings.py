from pathlib import Path


LFSDATA_DIR = Path(__file__).parent.parent / ".data"

S3_DATA_DIR = LFSDATA_DIR / "s3"
S6_DATA_DIR = LFSDATA_DIR / "s6"
FFSAR_DATA_DIR = LFSDATA_DIR / "s6" / "ffsar"
CS_DATA_DIR = LFSDATA_DIR / "cs"
