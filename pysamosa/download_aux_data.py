from pathlib import Path

from pysamosa.utils import download_untar_file

URL_TEST_DATA_FILE_SIZE = (
    "https://www.dropbox.com/s/ygagj9gebnhsm5b/pysamosa_data.tar?dl=1",
    218451704,
)


def download_test_data() -> Path:
    test_data_dest_path = Path(__file__).parent / ".testdata"

    return download_untar_file(
        url=URL_TEST_DATA_FILE_SIZE[0],
        dest_path=test_data_dest_path,
        expected_file_size=URL_TEST_DATA_FILE_SIZE[1],
    )
