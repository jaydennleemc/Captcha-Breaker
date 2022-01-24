import zipfile


def unzip_captcha(zip_file_path, captcha_path):
    """
    Unzip a zip file to a given path.

    :param zip_file_path: Path to the zip file.
    :param captcha_path: Path to the captcha.
    :return:
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(captcha_path)
