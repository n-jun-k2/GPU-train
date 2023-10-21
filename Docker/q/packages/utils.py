import os

def current_path(path: "__file__"):
    """実行スクリプトのディレクトリパスに移動する
    """
    dir = os.path.dirname(path)
    os.chdir(dir)