import os

def get_env(var, default=None):
    return os.environ.get(var, default)

config = {
    # 'app_name': get_env('APP_NAME'),
    # 'jobs_num': get_env('JOBS_NUM'),
    # 'host': get_env('HOST'),
    # 'port': get_env('PORT')
    }