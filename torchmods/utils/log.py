import time


def printlog(**kwargs):
    ctime = time.strftime('%Y-%m-%d/%H:%M:%S', time.localtime())
    msg = f'ctime={ctime}'
    for k in kwargs:
        msg += f'\t| {k}={kwargs[k]}'
    print(msg)
