import os
from datetime import datetime


def create(*args):
    path = '/'.join(a for a in args)
    if not os.path.isdir(path):
        os.makedirs(path)


class Logging:
    def __init__(self, saveroot, filename='log.txt', log_=True):
        self.log_path = os.path.join(saveroot, filename)
        self.log_ = log_

    def info(self, s, print_=True):
        if print_:
            print(f'{datetime.now()} / {s}')
        if self.log_:
            with open(self.log_path, 'a+') as f_log:
                f_log.write(f'{datetime.now()} / {s} \n')
