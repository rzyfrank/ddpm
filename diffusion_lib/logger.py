import copy
from pathlib import Path
from .tool import time_for_file
import os, sys, time, torch, random, PIL, copy, numpy as np


class Logger(object):
    def __init__(self, log_dir, logstr):
        """Create a summary writer logging to log_dir."""
        self.log_dir = Path(log_dir)
        self.logstr = logstr
        self.model_dir = Path(log_dir) / 'checkpoint'
        self.sample_dir = Path(log_dir) / 'sample_img'
        self.log_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.model_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.sample_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        # self.tensorboard_dir = self.log_dir / ('tensorboard-{:}'.format(time.strftime( '%d-%h-at-%H:%M:%S', time.gmtime(time.time()) )))
        self.logger_path = self.log_dir / '{:}.log'.format(logstr)
        self.logger_file = open(self.logger_path, 'w')

    def __repr__(self):
        return ('{name}(dir={log_dir}, use-tf={use_tf}, writer={writer})'.format(name=self.__class__.__name__,
                                                                                 **self.__dict__))

    def path(self, mode):
        if mode == 'diffusion_lib':
            return self.model_dir
        elif mode == 'log':
            return self.log_dir
        elif mode == 'sample':
            return self.sample_dir
        else:
            raise TypeError('Unknow mode = {:}'.format(mode))

    def last_info(self):
        return self.log_dir / 'last-info.pth'

    def extract_log(self):
        return self.logger_file

    def close(self):
        self.logger_file.close()

    def log(self, string, save=True):
        print(string)
        if save:
            self.logger_file.write('{:}\n'.format(string))
            self.logger_file.flush()

    def get_time(self):
        return self.logstr

def prepare_logger(xargs):
    args = copy.deepcopy(xargs)
    logstr = time_for_file()
    logger = Logger(args.save_path, logstr)
    logger.log('save_path:{:}'.format(args.save_path))
    logger.log('Arguments:------------------------------------------')
    for name, value in args._get_kwargs():
        logger.log('{:16} : {:}'.format(name, value))
    logger.log("Python  Version : {:}".format(sys.version.replace('\n', ' ')))
    logger.log("Pillow  Version : {:}".format(PIL.__version__))
    logger.log("PyTorch Version : {:}".format(torch.__version__))
    logger.log("cuDNN   Version : {:}".format(torch.backends.cudnn.version()))
    logger.log("CUDA available  : {:}".format(torch.cuda.is_available()))
    logger.log("CUDA GPU numbers: {:}".format(torch.cuda.device_count()))
    logger.log('-----------------------------------------------------')

    return logger


if __name__ == '__main__':
    log_dir = 'log//a-1//abd'
    log_str = 'test'
    logger = Logger(log_dir, log_str)

    print(logger.path('diffusion_lib'))
    print(logger.path('log'))

    print(logger.last_info())


    logger.log('test success {:}'.format(1))
    logger.close()
