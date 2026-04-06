import logging
import os
import sys

'''
This function configures logging for the application based on a specified configuration file.

2024/10/28 (Longchao Men): first version
'''

def configure_logging(config_filename = None):
    sys.path.append('configs//')
    configs = __import__(config_filename)
    
    # output log file
    log_dir = configs.logDir
    if not os.path.exists(log_dir) :
        os.mkdir(log_dir)
    
    log_filename = configs.outputFilename
    log_filepath = os.path.join(log_dir, log_filename+'.log')
    if os.path.exists(log_filepath):
        reply = input('log file exists! Do you want to overwrite? (y for yes)\n')
        if not reply == 'y':
            reply2 = input('Stop overwritting! Do you want to continue with new log filename? (y for yes)\n')
            if reply2 == 'y':
                retry = 0
                log_filename = log_filename + '_' + str(retry)
                while os.path.exists(os.path.join(log_dir, log_filename+'.txt')):
                    retry = retry + 1
                    log_filename[-1] = str(retry)

                log_filepath = os.path.join(log_dir, log_filename+'.txt')
                print('New log file path: ', log_filepath)
            else:
                raise Exception('Exit because there exists an old ouput file with same name: ' + log_filepath)
        else:
            fid = open(log_filepath, 'w+')
            fid.close()
            print('Overwrite the old file: ', log_filepath)
    
    return setup_logging(log_filepath)


def setup_logging(log_filepath):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
