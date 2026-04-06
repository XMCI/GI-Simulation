import os
import sys

''' 
2024/10/28 (Longchao Men): first version
'''
def configure_output(config_filename = None):
    sys.path.append('configs//')
    configs = __import__(config_filename)

    output_vars = configs.outputParasNames
    output_vars = str.split(output_vars,'_')
    output_num = len(output_vars)
    output_dir = configs.outputDir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # output data file
    output_filename = configs.outputFilename
    output_filepath = os.path.join(output_dir, output_filename)
    if os.path.exists(output_filepath):
        reply = input('Output file exists! Do you want to overwrite? (y for yes)\n')
        if not reply == 'y' :
            reply2 = input('Stop overwritting! Do you want to continue with new output filename? (y for yes)\n')
            if reply2 == 'y':
                retry = 0
                output_filename = output_filename + '_' + str(retry)
                while os.path.exists(os.path.join(output_dir, output_filename+'.npz')):
                    retry = retry + 1
                    output_filename[-1] = str(retry)

                output_filepath = os.path.join(output_dir, output_filename)
                print(('New output file path: ', output_filepath))
            else:
                print('Exit because there exists an old ouput file with same name: ', output_filepath)
                return
        else:
            print('Overwrite the old file: ', output_filepath)

    return output_vars, output_num, output_filepath