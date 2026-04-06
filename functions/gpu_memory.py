import pycuda.driver as cuda

'''
2024/10/28 (Longchao Men): first version
'''

def gpu_memory(type_, logger):
    cuda.init()
    device_count = cuda.Device.count()
    logger.info(type_+':')
    for i in range(device_count):
        # obtain the device
        device = cuda.Device(i)
        logger.info("  - GPU device {}: {}".format(i, device.name()))
        # create a context
        context = device.make_context()
        # obtain the memory information
        total_memory = device.total_memory()
        free_memory = cuda.mem_get_info()[0]
        allocated_memory = total_memory - free_memory
        logger.info("    - Total GPU memory: {:.2f} GB".format(total_memory / (1024 ** 3)))
        logger.info("    - Allocated GPU memory: {:.2f} GB".format(allocated_memory / (1024 ** 3)))
        logger.info("    - Free GPU memory: {:.2f} GB".format(free_memory / (1024 ** 3)))
        
        context.pop()

