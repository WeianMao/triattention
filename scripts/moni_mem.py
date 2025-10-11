import psutil
import time

largest_vmem = 0
# largest_smem = 0
largest_vmem_used = 0
largest_cpu_temp = 0
time_last = time.time()
while True:
    cur_vmem = psutil.virtual_memory().percent
    # cur_smem = psutil.swap_memory().percent
    cur_vmem_used = psutil.virtual_memory().used/1024/1024/1024
    try:
        cur_cpu_temp = psutil.sensors_temperatures()['coretemp'][0].current
    except:
        cur_cpu_temp = 0

    if cur_vmem>=largest_vmem:
        largest_vmem = cur_vmem
    if cur_vmem_used>=largest_vmem_used:
        largest_vmem_used = cur_vmem_used
    if cur_cpu_temp>=largest_cpu_temp:
        largest_cpu_temp = cur_cpu_temp
    # if cur_smem>=largest_smem:
    #     largest_smem = cur_smem
    time.sleep(0.1)

    if time.time()-time_last>1:
        print('-------------------------------')
        print('cur_vmem:{}'.format(cur_vmem))
        print('cur_vmem_used:{}'.format('%0.2f'%cur_vmem_used))
        print('cur_cpu_temp:{}'.format(cur_cpu_temp))
        # print('cur_smem:{}'.format(cur_smem))
        print('largest_vmem:{}'.format(largest_vmem))
        print('largest_vmem_used:{}'.format('%0.2f'%largest_vmem_used))
        print('largest_cpu_temp:{}'.format(largest_cpu_temp))
        # print('largest_smem:{}'.format(cur_smem))
        print('-------------------------------')
        time_last = time.time()