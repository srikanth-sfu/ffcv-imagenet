import os
import glob, json

def proc(folder):
    if not os.path.isfile(folder+'/log'):
        return 0,0,0
    log = open(folder + '/log').read().split('\n')
    params = json.load(open(folder + '/params.json'))
    cur_a, cur_b = params['model.alpha'], params['model.beta']
    if not log[-1]:
        log = log[:-1]
    acc_data = json.loads(log[-1])
    acc = acc_data['top_1'] if acc_data['epoch'] == 14 else 0.0
    return cur_a, cur_b, acc

alpha, beta = 1.46, 1.45
for phi in range(3,8):
    os.system('bash train_imagenet.sh %.02f %0.2f %d'%(alpha, beta, phi))
