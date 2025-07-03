import glob, os, json
import ast
import pprint

all_folders = glob.glob('final_weights/*')
dst = 'weights_b0/'
results = []
results_5 = []
for folder in all_folders:
    params = json.load(open(folder + '/params.json'))
    phi = params['model.phi']
    dst_fn = dst + 's'+ str(phi)+'.pt'
    src = folder + '/final_weights.pt'
    os.system('cp %s %s'%(src, dst_fn))
    log = open(folder + '/log').read().split('\n')[:-1]
    results.append((phi, ast.literal_eval(log[-1])['top_1']))
    results_5.append((phi, ast.literal_eval(log[-1])['top_5']))

results.sort()
pprint.pprint(results)
print('\n\n')
results_5.sort()
pprint.pprint(results_5)
