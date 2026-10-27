import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import torch
import numpy as np
from tqdm import tqdm

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
if opt.eval:
    model.eval()
print('Loading model %s' % opt.model)

def get_z_random2(batch_size, nz):
    num = batch_size
    base_vector = np.random.normal(loc=0, scale=1, size=(1, nz))
    base_vector /= np.linalg.norm(base_vector)  
   
    random_directions = np.random.normal(loc=0, scale=1, size=(num, nz))
    random_directions /= np.linalg.norm(random_directions, axis=1,keepdims=True)
    fixed_distance =0.5
    fixed_distance_vectors = base_vector + random_directions * fixed_distance
    z=torch.from_numpy(fixed_distance_vectors).float()
    return z

def output_path(p, oripath, savepath):
    sp = os.path.dirname(p)
    dp = sp.replace(oripath, savepath)
    return p.replace(sp, dp)


def saveimg(img, p, oripath, savepath):
    img = util.tensor2im(img)
    destination = output_path(p, oripath, savepath)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    util.save_image(img, destination)

if __name__ == '__main__':
    length = len(dataset)
    dataset = iter(dataset)
    for i in tqdm(range(0, length, opt.n_samples)):
        group_size = min(opt.n_samples, length - i)
        z_samples = get_z_random2(group_size, opt.nz)
        for j in range(group_size):
            data = next(dataset)
            destination = output_path(data['A_paths'][0], opt.dataroot, opt.results_dir)
            if os.path.exists(destination):
                if opt.resume:
                    continue
                raise FileExistsError('Refusing to overwrite {}; use --resume or another --results_dir'.format(destination))
            model.set_input(data)

            fake_B, p = model.test(z_samples[[j]], encode=False)
            # print(p)
            saveimg(fake_B, p[0], opt.dataroot, opt.results_dir)
    
