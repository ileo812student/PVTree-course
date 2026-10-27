import os, shutil
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def cross(vein_path,palm_path, newpath):
    img1 = cv2.imread(vein_path)
    img2 = cv2.imread(palm_path)
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    h, w, c = img1.shape
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img1,img1, mask = mask)
    img2_fg = cv2.bitwise_and(img2,img2, mask = mask_inv)
    dst = cv2.add(img1_bg, img2_fg)
    cv2.imwrite(newpath, dst)

def cross_folder(vein_path, palmprint_path, pv_path, sams=7, num_percase=10, num_case=4,
                 resume=False, force=False):
    # sams: num of samples
    # num: num_id/num_case,  4000/4=1000
    os.makedirs(pv_path, exist_ok=True)
    # list_v = os.listdir(vein_path)
    list_p = sorted(os.listdir(palmprint_path))
    for r in range(num_case):
        x = r * num_percase  # 0,1000,2000,3000
        for i in tqdm(range(x, x+num_percase)):
            palmimg = os.path.join(palmprint_path, list_p[i])
            for j in range(0, sams):
                identity = 'c{}_{}'.format(r+1, i-x)
                filename = identity + '_sample' + '{}.png'.format(j)
                veinimg = os.path.join(vein_path, filename)
                newimg = os.path.join(pv_path, identity, filename)
                os.makedirs(os.path.dirname(newimg), exist_ok=True)
                if os.path.exists(newimg) and resume:
                    continue
                if os.path.exists(newimg) and not force:
                    raise FileExistsError(f'Refusing to overwrite {newimg}')
                cross(veinimg, palmimg, newimg)


def aug(path, newpath, seed=2026, resume=False, force=False):

    transfor = transforms.Compose([

        transforms.Resize((265,265), interpolation=Image.BILINEAR),
        transforms.RandomPerspective(distortion_scale=0.1, p=1, fill=255),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.RandomCrop((256,256), padding=0, pad_if_needed=False),
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        # transforms.ColorJitter(brightness=(1, 1.3), contrast=0.1),
    ])

    if not os.path.exists(newpath):
        os.makedirs(newpath)
    inputs = sorted(
        os.path.join(root, filename)
        for root, _, files in os.walk(path)
        for filename in files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    )
    for imgpath in tqdm(inputs):
        relative = os.path.relpath(imgpath, path)
        newimgpath = os.path.join(newpath, relative)
        os.makedirs(os.path.dirname(newimgpath), exist_ok=True)
        if os.path.exists(newimgpath) and resume:
            continue
        if os.path.exists(newimgpath) and not force:
            raise FileExistsError(f'Refusing to overwrite {newimgpath}')
        import hashlib
        import torch
        item_seed = int.from_bytes(hashlib.blake2b(f'{seed}|{relative}'.encode(), digest_size=8).digest(), 'little')
        torch.manual_seed(item_seed % (2 ** 63 - 1))
        img = Image.open(imgpath)
        sam = transfor(img)
        sam.save(newimgpath)


def cvt(p):
    
    for img in sorted(os.listdir(p)):
        imgpath = os.path.join(p, img)
        if os.path.isdir(imgpath):
            pass
        else:
            case, ids, _ = img.split('_')
            idname = case + '_' + ids
            idpath = os.path.join(p, idname)
            os.makedirs(idpath, exist_ok=True)
            newimgpath = os.path.join(idpath, img)
            shutil.move(imgpath, newimgpath)
