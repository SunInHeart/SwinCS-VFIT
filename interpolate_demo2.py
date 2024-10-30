import os
import torch

import config
from PIL import Image
from torchvision import transforms
import torchvision.utils as utils

### Parse CmdLine Arguments ###
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# interpolate
interp_arg = config.add_argument_group('Interp')
interp_arg.add_argument('--img_path', type=str, default='../inter_data/')
interp_arg.add_argument('--out_path', type=str, default='../out_data/')


args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

if args.model == 'SwinCS_VFIT_S':
    from model.VFIT_S import UNet_3D_3D
    model_name = 'SwinCS-VFIT-S'
elif args.model == 'SwinCS_VFIT_B':
    from model.VFIT_B import UNet_3D_3D
    model_name = 'SwinCS-VFIT-B'

print("Building model: %s" % args.model)
model = UNet_3D_3D(n_inputs=args.nbr_frame, joinType=args.joinType)

model = torch.nn.DataParallel(model).to(device)
print("#params", sum([p.numel() for p in model.parameters()]))

def read_path(path):
    # paths
    paths = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    return paths

def generate(args):
    model.eval()

    # transform
    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    # get images
    dir_paths = read_path(args.img_path)

    for dir_path in dir_paths:
        imgpaths = [dir_path + f'/im{i}.png' for i in range(1, 8)]
        images = [Image.open(pth) for pth in imgpaths]
        images = [images[i-1] for i in [1, 3, 5, 7]]
        images = [trans(img) for img in images]

        with torch.no_grad():
            images = [img_.cuda().unsqueeze(0) for img_ in images]
            out = model(images)
            # print(out)
            dir_name = os.path.basename(dir_path)
            out_data_path = args.out_path + dir_name
            if not os.path.exists(out_data_path):
                os.makedirs(out_data_path)
            utils.save_image(out, out_data_path + f'/{model_name}_out.png')
            print(dir_name + " result saved!")


""" Entry Point """
def main(args):

    assert args.load_from is not None

    model.load_state_dict(torch.load(args.load_from)["state_dict"], strict=True)
    generate(args)


if __name__ == "__main__":
    main(args)