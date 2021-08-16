# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

import argparse
import math
from pathlib import Path
from urllib.request import urlopen
from tqdm import tqdm
import sys
sys.path.append('taming-transformers')

from base64 import b64encode
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
import taming.modules 

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import glob
import os
import time
output_frames=[]
import subprocess as sp

# Create the parser
vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

# Add the arguments
vq_parser.add_argument("-p", "--prompts", type=str, help="Text prompts", default=None, dest='prompts')
vq_parser.add_argument("-o", "--output", type=str, help="Output file", default="output", dest='output')
vq_parser.add_argument("-i", "--iterations", type=int, help="Number of iterations", default=500, dest='max_iterations')
vq_parser.add_argument("-ip", "--image_prompts", type=str, help="Image prompts / target image", default=[], dest='image_prompts')
vq_parser.add_argument("-nps", "--noise_prompt_seeds", nargs="*", type=int, help="Noise prompt seeds", default=[], dest='noise_prompt_seeds')
vq_parser.add_argument("-npw", "--noise_prompt_weights", nargs="*", type=float, help="Noise prompt weights", default=[], dest='noise_prompt_weights')
vq_parser.add_argument("-s", "--size", nargs=2, type=int, help="Image size (width height)", default=[512,512], dest='size')
vq_parser.add_argument("-ii", "--init_image", type=str, help="Initial image", default=None, dest='init_image')
vq_parser.add_argument("-iw", "--init_weight", type=float, help="Initial image weight", default=0., dest='init_weight')
vq_parser.add_argument("-m", "--clip_model", type=str, help="CLIP model", default='ViT-B/16', dest='clip_model')
vq_parser.add_argument("-conf", "--vqgan_config", type=str, help="VQGAN config", default=f'checkpoints/vqgan_imagenet_f16_16384.yaml', dest='vqgan_config')
vq_parser.add_argument("-ckpt", "--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=f'checkpoints/vqgan_imagenet_f16_16384.ckpt', dest='vqgan_checkpoint')
# vq_parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate", default=0.2, dest='step_size')
# THIS DEFAULT LEARNING RATE IS INTENDED FOR USE WITH SOME FORM OF PLATEAU OPTIMISER!
vq_parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate - REDUCE IF lr_optimiser IS DISABLED", default=1000, dest='step_size')
vq_parser.add_argument("-lrm", "--learning_rate_min", type=float, help="Minimum learning rate for cleanup (plateau) to reach", default=5e-6, dest='plateau_min_lr')
vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest='cutn')
vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')
vq_parser.add_argument("-se", "--save_every", type=int, help="Save image iterations", default=1, dest='display_freq')
vq_parser.add_argument("-sd", "--seed", type=int, help="Seed", default=None, dest='seed')
vq_parser.add_argument("-opt", "--optimiser", type=str, help="Optimiser (Adam, AdamW, Adagrad, Adamax, SGD) ", default='Adam', dest='optimiser')
vq_parser.add_argument("-np", "--negative_prompt", type=str, help="Text prompts with negative optimisation", default=None, dest='negative_prompts')
vq_parser.add_argument("-lo", "--lr_optimiser", type=str, help="Learning rate optimiser (None, Plateau, Anneal, Wave)", default="Plateau", dest='lr_opt')
vq_parser.add_argument("-osq", "--optimal_sequence", help="Only output frames with new optimal loss", action='store_true', dest='opt_seq')
# vq_parser.add_argument("-nov", "--no_overtime", help="Do not allow for some extra iterations during the cleanup (plateau) pass", action='store_true', dest='no_overtime')
vq_parser.add_argument("-ovf", "--overtime_factor", type=float, help="Allow for some extra iterations during the cleanup (plateau) pass. Factor of iteration count.", default=0.25, dest='overtime_factor')
vq_parser.add_argument("-nvd", "--no_video", help="Do not add a true video file as an output. (Video requires ffmpeg executable.)", action='store_true', dest='no_video')
vq_parser.add_argument("-sif", "--save_intermediate_frames", help="Re-save output png on every -se interval. Provides progress updates but slows down the process.", action='store_true', dest='save_intermediate')
vq_parser.add_argument("-mgs", "--max_gif_size_mb", type=float, help="Size limit for the gif file in MB. Intermediate frames will be dropped until this fits.", default=8, dest='max_gif_size_mb')
vq_parser.add_argument("-ncb", "--no_cudnn_benchmark", help="Don't run cudnn benchmark (normally used to optimise processing performance)", action='store_true', dest='no_cudnn_bench')
vq_parser.add_argument("-cdi", "--cuda_device_id", type=int, help="Set CUDA device ID. Only required if a secondary CUDA device is available and should be used.", default=0, dest='cuda_device_id')
vq_parser.add_argument("-pd", "--plateau_delay", type=float, help="Factor of overall iterations to wait until scheduler is applied in plateau", default=0.175, dest='plateau_delay')
vq_parser.add_argument("-pp", "--plateau_patience", type=int, help="Patience value for plateau scheduler", default=3, dest='plateau_patience')
vq_parser.add_argument("-pf", "--plateau_factor", type=float, help="LR factor applied in a plateau step", default=0.8, dest='plateau_factor')
vq_parser.add_argument("-pe", "--plateau_exit_early", help="Permit early exit if min lr is reached by plateau", action='store_true', dest='exit_early')

# Execute the parse_args() method
args = vq_parser.parse_args()
if not args.no_cudnn_bench:
    print("Running cudnn benchmark to (hopefully) boost performance. Disable with -ncb")
torch.backends.cudnn.benchmark = not args.no_cudnn_bench	# NR: True is a bit faster, but can lead to OOM. False is also more deterministic.
torch.use_deterministic_algorithms(False)

should_make_video = not args.no_video
max_overtime = 1+args.overtime_factor
plateau_min_lr = args.plateau_min_lr

# make some of the string based option selector args case insensitive
if args.lr_opt:
    args.lr_opt = args.lr_opt.lower()
if args.optimiser:
    args.optimiser = args.optimiser.lower()

if args.opt_seq:
    print("output will only contain frames with new optimal loss")

if not args.prompts and not args.negative_prompts:
    args.prompts = "painting"

# Split text prompts using the pipe character
if args.prompts:
    args.prompts = [phrase.strip() for phrase in args.prompts.split("|")]

if args.negative_prompts:
    args.negative_prompts = [phrase.strip() for phrase in args.negative_prompts.split("|")]

# Split target images using the pipe character
if args.image_prompts:
    args.image_prompts = args.image_prompts.split("|")
    args.image_prompts = [image.strip() for image in args.image_prompts]

# init. output file timestamp
timeObj = time.localtime(time.time())
timestamp = '%d_%d_%d-%d_%d_%d' % (timeObj.tm_year, timeObj.tm_mon, timeObj.tm_mday, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
png_file_path = "outputs/" + args.output + timestamp + ".png"
# try to create 'outputs' folder if not present.
try:
    os.makedirs('outputs')
except FileExistsError as e:
    pass
except Exception as e:
    print(f"'outputs' folder could not be created: {e}")


# Functions and classes
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


# Not used?
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.augs = nn.Sequential(
            # K.RandomHorizontalFlip(p=0.5),
            # K.RandomVerticalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            # K.RandomSharpness(0.3,p=0.4),
            # K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5),
            # K.RandomCrop(size=(self.cut_size,self.cut_size), p=0.5),
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),
            
)
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):

            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            # offsetx = torch.randint(0, sideX - size + 1, ())
            # offsety = torch.randint(0, sideY - size + 1, ())
            # cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            # cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

            # cutout = transforms.Resize(size=(self.cut_size, self.cut_size))(input)
            
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    global gumbel
    gumbel = False
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)



# Do it
device_name = f"cuda:{args.cuda_device_id}" if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
jit = True if float(torch.__version__[:3]) < 1.8 else False
print("available CLIP models:")
print(clip.available_models())
print(f"using: {args.clip_model} (select a different model with -m)")
perceptor = clip.load(args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)

# clock=deepcopy(perceptor.visual.positional_embedding.data)
# perceptor.visual.positional_embedding.data = clock/clock.max()
# perceptor.visual.positional_embedding.data=clamp_with_grad(clock,0,1)

cut_size = perceptor.visual.input_resolution

f = 2**(model.decoder.num_resolutions - 1)
make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)

toksX, toksY = args.size[0] // f, args.size[1] // f
sideX, sideY = toksX * f, toksY * f

if gumbel:
    e_dim = 256
    n_toks = model.quantize.n_embed
    z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
else:
    e_dim = model.quantize.e_dim
    n_toks = model.quantize.n_e
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    
# z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
# z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

# normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                            std=[0.229, 0.224, 0.225])

if args.init_image:
    if 'http' in args.init_image:
      img = Image.open(urlopen(args.init_image))
    else:
      img = Image.open(args.init_image)

    pil_image = img.convert('RGB')
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
else:
    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
    # z = one_hot @ model.quantize.embedding.weight
    if gumbel:
        z = one_hot @ model.quantize.embed.weight
    else:
        z = one_hot @ model.quantize.embedding.weight

    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
    z = torch.rand_like(z)*2

z_orig = z.clone()
z.requires_grad_(True)

pMs = []
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])


# Set the optimiser
if args.optimiser == "adam":
    opt = optim.Adam([z], lr=args.step_size)		# LR=0.1    
elif args.optimiser == "adamw":
    opt = optim.AdamW([z], lr=args.step_size)		# LR=0.2    
elif args.optimiser == "adagrad":
    opt = optim.Adagrad([z], lr=args.step_size)	# LR=0.5+
elif args.optimiser == "adamax":
    opt = optim.Adamax([z], lr=args.step_size)	# LR=0.2
elif args.optimiser == "sgd":
    args.step_size *= 5000
    opt = optim.SGD([z], lr=args.step_size, momentum=1)
else:
    print("WARNING: unknown optimiser requested!")

# Output for the user
print(f"Using device: {device} [{device_name}]")
print(f"Optimising using: {opt}")

if args.prompts:
    print('Using text prompts:', args.prompts)
if args.negative_prompts:
     print('Using negative text prompts:', args.negative_prompts)
if args.image_prompts:
    print('Using image prompts:', args.image_prompts)
if args.init_image:
    print('Using initial image:', args.init_image)
if args.noise_prompt_weights:
    print('Noise prompt weights:', args.noise_prompt_weights)    
if args.seed is None:
    seed = torch.seed()
else:
    seed = args.seed  
torch.manual_seed(seed)
print('Using seed:', seed)


if args.prompts:
    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

# add negative prompts the same way as normal prompts
# attach an attribute to mark them as negative for loss calculations
if args.negative_prompts:
    for prompt in args.negative_prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pM = Prompt(embed, weight, stop).to(device)
        setattr(pM, 'negative_prompt', True)
        pMs.append(pM)

for prompt in args.image_prompts:
    path, weight, stop = parse_prompt(prompt)
    img = Image.open(path)
    pil_image = img.convert('RGB')
    img = resize_image(pil_image, (sideX, sideY))
    batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
    embed = perceptor.encode_image(normalize(batch)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))

for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):			# NR: weights
    gen = torch.Generator().manual_seed(seed)
    embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
    pMs.append(Prompt(embed, weight).to(device))


def synth(z):
    if gumbel:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
    else:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


global best_loss
best_loss = 1
global unsaved_best_loss_sequence
unsaved_best_loss_sequence = False
global last_saved_frame
last_saved_frame = -9999
global current_lr
current_lr = args.step_size


@torch.no_grad()
def checkin(i, losses):
    global best_loss
    global current_lr
    global unsaved_best_loss_sequence
    global last_saved_frame
    global generated_image

    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    current_lr = opt.param_groups[0]['lr']
    learning_rate_str = "{:.2e}".format(current_lr)
    loss_value = sum(losses).item()

    # if there is already an unsaved "best frame", or this is a new best frame, there will be an unsaved best frame
    new_best_loss = loss_value < best_loss
    unsaved_best_loss_sequence = unsaved_best_loss_sequence or new_best_loss

    # don't waste time generating the image or checking in if it won't actually be required
    is_final_frame = (i == args.max_iterations) or (i >= max_overtime*args.max_iterations)
    not_opt_seq_but_timer = (i % args.display_freq == 0 or is_final_frame) and not args.opt_seq
    if new_best_loss or not_opt_seq_but_timer:
        tqdm.write(f'i: {i}, loss: {loss_value:g}, losses: {losses_str}, lr: {learning_rate_str}')
        out = synth(z)

        # if this image is a new best image or we reached this point due to the 'save every' timer, generate the image.
        generated_image = TF.to_pil_image(out[0].cpu())

        try:
            # if intermediate frames should be saved as progress updates, write a png file
            if args.save_intermediate:
                generated_image.save(png_file_path)
            best_loss = loss_value
            # if this is a new best loss value, store it now.
            # this is done here so that the value is only updated if saving was actually a success (if intermediate frames are saved).
            if new_best_loss:
                best_loss = loss_value
        except Exception as e:
            print(e)

    # if this frame is supposed to be part of the sequence, either by being a new best, or because of the timer, add it.
    try:
        if i - last_saved_frame > args.display_freq and (unsaved_best_loss_sequence or not args.opt_seq):
            last_saved_frame = i
            unsaved_best_loss_sequence = False
            # save individual frames
            # generated_image.save("outputs/"+str(i).zfill(6)+".png")
            output_frames.append(generated_image)
    except Exception as e:
        print(e)


def ascend_txt():
    global i
    out = synth(z)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
    
    result = []

    if args.init_weight:
        # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
        result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1/torch.tensor(i*2 + 1))*args.init_weight) / 2)
    for prompt in pMs:
        # if a prompt was marked as negative, its effective loss goes up for a high accuracy of the prompt
        # this is currently modelled by applying ln(2/x)
        if getattr(prompt, 'negative_prompt', False):
            result.append( torch.log(2/ prompt(iii)) )
        else:
            result.append(prompt(iii))
        
    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
    img = np.transpose(img, (1, 2, 0))

    return result


# plateau scheduler, minimum LR is set by an arg - reaching it is used as the exit condition for overtime
sched_plateau = optim.lr_scheduler.ReduceLROnPlateau(opt, min_lr = plateau_min_lr, verbose=False, factor=args.plateau_factor, patience=args.plateau_patience)

# try annealing, see what happens?
sched_anneal = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=75, T_mult=2, verbose=False)

# cyclic LR
# initial_learning_rate = opt.param_groups[0]['lr']
# this one won't work with our default optimiser
# sched_cycle = optim.lr_scheduler.CyclicLR(opt, base_lr=initial_learning_rate*(1/50), max_lr=initial_learning_rate*50, step_size_up=100)
# use another annealer but without increasing steps to make a sort-of wave LR
sched_wave = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=66, verbose=False)


def train(i):
    # run a step
    opt.zero_grad(set_to_none=True)
    lossAll = ascend_txt()

    # perform checkin (save current frame, get metrics)
    checkin(i, lossAll)

    # calculate loss value, run optimiser
    loss = sum(lossAll)
    loss.backward()
    opt.step()
    
    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))

    # apply learning rate schedulers:
    # for annealing, the final few iterations are used to 'clean up' the current image with plateau
    if i > args.max_iterations * 0.8 and (args.lr_opt == "anneal" or args.lr_opt == "wave"):
        sched_plateau.step(loss)
    elif args.lr_opt == "anneal":
        sched_anneal.step(i)
    elif args.lr_opt == "wave":
        sched_wave.step(i)
    elif args.lr_opt == "plateau":
        if i > int(args.max_iterations*args.plateau_delay):
            sched_plateau.step(loss)


i = 0
try:
    with tqdm() as pbar:
        while True:
            train(i)
            if (i >= args.max_iterations or args.exit_early):
                if (max_overtime > 0) and (args.lr_opt in ("anneal", "wave", "plateau")) and (round(current_lr, 8) > round(plateau_min_lr, 8)) and (i < args.max_iterations * max_overtime):
                    # if overtime is allowed, some extra frames can be appended.
                    if i == args.max_iterations:
                        # only print this message once! (if overtime triggeres, but i is equal to max)
                        print(f"\nplateau lr:{round(current_lr, 8)} still greater than min: {round(plateau_min_lr, 8)} allowing up to {max_overtime-1} overtime (until i: {int(args.max_iterations * max_overtime)}).")
                        print("This factor can be changed with -ovf, set to 0 to disable.")
                    # pass
                else:
                    break
            i += 1
            pbar.update()
except KeyboardInterrupt:
    pass


for i in range(1,5):
    # attempt to save final image up to 5 times
    try:
        output_frames[-1].save(png_file_path)
    except Exception as e:
        print(e)
        print(f"png save failed, attempt: {i+1} / 5")
        time.sleep(1)
        continue
    break


# function to write a list of frames to a gif file
def gif_frames_to_file(frames, path):
    for i in range(1,5):
        # attempt to save gif up to 5 times
        try:
            # loop = 0 may be added to have the gif loop.
            generated_image.save(fp=path, format='GIF', append_images=frames, save_all=True, duration=1, minimize_size=True)
        except Exception as e:
            print(e)
            print(f"gif save failed, attempt: {i+1} / 5")
            time.sleep(1)
            continue
        break

# write the gif file. drop every n frames, increasing n, until the gif is small enough for the size limit
try:
    gif_frames = output_frames
    gif_filename = "outputs/"+args.output+timestamp+".gif"
    gif_frames_to_file(gif_frames, gif_filename)
    gif_size = os.path.getsize(gif_filename)
    skip_n = 0
    max_gif_size_b = args.max_gif_size_mb *1024*1024
    while gif_size > max_gif_size_b:
        print(f"optimising gif size: skipping at a rate of {skip_n}")
        # heuristically accelerate the process: if size is off by more than factor 10, double the amount of drops
        # skip_n may not be multiplied if it is still zero
        if (gif_size > max_gif_size_b * 10) and (skip_n > 0):
            skip_n *= 2
        else:
            skip_n += 1
        # put a copy of the last frame at the start. the final image acts as a 'thumbnail'
        gif_frames = []
        gif_frames.append(output_frames[-1])

        for i in range(len(output_frames)):
            # the first and last frames are always kept
            if (i%skip_n == 0) or (i == 0) or (i == len(output_frames)-1):
                gif_frames.append(output_frames[i])

        # write the new sequence and check file size
        gif_frames_to_file(gif_frames, gif_filename)
        gif_size = os.path.getsize(gif_filename)
    print(f"Skipping every {skip_n} frames to achieve gif size limit: {gif_size}/{max_gif_size_b} ({args.max_gif_size_mb}MB, configurable with -mgs)")
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(e,exc_tb.tb_lineno)

if should_make_video:
    video_file_path = "outputs/"+args.output+timestamp
    print("Calling ffmpeg for video creation... Video output can be disabled with -nvd.")
    cmd_out = ['ffmpeg',
           '-f', 'image2pipe',
           '-vcodec', 'png',
           '-r', '30',  # FPS 
           '-i', '-',  # Indicated input comes from pipe 
           '-qscale', '0',
           video_file_path+"n.mp4"]
    null_out = open(os.devnull, 'wb')
    try:
        pipe = sp.Popen(cmd_out, stdin=sp.PIPE, shell=False, stdout=null_out, stderr=null_out)
    except FileNotFoundError as e:
        print("Video creation requires ffmpeg executable! Install or place a ffmpeg binary in this folder.")
        # simply exit with the error, there is no need to continue attempting to use ffmpeg
        raise e

    # add final frame to the front as a 'cover'
    output_frames[-1].save(pipe.stdin, format='PNG')
    for img in output_frames:
        img.save(pipe.stdin, format='PNG')
    pipe.stdin.close()
    pipe.wait()
    if pipe.returncode != 0:
        print(sp.CalledProcessError(pipe.returncode, cmd_out))
    else:
        print("ffmpeg call returned code 0.")

    print("Attaching final image as thumbnail via ffmpeg...")
    cmd_out = ['ffmpeg', '-i', video_file_path+"n.mp4", '-i', png_file_path, '-map', '0', '-map', '1', '-c', 'copy', '-c:v:1', 'png', '-disposition:v:1', 'attached_pic', video_file_path+".mp4", '-y']
    pipe = sp.Popen(cmd_out, stdin=sp.PIPE, shell=False, stdout=null_out, stderr=null_out)
    pipe.wait()
    if pipe.returncode != 0:
        print(sp.CalledProcessError(pipe.returncode, cmd_out))
    else:
        print("ffmpeg call returned code 0. Removing intermediary file without thumbnail.")
        os.remove(video_file_path+"n.mp4")
