"""
Run segmentation inference over one or more manifest-defined datasets.
"""

import argparse
import os

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist

from monai import data, transforms
from monai.data import load_decathlon_datalist, decollate_batch

from monai.inferers import sliding_window_inference

from models import smit, configs_smit
from models import swinunetr
from project_paths import RESULTS_ROOT

from tqdm import tqdm

def list_of_strings(arg):
    return arg.split(',')

parser = argparse.ArgumentParser(description="Segmentation pipeline")
parser.add_argument("--data_dir", default=None, type=str, help="dataset directory")
parser.add_argument("--dataset_class", default='main', type=str, help="dataset class type")
parser.add_argument("--model_name", default=None, type=str, help="swinunetr, smit, smitlite")
parser.add_argument("--json_list", default=None, type=str, help="dataset json file")
parser.add_argument("--datasets", default=None, type=list_of_strings, help="list of datasets to pull from json file")
parser.add_argument("--pretrained_model_path", default=None, type=str, help="pretrained model path")
parser.add_argument("--organ_eval", default=None, type=str, help="inference organ for model")
parser.add_argument("--output_dir", default=None, type=str, help="output image directory")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--debug", action="store_true", help="used before testing any new code")
parser.add_argument("--orientation", default="RAS", type=str, help="orientation axcodes for Orientationd (e.g. RAS, PRI)")
parser.add_argument("--overwrite", action="store_true", help="overwrite existing output directories")
parser.add_argument("--distributed", action="store_true", help="use multi-GPU inference via torch DDP")
parser.add_argument("--save_npy", action="store_true", help="save raw softmax .npy outputs (can be large)")
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--sw_batch_size", default=8, type=int, help="sliding window batch size")

def get_img_name(batch, dataset):
    if dataset == 'test_phantom':
        return 'phantom'
    return os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_master():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def main():

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # DDP setup                                                            #
    # ------------------------------------------------------------------ #
    if args.distributed:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    if is_master():
        print("Working now..., {}".format(args))

    # ------------------------------------------------------------------ #
    # Transforms                                                           #
    # ------------------------------------------------------------------ #
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes=args.orientation),
            transforms.Spacingd(keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear")),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True),
            transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.SpatialPadd(keys=["image"], mode="minimum", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    post_transforms = transforms.Compose([
        transforms.EnsureTyped(keys="pred"),
        transforms.Invertd(
            keys="pred",
            transform=test_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
    ])

    post_transforms_softmax = transforms.AsDiscreted(keys="pred", argmax=True)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    if args.model_name == 'swinunetr':
        model = swinunetr.SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        )
    elif args.model_name == 'smit':
        config = configs_smit.get_SMIT_128_bias_True()
        model = smit.SMIT_3D_Seg(config, out_channels=args.out_channels)
    elif args.model_name == 'smitlite':
        config = configs_smit.get_SMIT_small_96_bias_True()
        model = smit.SMIT_3D_Seg(config, out_channels=args.out_channels)
    else:
        raise NotImplementedError(f"Unknown model: '{args.model_name}'")

    model_dict = torch.load(args.pretrained_model_path, map_location="cpu")["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    if is_master():
        print("Finished loading model dictionary...")

    # ------------------------------------------------------------------ #
    # Per-dataset inference                                                #
    # ------------------------------------------------------------------ #
    for dataset in args.datasets:
        if is_master():
            print('Working on', dataset)

        output_directory = os.path.join(
            str(RESULTS_ROOT),
            '{}_{}'.format(args.model_name, args.dataset_class),
            '{}_{}_{}'.format(args.organ_eval, dataset, args.output_dir)
        )

        # Only master creates/checks dirs; barrier ensures they exist on all ranks
        if is_master():
            if os.path.exists(output_directory) and not args.overwrite:
                print(f"Output directory already exists, skipping: {output_directory}")
                open(os.path.join(output_directory, '.skip'), 'w').close()
            else:
                skip_flag = os.path.join(output_directory, '.skip')
                if os.path.exists(skip_flag):
                    os.remove(skip_flag)
                os.makedirs(os.path.join(output_directory, 'numpy'), exist_ok=True)
                os.makedirs(os.path.join(output_directory, 'nii'), exist_ok=True)

        if args.distributed:
            dist.barrier()

        if os.path.exists(os.path.join(output_directory, '.skip')):
            continue

        datalist_json = os.path.join(args.data_dir, args.json_list)
        test_files = load_decathlon_datalist(datalist_json, True, dataset, base_dir=args.data_dir)
        test_ds = data.Dataset(test_files, transform=test_transform)

        if args.distributed:
            # Each rank processes a non-overlapping shard — no duplicate .npy writes
            sampler = torch.utils.data.distributed.DistributedSampler(
                test_ds, shuffle=False
            )
            test_loader = data.DataLoader(
                test_ds, batch_size=1, shuffle=False,
                sampler=sampler, pin_memory=True
            )
        else:
            test_loader = data.DataLoader(
                test_ds, batch_size=1, shuffle=False,
                pin_memory=True, num_workers=args.num_workers
            )

        with torch.no_grad():
            for batch in tqdm(test_loader, disable=not is_master()):

                val_inputs = batch["image"].to(device)
                img_name = get_img_name(batch, dataset).replace(".nii.gz", "")

                batch["pred"] = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.infer_overlap,
                    mode="gaussian",
                    progress=False
                )

                batch = [post_transforms(i) for i in decollate_batch(batch)]

                # .npy is opt-in — float32 softmax over [2, H, W, D] can be ~100MB+ per volume
                if args.save_npy:
                    np.save(
                        os.path.join(output_directory, 'numpy', img_name + ".npy"),
                        batch[0]["pred"].cpu()
                    )

                seg_batch = post_transforms_softmax(batch[0])
                seg_ori_size = seg_batch['pred'].cpu().numpy().astype(np.uint8)
                seg_ori_size = np.squeeze(seg_ori_size)

                affine = batch[0]["image_meta_dict"].get("original_affine", np.eye(4))
                nib.save(
                    nib.Nifti1Image(seg_ori_size, affine),
                    os.path.join(output_directory, 'nii', img_name + ".nii.gz")
                )

        if args.distributed:
            dist.barrier()  # all ranks finish before moving to next dataset

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
