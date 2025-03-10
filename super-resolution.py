import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def SuperRes(img_path):
    model_name = "RealESRGAN_x4plus_finetune_1000iter"
    model_path = os.path.join("weights", model_name + ".pth")
    netscale = 4
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )

    f_input = img_path
    output_dir = "results/finetune_1000iter"
    os.makedirs(output_dir, exist_ok=True)

    denoise_strength = 0.5
    outscale = 4
    suffix = "_out"
    tile = 0
    tile_pad = 10
    pre_pad = 0
    fp32 = False
    alpha_upsampler = "realesrgan"
    ext = "auto"
    gpu_id = 0
    dni_weight = None

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id,
    )

    if os.path.isfile(f_input):
        paths = [f_input]
    else:
        paths = sorted(glob.glob(os.path.join(f_input, "*")))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print("Testing", idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = "RGBA"
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print("Error", error)
            print(
                "If you encounter CUDA out of memory, try to set --tile with a smaller number."
            )
        else:
            if ext == "auto":
                extension = extension[1:]
            else:
                extension = ext
            if img_mode == "RGBA":  # RGBA images should be saved in png format
                extension = "png"
            if suffix == "":
                save_path = os.path.join(output_dir, f"{imgname}.{extension}")
            else:
                save_path = os.path.join(output_dir, f"{imgname}_{suffix}.{extension}")
                print(save_path)
            cv2.imwrite(save_path, output)


SuperRes("inputs/detected_objects")