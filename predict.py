""" Cog interface for 3D Moments from Near-Duplicate Photos"""
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import glob
import tempfile
import time
from operator import indexOf

import cv2
import imageio
import torch.utils.data.distributed
import torchvision
from cog import BasePredictor, Input, Path
from PIL import Image, ImageFile
from tqdm import tqdm

import config
from core.inpainter import Inpainter
from core.renderer import ImgRenderer
from core.scene_flow import SceneFlowEstimator
from core.utils import *
from data_loaders.data_utils import resize_img
from demo import *
from model import SpaceTimeModel, get_raft_model
from third_party.DPT.run_monodepth import run_dpt
from third_party.RAFT.core.utils.utils import InputPadder
from utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        args = set_default_args()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading model checkpoint......")
        model = SpaceTimeModel(args)
        if model.start_step == 0:
            raise Exception("no pretrained model found! please check the model path.")

        scene_flow_estimator = SceneFlowEstimator(args, model.raft)
        inpainter = Inpainter(args)
        renderer = ImgRenderer(args, model, scene_flow_estimator, inpainter, device)

        model.switch_to_eval()

        self.args = args
        self.renderer = renderer
        print("Done setting up")

    def predict(
        self,
        image1: Path = Input(description="Input image #1"),
        image2: Path = Input(
            description="Input image #2; should be near duplicate to image 1"
        ),
        effect: str = Input(
            description="Video animation effect",
            choices=["up-down", "zoom-in", "side", "circle"],
        ),
    ) -> Path:

        effect = str(effect)

        # make it work with pngs
        image1, image2 = str(image1), str(image2)
        im1 = Image.open(image1).convert("RGB")
        im1.save("0.png")
        im2 = Image.open(image2).convert("RGB")
        im2.save("1.png")

        print("Setting arguments......")

        homography_warp_pairs(self.args)

        print("=========================run 3D Moments...=========================")

        data = get_input_data(self.args)
        rgb_file1 = data["src_rgb_file1"][0]
        rgb_file2 = data["src_rgb_file2"][0]

        video_out_folder = os.path.join(self.args.input_dir, "out")
        os.makedirs(video_out_folder, exist_ok=True)

        print("Performing model inference......")

        with torch.no_grad():
            self.renderer.process_data(data)

            (
                pts1,
                pts2,
                rgb1,
                rgb2,
                feat1,
                feat2,
                mask,
                side_ids,
                optical_flow,
            ) = self.renderer.render_rgbda_layers_with_scene_flow(return_pts=True)

            num_frames = [60, 60, 60, 90]
            video_paths = ["up-down", "zoom-in", "side", "circle"]
            Ts = [
                define_camera_path(
                    num_frames[0],
                    0.0,
                    -0.08,
                    0.0,
                    path_type="double-straight-line",
                    return_t_only=True,
                ),
                define_camera_path(
                    num_frames[1],
                    0.0,
                    0.0,
                    -0.24,
                    path_type="straight-line",
                    return_t_only=True,
                ),
                define_camera_path(
                    num_frames[2],
                    -0.09,
                    0,
                    -0,
                    path_type="double-straight-line",
                    return_t_only=True,
                ),
                define_camera_path(
                    num_frames[3],
                    -0.04,
                    -0.04,
                    -0.09,
                    path_type="circle",
                    return_t_only=True,
                ),
            ]
            crop = 32

            effect2idx = {"up-down": 0, "zoom-in": 1, "side": 2, "circle": 3}

            j = effect2idx[effect]  # index
            T = Ts[j]
            T = torch.from_numpy(T).float().to(self.renderer.device)
            time_steps = np.linspace(0, 1, num_frames[j])
            frames = []
            for i, t_step in tqdm(
                enumerate(time_steps),
                total=len(time_steps),
                desc="generating video of {} camera trajectory".format(video_paths[j]),
            ):
                pred_img, _, meta = self.renderer.render_pcd(
                    pts1,
                    pts2,
                    rgb1,
                    rgb2,
                    feat1,
                    feat2,
                    mask,
                    side_ids,
                    t=T[i],
                    time=t_step,
                )
                frame = (
                    255.0 * pred_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                ).astype(np.uint8)
                # mask out fuzzy image boundaries due to no outpainting
                img_boundary_mask = (
                    (meta["acc"] > 0.5)
                    .detach()
                    .cpu()
                    .squeeze()
                    .numpy()
                    .astype(np.uint8)
                )
                img_boundary_mask_cleaned = process_boundary_mask(img_boundary_mask)
                frame = frame * img_boundary_mask_cleaned[..., None]
                frame = frame[crop:-crop, crop:-crop]
                frames.append(frame)

            video_out_file = Path(tempfile.mkdtemp()) / "out.mp4"

            print("Writing output video......")

            imageio.mimwrite(str(video_out_file), frames, fps=25, quality=8)

            return video_out_file


def set_default_args():
    ########## training options ##########\
    class Args:
        def __init__(self):
            pass

    args = Args()
    args.adaptive_pts_radius = True
    args.batch_size = 1
    args.boundary_crop_ratio = 0
    args.ckpt_path = "pretrained/model_250000.pth"
    args.config = "configs/render.txt"
    args.dataset_weights = []
    args.distributed = False
    args.eval_dataset = "jamie"
    args.eval_mode = True
    args.expname = "exp"
    args.feature_dim = 32
    args.i_img = 500
    args.i_print = 100
    args.i_weights = 10000
    args.inpainting = False
    args.input_dir = ""
    args.local_rank = 0
    args.loss_mode = "vgg19"
    args.lr = 0.0003
    args.lr_raft = 5e-06
    args.lrate_decay_factor = 0.5
    args.lrate_decay_steps = 50000
    args.n_iters = 250000
    args.no_load_opt = True
    args.no_load_scheduler = True
    args.no_reload = False
    args.point_radius = 1.5
    args.rootdir = "./"
    args.train_dataset = "tiktok"
    args.train_raft = False
    args.use_depth_for_decoding = True
    args.use_depth_for_feature = False
    args.use_inpainting_mask_for_feature = False
    args.use_mask_for_decoding = False
    args.vary_pts_radius = False
    args.visualize_rgbda_layers = False
    args.workers = 8

    return args
