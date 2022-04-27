import argparse
import imageio
import os
import matplotlib.pyplot as plt

from coperception.datasets import V2XSimDet, V2XSimSeg
from coperception.configs import Config, ConfigGlobal
from coperception.utils.obj_util import *


def save_rendered_image():
    folder_name = os.path.join(
        "./logs", "visualization", f"agent{agent_idx}/scene_{current_scene_idx}"
    )
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(f"{folder_name}/{idx}.png", bbox_inches="tight")


def render_gif():

    output_gif_dir = f"./logs/visualization/agent{agent_idx}/scene_{last_scene_idx}"
    # if no image output for the last scene
    if not os.path.exists(output_gif_dir):
        return

    print(f"Rendering gif for scene {last_scene_idx}...")
    output_gif_inner_dir = f"{output_gif_dir}/gif"

    images_path_list = [
        f.split(".") for f in os.listdir(output_gif_dir) if f.endswith(".png")
    ]
    images_path_list.sort(key=lambda x: int(x[0]))
    images_path_list = [
        f'{output_gif_dir}/{".".join(file)}' for file in images_path_list
    ]

    ims = [imageio.imread(file) for file in images_path_list]
    os.makedirs(output_gif_inner_dir, exist_ok=True)
    output_gif_path = f"{output_gif_inner_dir}/out.gif"
    imageio.mimwrite(output_gif_path, ims, fps=5)

    print(f"Rendered {output_gif_path}")


def set_title_for_axes(ax, title):
    ax.title.set_text(title)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_idx", default=0, type=int, help="which agent")
    parser.add_argument("--split", default="train", type=str)
    args = parser.parse_args()

    split = args.split
    config = Config(binary=True, split=split, use_vis=True)

    agent_idx = args.agent_idx
    data_path = f"/scratch/dm4524/data/V2X-Sim-seg/{split}/agent{agent_idx}"
    data_path_det = f"/scratch/dm4524/data/V2X-Sim-det/{split}/agent{agent_idx}"

    data_carscenes = V2XSimSeg(
        dataset_roots=[data_path],
        split=split,
        config=config,
        val=True,
        kd_flag=True,
        bound="both",
        no_cross_road=False,
    )

    data_carscenes_no_cross_road = V2XSimSeg(
        dataset_roots=[data_path],
        split=split,
        config=config,
        val=True,
        kd_flag=True,
        bound="both",
        no_cross_road=True,
    )

    config_global = ConfigGlobal(binary=True, split=split)
    data_carscenes_det = V2XSimDet(
        dataset_roots=[data_path_det],
        split=split,
        config_global=config_global,
        config=config,
        val=True,
        bound="both",
    )
    data_carscenes_det_no_cross_road = V2XSimDet(
        dataset_roots=[data_path_det],
        split=split,
        config_global=config_global,
        config=config,
        val=True,
        no_cross_road=True,
        bound="both",
    )

    last_scene_idx = data_carscenes.seq_scenes[0][0]
    for idx in range(len(data_carscenes)):
        if idx % 20 == 0:
            print(f"Currently at frame {idx} / {len(data_carscenes)}")

        padded_voxel_points, padded_voxel_points_teacher, seg_bev = data_carscenes[idx][
            0
        ]
        (
            padded_voxel_points_ncr,
            padded_voxel_points_teacher_ncr,
            seg_bev,
        ) = data_carscenes_no_cross_road[idx][0]

        (
            padded_voxel_points_det,
            padded_voxel_points_teacher_det,
            label_one_hot,
            reg_target,
            reg_loss_mask,
            anchors_map,
            vis_maps,
            gt_max_iou,
            filename,
            target_agent_id,
            num_sensor,
            trans_matrix,
        ) = data_carscenes_det[idx][0]

        # empty
        if num_sensor == 0:
            continue

        (
            padded_voxel_points_det_ncr,
            padded_voxel_points_teacher_det_ncr,
            label_one_hot,
            reg_target,
            reg_loss_mask,
            _,
            vis_maps,
            gt_max_iou,
            filename,
            target_agent_id,
            num_sensor,
            trans_matrix,
        ) = data_carscenes_det_no_cross_road[idx][0]

        plt.clf()
        for p in range(data_carscenes.pred_len):
            plt.xlim(0, 256)
            plt.ylim(0, 256)

            # just put it here so that we know the ratio is adjustable
            gs_kw = dict(width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])
            fig, axes = plt.subplot_mosaic(
                [
                    ["a", "b", "c", "seg1"],
                    ["d", "e", "f", "seg2"],  # seg 1 and seg 2 are the same
                ],
                gridspec_kw=gs_kw,
                figsize=(8, 5),
                constrained_layout=True,
            )

            plt.axis("off")

            # convert tensors to run on CPU
            padded_voxel_points = padded_voxel_points.cpu().detach().numpy()
            padded_voxel_points_teacher = (
                padded_voxel_points_teacher.cpu().detach().numpy()
            )
            padded_voxel_points_teacher_ncr = (
                padded_voxel_points_teacher_ncr.cpu().detach().numpy()
            )

            axes["a"].imshow(
                np.max(padded_voxel_points.reshape(256, 256, 13), axis=2),
                alpha=1.0,
                zorder=12,
            )
            set_title_for_axes(axes["a"], "seg single-view")

            axes["b"].imshow(
                np.max(padded_voxel_points_teacher.reshape(256, 256, 13), axis=2),
                alpha=1.0,
                zorder=12,
            )
            set_title_for_axes(axes["b"], "seg multi-view\n w/ cross")

            axes["c"].imshow(
                np.max(padded_voxel_points_teacher_ncr.reshape(256, 256, 13), axis=2),
                alpha=1.0,
                zorder=12,
            )
            set_title_for_axes(axes["c"], "seg multi-view\n w/o cross")

            # ===== det
            def draw_gt_box(ax):
                gt_max_iou_idx = gt_max_iou[0]["gt_box"]
                for p in range(data_carscenes_det.pred_len):

                    for k in range(len(gt_max_iou_idx)):
                        anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]
                        decode_corner = center_to_corner_box2d(
                            np.asarray([anchor[:2]]),
                            np.asarray([anchor[2:4]]),
                            np.asarray([anchor[4:]]),
                        )[0]

                        corners = coor_to_vis(
                            decode_corner,
                            data_carscenes_det.area_extents,
                            data_carscenes_det.voxel_size,
                        )

                        c_x, c_y = np.mean(corners, axis=0)
                        corners = np.concatenate([corners, corners[[0]]])

                        ax.plot(
                            corners[:, 0],
                            corners[:, 1],
                            c="g",
                            linewidth=2.0,
                            zorder=20,
                        )
                        ax.scatter(c_x, c_y, s=3, c="g", zorder=20)
                        ax.plot(
                            [c_x, (corners[1][0] + corners[0][0]) / 2.0],
                            [c_y, (corners[1][1] + corners[0][1]) / 2.0],
                            linewidth=2.0,
                            c="g",
                            zorder=20,
                        )

                    occupy = np.max(vis_maps, axis=-1)
                    m = np.stack([occupy, occupy, occupy], axis=-1)
                    m[m > 0] = 0.99
                    occupy = (m * 255).astype(np.uint8)
                    # -----------#
                    free = np.min(vis_maps, axis=-1)
                    m = np.stack([free, free, free], axis=-1)
                    m[m < 0] = 0.5
                    free = (m * 255).astype(np.uint8)
                    # ---------------

            draw_gt_box(axes["d"])
            axes["d"].imshow(
                np.max(padded_voxel_points_det.reshape(256, 256, 13), axis=2),
                alpha=1.0,
                zorder=12,
            )
            set_title_for_axes(axes["d"], "det single-view")

            draw_gt_box(axes["e"])
            axes["e"].imshow(
                np.max(padded_voxel_points_teacher_det.reshape(256, 256, 13), axis=2),
                alpha=1.0,
                zorder=12,
            )
            set_title_for_axes(axes["e"], "det multi-view\n w/ cross")
            # ========

            draw_gt_box(axes["f"])
            axes["f"].imshow(
                np.max(
                    padded_voxel_points_teacher_det_ncr.reshape(256, 256, 13), axis=2
                ),
                alpha=1.0,
                zorder=12,
            )
            set_title_for_axes(axes["f"], "det muti-view\n w/o cross")

            seg_image = np.zeros(shape=(256, 256, 3), dtype=np.dtype("uint8"))

            for k in ("seg1", "seg2"):
                for key, value in config.class_to_rgb.items():
                    seg_image[np.where(seg_bev == key)] = value
                axes[k].imshow(seg_image)
                set_title_for_axes(axes[k], "seg")

            current_scene_idx = data_carscenes.seq_scenes[0][idx]

            save_rendered_image()
            if (
                current_scene_idx != last_scene_idx or idx == len(data_carscenes) - 1
            ):  # last scene finishes, output gif
                render_gif()
                last_scene_idx = current_scene_idx

            plt.close("all")

    if (
        data_carscenes.seq_scenes[0][idx] != last_scene_idx
        or idx == len(data_carscenes) - 1
    ):  # last scene finishes, output gif
        render_gif()
        last_scene_idx = data_carscenes.seq_scenes[0][idx]
