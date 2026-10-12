import os
import multiprocessing
from glob import glob
import logging

import numpy as np
from transforms3d import quaternions as tq

from dexonomy.sim import MuJoCo_VisEnv, HandCfg, MuJoCo_EvalCfg, MuJoCo_OptCfg
from dexonomy.util.usd_util import UsdStage, Material
from dexonomy.util.file_util import get_template_names, load_scene_cfg, safe_wrapper


@safe_wrapper
def _single_vusd(param):
    npy_path, cfg = param[0], param[1]
    op_cfg = cfg.op

    data = np.load(npy_path, allow_pickle=True).item()

    if cfg.legacy_api:
        scene_cfg = data["scene_cfg"]
    else:
        scene_cfg = load_scene_cfg(data["scene_path"])
    
    if op_cfg.data == "succ_traj":
        vis_env = MuJoCo_VisEnv(
            hand_cfg=HandCfg(cfg.hand.arm_hand.xml_path),
            scene_cfg=scene_cfg,
            sim_cfg=MuJoCo_EvalCfg(),
            vis_mode=op_cfg.hand.mode,
        )
    else:
        vis_env = MuJoCo_VisEnv(
            hand_cfg=HandCfg(cfg.hand.xml_path, freejoint=True),
            scene_cfg=scene_cfg,
            sim_cfg=MuJoCo_OptCfg(),
            vis_mode=op_cfg.hand.mode,
        )
    all_body_name = vis_env.get_all_body_names()
    all_body_mesh = [vis_env.get_body_mesh(body_name) for body_name in all_body_name]
    all_body_pose = []
    if isinstance(op_cfg.qpos, str):
        qpos_name_lst = [op_cfg.qpos]
    else:
        qpos_name_lst = op_cfg.qpos
    for qpos_name in qpos_name_lst:
        for qpos_id in range(data[f"{qpos_name}_qpos"].shape[0]):
            xmat, xpos = vis_env.forward_kinematics(data[f"{qpos_name}_qpos"][qpos_id])
            body_pose = []
            for body_name in all_body_name:
                body_id = vis_env.get_body_id(body_name)
                body_pose.append(
                    np.concatenate(
                        [xpos[body_id], tq.mat2quat(xmat[body_id]), np.ones(1)]
                    )
                )
            body_pose = np.stack(body_pose, axis=0)
            all_body_pose.append(body_pose)

    return {
        "body_name": all_body_name,
        "body_mesh": all_body_mesh,
        "body_pose": np.stack(all_body_pose, axis=0),
    }


def operate_vusd(cfg):
    op_cfg = cfg.op
    if op_cfg.data == "init":
        data_dir, check_dir = cfg.init_dir, cfg.grasp_dir
    elif op_cfg.data == "grasp":
        data_dir, check_dir = cfg.grasp_dir, cfg.succ_grasp_dir
        if not os.path.exists(check_dir):
            check_dir = cfg.succ_traj_dir
    elif op_cfg.data == "succ_grasp":
        data_dir, check_dir = cfg.succ_grasp_dir, None
    elif op_cfg.data == "succ_traj":
        data_dir, check_dir = cfg.succ_traj_dir, None
    elif op_cfg.data == "new_tmpl":
        data_dir, check_dir = cfg.new_tmpl_dir, None
    else:
        raise NotImplementedError(
            f"Valid choices: 'init', 'grasp', 'succ_grasp', 'succ_traj', 'new_tmpl'. Current: '{op_cfg.data}' "
        )

    if not os.path.isdir(data_dir):
        logging.warning(f"Data directory does not exist: {data_dir}")
        return

    # Output data is grouped by template/taxonomy. Respecting tmpl_name avoids
    # scanning every released grasp category when only one is requested.
    tmp_lst = sorted(get_template_names(cfg.tmpl_name, data_dir))

    for temp_name in tmp_lst:
        input_path_lst = glob(
            os.path.join(data_dir, temp_name, "**/*.npy"), recursive=True
        )
        if cfg.debug_name is not None:
            input_path_lst = [p for p in input_path_lst if cfg.debug_name in p]

        if check_dir is not None and op_cfg.succ is not None:
            check_path_lst = glob(
                os.path.join(check_dir, temp_name, "**/*.npy"), recursive=True
            )
            if cfg.debug_name is not None:
                check_path_lst = [p for p in check_path_lst if cfg.debug_name in p]
            check_path_lst = [p.replace(check_dir, data_dir) for p in check_path_lst]

            if op_cfg.succ:
                input_path_lst = list(set(input_path_lst).intersection(check_path_lst))
            else:
                input_path_lst = list(set(input_path_lst).difference(check_path_lst))

        input_path_lst = sorted(input_path_lst)
        if len(input_path_lst) == 0:
            continue
        logging.info(
            f"Find {len(input_path_lst)} in {data_dir}. Debug name: {cfg.debug_name}. Check success: {op_cfg.succ}"
        )

        start = max(0, int(op_cfg.start))
        input_path_lst = input_path_lst[start:]
        if op_cfg.n_max > 0:
            input_path_lst = input_path_lst[: int(op_cfg.n_max)]
        if len(input_path_lst) == 0:
            logging.warning(f"No data remains for {temp_name} at start index {start}")
            continue
        logging.info(f"Use {len(input_path_lst)} ordered samples from index {start}")

        if op_cfg.chunk_size is None:
            chunk_size = len(input_path_lst)
        else:
            chunk_size = int(op_cfg.chunk_size)
            if chunk_size <= 0:
                raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        with multiprocessing.Pool(processes=cfg.n_worker) as pool:
            for chunk_offset in range(0, len(input_path_lst), chunk_size):
                chunk_paths = input_path_lst[chunk_offset : chunk_offset + chunk_size]
                param_lst = [(path, cfg) for path in chunk_paths]
                # imap preserves the sorted input order, which defines playback order.
                raw_results = list(pool.imap(_single_vusd, param_lst))
                valid_results = [
                    (path, result)
                    for path, result in zip(chunk_paths, raw_results)
                    if isinstance(result, dict)
                ]

                if len(valid_results) == 0:
                    logging.warning(
                        f"No valid visualization result for {temp_name} chunk {chunk_offset}"
                    )
                    continue

                absolute_start = start + chunk_offset
                absolute_end = absolute_start + len(chunk_paths) - 1
                output_stem = temp_name.replace("**", "all")
                if op_cfg.chunk_size is not None or start > 0:
                    output_stem += f"_{absolute_start:06d}_{absolute_end:06d}"
                save_path = os.path.join(cfg.vis_usd_dir, "usd", f"{output_stem}.usd")
                manifest_path = save_path.replace(".usd", ".manifest.tsv")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                total_frames = sum(
                    result["body_pose"].shape[0] for _, result in valid_results
                )
                usd_stage = UsdStage(save_path, timestep=total_frames, dt=0.01)

                count = 0
                manifest_rows = ["start_frame\tend_frame_exclusive\tdataset_path"]
                for data_path, result in valid_results:
                    data_length = result["body_pose"].shape[0]
                    usd_stage.add_mesh_lst(
                        result["body_mesh"],
                        result["body_name"],
                        result["body_pose"],
                        [(count, count + data_length)] * len(result["body_name"]),
                        material=Material(color=cfg.hand.color, name="obj"),
                    )
                    manifest_rows.append(
                        f"{count}\t{count + data_length}\t{os.path.relpath(data_path, data_dir)}"
                    )
                    count += data_length

                logging.info(f"Save to {os.path.abspath(save_path)}")
                usd_stage.write_stage_to_file(save_path)
                with open(manifest_path, "w") as manifest_file:
                    manifest_file.write("\n".join(manifest_rows) + "\n")
