import json
import numpy as np
import os
import re
import sys
import torch
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from tqdm import tqdm
from robomimic.state_infuse.get_state_awarness_of_openai import get_internal_state_form_openai
import pickle
import cv2
import re
import ast

TASK_MAPPING_50_DEMO = {
    "PnPCounterToCab": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5",
   "TurnOffMicrowave": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5"
}


def format_image(image):
    image = np.array(image)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def extract_and_export_image(all_demo_dataset):
    task_progress_states_mapping = {
        # (task, progress) : state
    }

    for di, demo_dataset in enumerate(all_demo_dataset.datasets):
        exporting_dataset = demo_dataset

        eye_names = ['robot0_agentview_left_image', 'robot0_eye_in_hand_image', 'robot0_agentview_right_image']

        print('PROCESSING... dataset index with: ', di)

        for i in tqdm(range(len(exporting_dataset))):
            if i > 100: break

            left_image = exporting_dataset[i]['obs'][eye_names[0]][0]
            hand_image = exporting_dataset[i]['obs'][eye_names[1]][0]
            right_image = exporting_dataset[i]['obs'][eye_names[2]][0]

            print('left_image shape: ', left_image.shape)

            demo_id = exporting_dataset._index_to_demo_id[i]
            demo_start_index = exporting_dataset._demo_id_to_start_indices[demo_id]
            demo_length = exporting_dataset._demo_id_to_demo_length[demo_id]

            # start at offset index if not padding for frame stacking
            demo_index_offset = 0 if exporting_dataset.pad_frame_stack else (exporting_dataset.n_frame_stack - 1)
            index_in_demo = i - demo_start_index + demo_index_offset

            complete_rate = round(index_in_demo / demo_length, 0)

            task_description = exporting_dataset._demo_id_to_demo_lang_str[demo_id]

            save_key = (task_description, complete_rate)

            if save_key not in task_progress_states_mapping:
                print(f'getting task {task_description} in progress {complete_rate} from openai')

                left_image = format_image(left_image)
                hand_image = format_image(hand_image)
                right_image = format_image(right_image)

                try:
                    internal_state = get_internal_state_form_openai(
                        left_image, hand_image,
                        right_image,
                        complete_rate,
                        task_description,
                        with_complete_rate=True,
                        write_image=False,
                        with_image_format_change=False
                    )

                    internal_state = re.sub(r'[\n\t\s]+', '', internal_state)
                    internal_state = internal_state.replace('python', '')

                    internal_state = ast.literal_eval(internal_state)

                    print('get response: ', internal_state)

                    task_progress_states_mapping[save_key] = internal_state
                except Exception as e:
                    print('get error: ', e)
                    print('when processing task: ', task_description, ' with progress: ', complete_rate)

        with open('task_progress_states_mapping.pkl', 'wb') as f:
            pickle.dump(task_progress_states_mapping, f)


def generate_concated_images_from_demo_path(task_name=None, file_path=None):
    config_path_compsoite = "/home/minquangao/completion-infuse-robot/robomimic/scripts/run_configs/seed_123_ds_human-50.json"
    # config_path_compsoite = "/home/minquangao/pretrained_models/configs/seed_123_ds_human-50.json"
    ext_cfg = json.load(open(config_path_compsoite, 'r'))

    if task_name:
        ext_cfg['train']['data'].append({'path':file_path if file_path else TASK_MAPPING_50_DEMO.get(task_name)})
        # print('loading from path ', TASK_PATH_MAPPING[task_name])
    else:
        for path in TASK_MAPPING_50_DEMO.values():
            ext_cfg['train']['data'].append({'path': path})

    config = config_factory(ext_cfg["algo_name"])

    with config.values_unlocked():
        config.update(ext_cfg)

    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    # print("\n============= New Training Run with Config =============")
    # print(config)
    # print("")
    # print(config)
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        # print("\n============= Loaded Environment Metadata =============")
        # print(dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        # update env meta if applicable
        from self_correct_robot.utils.script_utils import deep_update
        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=False
        )
        shape_meta_list.append(shape_meta)

    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], lang_encoder=None)

    # TODO: combine trainset and validset
    demo_dataset = trainset

    extract_and_export_image(demo_dataset)


if __name__ == '__main__':
    # import argparse
    #
    #
    # parser = argparse.ArgumentParser(description='Train a Value Predication Model Via Vision Transformer model.')
    # parser.add_argument('--task_id', type=int, required=True, help='specify the task id to expoert')
    #
    # task_id = parser.parse_args().task_id

    # task_path_mapping = list(TASK_PATH_MAPPING.items())

    # for key, value in TASK_PATH_MAPPING.items():
    #     print('PROCESSING.... ', key)
    #     print('FROM PATH.... ', value)
    generate_concated_images_from_demo_path()



