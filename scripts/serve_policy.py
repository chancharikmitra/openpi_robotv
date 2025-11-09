import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    DROID05 = "DROID05"
    FAST="FAST"
    NON_FAST = "NON_FAST"
    cube_adapted_marker_all_heads = "cube_adapted_marker_all_heads"
    cube_adapted_marker_KNN = "cube_adapted_marker_KNN"
    current = "current"


    KNN_pick_up_red_cube = "KNN_pick_up_red_cube"
    KNN_pick_up_red_mug = "KNN_pick_up_red_mug"
    KNN_push_red_bowl_to_red_cup = "KNN_push_red_bowl_to_red_cup"
    ALL_HEADS_place_green_cube_in_red_bowl = "ALL_HEADS_place_green_cube_in_red_bowl"

    ALL_HEAD_pick_up_red_cube = "ALL_HEAD_pick_up_red_cube"
    ALL_HEADS_pick_up_red_mug = "ALL_HEADS_pick_up_red_mug"
    ALL_HEADS_push_red_bowl_to_red_cup = "ALL_HEADS_push_red_bowl_to_red_cup"

    ALL_HEADS_adapted = "ALL_HEADS_adapted"
    KNN_adapted = "KNN_adapted"
    KNN_place_marker_in_mug = "KNN_place_marker_in_mug"
    ALL_HEADS_place_marker_in_mug = "ALL_HEADS_place_marker_in_mug"   

    KNN_place_green_cube_in_red_bowl = "KNN_place_green_cube_in_red_bowl"

    V10 = "V10"
    V40 = "V40"
    V80 = "V80"
    V1K = "V1K"

    KNN_t4 = "KNN_t4"
    CMA = "CMA"
    REINFORCE = "REINFORCE"

    Sanity_Marker_in_Mug = "Sanity_Marker_in_Mug"

    exp1 = "exp1"
    exp2 = "exp2"
    exp3 = "exp3"

    gradient1 = "gradient1"
    gradient2 = "gradient2"
    gradient3 = "gradient3"
    gradient4 = "gradient4" 

    test = "test"

    # LoRA Table 1
    LoRA_Pick_Up_Red_Cube = "LoRA_Pick_Up_Red_Cube"
    LoRA_place_green_cube_in_red_mug = "LoRA_place_green_cube_in_red_mug"
    LoRA_press_red_button_hard = "LoRA_press_red_button_hard"
    LoRA_place_marker_in_mug = "LoRA_place_marker_in_mug"

    #KNN Table 2
    KNN_T1_place_marker_in_mug = "KNN_T1_place_marker_in_mug"
    KNN_T1_place_green_cube_in_red_bowl = "KNN_T1_place_green_cube_in_red_bowl"
    KNN_T1_press_button_hard = "KNN_T1_press_button_hard"
    KNN_T1_pick_up_red_cube = "KNN_T1_pick_up_red_cube"





@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

TABLE1_LORA = "pi05_All_heads_LoRA",  
TABLE1_KNN = "pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP"


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID05: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_droid",
        dir="gs://openpi-assets/checkpoints/pi0_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.NON_FAST: Checkpoint(
        config="pi0_droid_lerobot_finetune_green_cube",
        dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/9.14_train/pi0_droid_lerobot_finetune_green_cube/debug_lerobot_all_heads_sanity_check_pick_green_cube_droid_joint_velocity_20/4999",
    ),
    EnvMode.FAST: Checkpoint(
        config="pi0_fast_droid_lerobot_finetune_green_cube",
        dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/9.14_train/pi0_fast_droid_lerobot_finetune_green_cube/debug_lerobot_all_heads_sanity_check_pick_green_cube_droid_joint_velocity_20/13000",
    ),
    EnvMode.cube_adapted_marker_all_heads: Checkpoint(
        config="All_heads_LoRA_Adaption",
        dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/All_heads_LoRA_Adaption/pick_up_red_cube_20_adapted_from_place_marker_in_mug_200/4000",
    ),
    EnvMode.cube_adapted_marker_KNN: Checkpoint(
        config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_Adaption",
        dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_Adaption/pick_up_red_cube_20_adapted_from_place_marker_in_mug_200/2000",
    ),
    EnvMode.current: Checkpoint(
         config="pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
        dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/pi05_ckpts/pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/pi05_KNN_place_marker_in_mug_40_new/3000",
    ),

    #Table 1
    #LoRa
    EnvMode.LoRA_Pick_Up_Red_Cube: Checkpoint(
         config= TABLE1_LORA,
         dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/pi05_ckpts/pi05_All_heads_LoRA/pi05_All_heads_LoRA_pick_up_red_cube_20_new/2999",
    ),
    EnvMode.LoRA_place_green_cube_in_red_mug: Checkpoint(
         config = TABLE1_LORA[0], 
         dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/pi05_ckpts/pi05_All_heads_LoRA/pi05_All_heads_LoRA_place_green_cube_in_red_bowl_20_new/2999",
    ),
    EnvMode.LoRA_press_red_button_hard: Checkpoint(
         config= TABLE1_LORA[0],
         dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/pi05_ckpts/pi05_All_heads_LoRA/pi05_All_heads_LoRA_press_red_button_hard_20_new/2999",
    ),
    EnvMode.LoRA_place_marker_in_mug: Checkpoint(
         config= TABLE1_LORA[0],
         dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/pi05_ckpts/pi05_All_heads_LoRA/pi05_All_heads_LoRA_place_marker_in_mug_20_new/2999",
    ),

    #KNN
    EnvMode.KNN_T1_pick_up_red_cube: Checkpoint(
         config= "pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP", #TABLE1_KNN[0]
         dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/pi05_ckpts/finetuned_models_table1/pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/pi05_KNN_pick_up_red_cube_20_new/2999",
    ),
    EnvMode.KNN_T1_place_marker_in_mug: Checkpoint(
         config= TABLE1_KNN,
         dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/pi05_ckpts/finetuned_models_table1/pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/pi05_KNN_place_marker_in_mug_20_new/3000",
    ),

    EnvMode.KNN_T1_place_green_cube_in_red_bowl: Checkpoint(
         config= TABLE1_KNN,
         dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/pi05_ckpts/finetuned_models_table1/pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/pi05_KNN_place_green_cube_in_red_bowl_20_new/2999",
    ),

    EnvMode.KNN_T1_press_button_hard: Checkpoint(
         config= TABLE1_KNN,
         dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/pi05_ckpts/finetuned_models_table1/pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/pi05_KNN_press_red_button_hard_20_new/2999",
    ),


    #Table 2
   #Single Task Training 

    # ########################
    # # Simple Tasks Table 1
    # ##########################
    # EnvMode.ALL_HEADS_pick_up_red_mug: Checkpoint(
    #     config="All_heads_LoRA",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/All_heads_LoRA/table1_pick_up_red_mug_20/4999",
    # ),
    # EnvMode.ALL_HEADS_push_red_bowl_to_red_cup: Checkpoint(
    #     config="All_heads_LoRA",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/All_heads_LoRA/table1_push_red_bowl_to_red_cup_20/4999",
    # ),
    # EnvMode.ALL_HEADS_place_green_cube_in_red_bowl: Checkpoint(
    #     config="All_heads_LoRA",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/All_heads_LoRA/table1_place_green_cube_in_red_bowl_20/4999",
    # ),

    # EnvMode.KNN_pick_up_red_mug: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/table1_pick_up_red_mug_20/4999",
    # ),
    # EnvMode.KNN_push_red_bowl_to_red_cup: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/debug_lerobot_KNN_heads_push_red_bowl_to_red_cup_20/4999",
    # ),
    # EnvMode.KNN_place_green_cube_in_red_bowl: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/9.23_table1_1/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/table1_place_green_cube_in_red_bowl_20/4999",
    # ),
    #  EnvMode.KNN_pick_up_red_cube: Checkpoint(
    #     config="pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/11.7/pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/pi05_KNN_place_marker_in_mug_40_new/4999",
    # ),


    # ##########Already done
    # EnvMode.KNN_pick_up_red_cube: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_Adaption",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_Adaption/pick_up_red_cube_20_from_droid/4999",
    # ),
    
    
    # EnvMode.ALL_HEAD_pick_up_red_cube: Checkpoint(
    #     config="All_heads_LoRA",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/All_heads_LoRA/table1_pick_up_red_cube_20/4999",
    # ),
    
    # ########################
    # # Hard Tasks Table 2
    # ########################
    # EnvMode.ALL_HEADS_adapted: Checkpoint(
    #     config="All_heads_LoRA_Adaption",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/9.23_table2/All_heads_LoRA_Adaption/place_green_cube_in_red_bowl_20_adapted_from_place_marker_in_mug_200'/4000",
    # ),
    # EnvMode.KNN_adapted: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_Adaption",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/9.23_table2/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_Adaption/place_green_cube_in_red_bowl_20_from_place_marker_in_mug_200/4000",
    # ),
    # ##### Variation Testing
    # EnvMode.ALL_HEADS_place_marker_in_mug: Checkpoint(
    #     config="All_heads_LoRA",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/All_heads_LoRA/debug_lerobot_all_heads_place_marker_in_mug_200/4999",
    # ),
    # EnvMode.KNN_place_marker_in_mug: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/debug_lerobot_KNN_heads_place_marker_in_mug_200/4999",
    # ),

    # EnvMode.exp1: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_joint_training",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/10.10/table2/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_joint_training/joint_training_place_marker_in_mug_20_place_green_cube_in_red_bowl_20/4999",
    # ),
    # EnvMode.exp2: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_joint_training",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/10.10/table2/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_joint_training/joint_training_place_marker_in_mug_20_place_green_cube_in_red_bowl_20_from_merged/4999",
    # ),
    # EnvMode.exp3: Checkpoint(
    #     config="All_heads_LoRA",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/10.10/table2/All_heads_LoRA/joint_training_place_marker_in_mug_20_place_green_cube_in_red_bowl_20/4999",
    # ),


    # ##################
    # # Figure 3
    # #################
    # EnvMode.V10: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_head_variation",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/figure3_scaling/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_head_variation/place_marker_in_mug_200_head_10/4999",
    # ),
    # EnvMode.V40: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_head_variation",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/figure3_scaling/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_head_variation/place_marker_in_mug_200_head_40/4999",
    # ),
    # EnvMode.V80: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_head_variation",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/figure3_scaling/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_head_variation/place_marker_in_mug_200_head_80/4999",
    # ),
    # EnvMode.V1K: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/figure3_scaling/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/debug_lerobot_KNN_heads_place_marker_in_mug_200/1000",
    # ),

    # #######################
    # # tABLE 4
    # #######################
    # EnvMode.KNN_t4: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_variation",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/Experiments/table3_ablations/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_variation/place_marker_in_mug_200_freeze_MLP/4999",
    # ),
    #  EnvMode.CMA: Checkpoint(
    #     config="CMA_heads_pi0_droid_lerobot_finetune_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/9.24_table3_1/CMA_heads_pi0_droid_lerobot_finetune_freeze_KV_SIGLIP_ActionExpert_MLP/debug_lerobot_CMA_heads_place_marker_in_mug_200/4999",
    # ),

    #  EnvMode.REINFORCE: Checkpoint(
    #     config="REINFORCE_heads_pi0_droid_lerobot_finetune_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/9.24_table3_1/REINFORCE_heads_pi0_droid_lerobot_finetune_freeze_KV_SIGLIP_ActionExpert_MLP/debug_lerobot_REINFORCE_heads_place_marker_in_mug_200/4999",
    # ),
    #  #######################
    # # Ablations
    # #######################
    # EnvMode.Sanity_Marker_in_Mug: Checkpoint(
    #     config="Sanity_check_first_20_heads_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/ICLR_Revisions/package/Sanity_check_first_20_heads_freeze_KV_SIGLIP_ActionExpert_MLP/Sanity_check_first_20_heads_place_marker_in_mug_200/4999",
    # ), 
    
    # EnvMode.test: Checkpoint(
    #     config="KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_Adaption",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/ICLR_Revisions/package/KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_Adaption/place_green_cube_in_red_bowl_20_adapted_from_place_marker_in_mug_200_non_overlapping/4999",
    # ),

    # EnvMode.gradient1: Checkpoint(
    #     config="Gradient_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/10.10/table3/Gradient_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/place_marker_in_mug_200_1000_steps/4999",
    # ),
    # EnvMode.gradient2: Checkpoint(
    #     config="Gradient_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/10.10/table3/Gradient_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/place_marker_in_mug_200_normalized_1000_steps/4999",
    # ),
    # EnvMode.gradient3: Checkpoint(
    #     config="Gradient_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/10.10/table3/Gradient_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/place_marker_in_mug_200_1000_steps_exclude_KV/4999",
    # ),
    # EnvMode.gradient4: Checkpoint(
    #     config="Gradient_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP",
    #     dir="/darrell_robotics/raj_home/rtv/openpi_robotv/checkpoints/10.10/table3/Gradient_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP/place_marker_in_mug_200_normalized_1000_steps_exclude_KV/4999",
    # ),


}
def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
