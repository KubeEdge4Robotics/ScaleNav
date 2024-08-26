from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, default="logs/test")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--env-name', type=str, default="", choices=[""])
    parser.add_argument('--config-dir', type=str, default="custom.yaml")
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--distributed-training', type=int, default=1)
    parser.add_argument('--master-port', type=str, default="12350")
    
    parser.add_argument('--rewards_type', type=str, default="survival", choices=[""])
    parser.add_argument('--image_goal', action="store_true", default=True)
    parser.add_argument('--curriculum', action="store_true")
    parser.add_argument('--warm_up', action="store_true")
    parser.add_argument('--no_aug', action="store_true")
    parser.add_argument('--diff_goal', action="store_true")
    

    parser.add_argument('--random_mask', action="store_true")
    parser.add_argument('--seg', action="store_true")
    parser.add_argument('--use_film', action="store_true")
    parser.add_argument('--contrastive', action="store_true")
    parser.add_argument('--rate', type=int, default=30)
    parser.add_argument('--z_var', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_ln', action="store_true")
    parser.add_argument('--revind', action="store_true")
    parser.add_argument("--other_params",
                        nargs='*',
                        default=[],
                        type=str)
    
    ## deployment
    parser.add_argument('--no_ros', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument("--revise_mode", 
                        default='affordance',
                        type=str
                        )
    args = parser.parse_args()
    return args