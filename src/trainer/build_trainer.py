import torch
from models.policies import DeterministicPolicy, GaussianPolicy
from models.value_networks import QNet, TwinQ, ValueFunction
from models.encoder import ImageEncoder
from models.RND import RND, RND_SA
from models.affordance import SimpleCcVae, RNNSimpleCcVae, AttSimpleCcVae
from trainer.iql_trainer import IQLTrainer


def build_iql_trainer(config, iql_pretrain_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_dim = config['Data']['obs_dim']
    act_dim = config['Data']['act_dim']
    print("obs_dim:", obs_dim, " act_dim:", act_dim)
    print("build impilict q learning trainer.")
    if config['IQL']['deterministic_policy']:
        policy = DeterministicPolicy(obs_dim, act_dim, config)
    else:  # default
        policy = GaussianPolicy(obs_dim, act_dim, config)

    if config['IQL']['image_encode']:
        encoder_dim = config['IQL']['encoder_dim']
        image_encoder = ImageEncoder(encoder_dim, config)
    else:
        image_encoder = None
    
    if config['RND']['use_rnd_penalty']:
        print('use RND')
        hidden_dim = config['RND']['hidden_dim']
        encoder_dim = config['IQL']['encoder_dim']
        pretrain_path = config['RND']['pretrain_path']
        print('load pretrained rnd:', pretrain_path)
        if config['RND']['use_film']:
            rnd = RND_SA(encoder_dim, hidden_dim, pretrain_path).eval()
        else:
            rnd = RND(hidden_dim=hidden_dim, 
                      enc_dim=encoder_dim, 
                      pretrain_path=pretrain_path).eval()
    else:
        rnd = None
    
    
    if config['Affordance']['use_afford']:
        params = dict(
            input_dim = config['IQL']['encoder_dim'],
            z_dim = config['Affordance']['z_dim'],
            hidden_dim = config['Affordance']['hidden_dim'],
            use_film = config['use_film'],
            z_var = config['z_var'],
            diff_goal = config['diff_goal'],
            n_hidden = config['Affordance']['n_hidden'],
            bias=config['Affordance']['bias'],
        )
        if config['Affordance']['att_afford']:
            assert config['Affordance']['rnn_afford']
            print("use attention affordance.")
            affordance = AttSimpleCcVae(**params)
        elif config['Affordance']['rnn_afford']:
            print("use rnn affordance.")
            affordance = RNNSimpleCcVae(**params)
        else:
            print("use affordance.")
            affordance = SimpleCcVae(**params)
        if config['Affordance']['use_pretrain']:
            pretrain_path = config['Affordance']['pretrain_path']
            print('load pretrained affordance:', pretrain_path)
            state_dict = torch.load(pretrain_path)
            afford_state_dict = state_dict['affordance']
            affordance.load_state_dict(afford_state_dict)
            if config['IQL']['encode_type'] == 'vqvae':
                encoder_state_dict = state_dict['obs_encoder']
                image_encoder.enc.load_state_dict(encoder_state_dict)
    else:
        affordance = None
    
    
    iql = IQLTrainer(
        image_encoder=image_encoder,
        qf1=QNet(obs_dim, act_dim, config),
        qf2=QNet(obs_dim, act_dim, config),
        vf=ValueFunction(obs_dim, config),
        policy=policy,
        cfg=config,
        affordance=affordance,
        rnd=rnd,
        device=device
    )
    if iql_pretrain_path is not None:
        print('loading pretrain_model:', iql_pretrain_path)
        pretrain_model_dict = torch.load(iql_pretrain_path)['model_state']
        iql.load_state_dict(pretrain_model_dict, strict=False)
    return iql