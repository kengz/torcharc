REF_ARCS = {
    # basic modules
    'Conv1d': {
        'type': 'Conv1d',
        'in_shape': [3, 20],
        'layers': [
            [16, 4, 2, 0, 1],
            [16, 4, 1, 0, 1]
        ],
        'batch_norm': True,
        'activation': 'ReLU',
        'dropout': 0.2,
        'init': 'kaiming_uniform_',
    },
    'Conv2d': {
        'type': 'Conv2d',
        'in_shape': [3, 20, 20],
        'layers': [
            [16, 4, 2, 0, 1],
            [16, 4, 1, 0, 1]
        ],
        'batch_norm': True,
        'activation': 'ReLU',
        'dropout': 0.2,
        'init': 'kaiming_uniform_',
    },
    'Conv3d': {
        'type': 'Conv3d',
        'in_shape': [3, 20, 20, 20],
        'layers': [
            [16, 4, 2, 0, 1],
            [16, 4, 1, 0, 1]
        ],
        'batch_norm': True,
        'activation': 'ReLU',
        'dropout': 0.2,
        'init': 'kaiming_uniform_',
    },
    'Linear': {
        'type': 'Linear',
        'in_features': 8,
        'layers': [64, 32],
        'batch_norm': True,
        'activation': 'ReLU',
        'dropout': 0.2,
        'init': {
            'type': 'normal_',
            'std': 0.01,
        },
    },
    'tstransformer': {
        'type': 'TSTransformer',
        'd_model': 64,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dropout': 0.2,
        'dim_feedforward': 2048,
        'activation': 'relu',
        'in_embedding': 'Linear',
        'pe': 'sinusoid',
        'attention_size': None,
        'in_channels': 1,
        'out_channels': 1,
        'q': 8,
        'v': 8,
        'chunk_mode': None,
    },
    'pytorch_tstransformer': {
        'type': 'PTTSTransformer',
        'd_model': 64,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dropout': 0.2,
        'dim_feedforward': 2048,
        'activation': 'relu',
        'in_embedding': 'Linear',
        'pe': 'sinusoid',
        'attention_size': None,
        'in_channels': 1,
        'out_channels': 1,
        'q': 8,
        'v': 8,
        'chunk_mode': None,
    },
    # DAGs
    'forward': {
        'dag_in_shape': [8],
        'type': 'Linear',
        'layers': [64, 32],
        'batch_norm': True,
        'activation': 'ReLU',
        'dropout': 0.2,
        'init': 'kaiming_uniform_',
    },
    'ConcatMerge': {
        'dag_in_shape': {'vector_0': [4], 'vector_1': [6], 'vector_2': [8]},
        'type': 'ConcatMerge',
        'in_names': ['vector_0', 'vector_1', 'vector_2'],
    },
    'FiLMMerge': {
        'dag_in_shape': {'image': [3, 20, 20], 'vector': [8]},
        'type': 'FiLMMerge',
        'in_names': ['image', 'vector'],
        'names': {'feature': 'image', 'conditioner': 'vector'},
    },
    'ReuseFork': {
        'dag_in_shape': [8],
        'type': 'ReuseFork',
        'names': ['reuse_0', 'reuse_1'],
    },
    'SplitFork': {
        'dag_in_shape': [8],
        'type': 'SplitFork',
        'shapes': {'mean': [4], 'std': [4]},
    },
    'merge_fork': {
        'dag_in_shape': {'vector_0': [8], 'vector_1': [16]},
        'merge': {
            'type': 'FiLMMerge',
            'in_names': ['vector_0', 'vector_1'],
            'names': {'feature': 'vector_0', 'conditioner': 'vector_1'},
        },
        'split': {
            'type': 'SplitFork',
            'shapes': {'mean': [4], 'std': [4]},
        }
    },
    'fork_merge': {
        'dag_in_shape': [8],
        'fork': {
            'type': 'SplitFork',
            'shapes': {'mean': [4], 'std': [4]},
        },
        'merge': {
            'type': 'FiLMMerge',
            'in_names': ['mean', 'std'],
            'names': {'feature': 'mean', 'conditioner': 'std'},
        }
    },
    'reuse_fork_forward': {
        'dag_in_shape': [8],
        'fork': {
            'type': 'ReuseFork',
            'names': ['left', 'right'],
        },
        'forward_left': {
            'type': 'Linear',
            'in_names': ['left'],
            'out_features': 8,
        },
        'forward_right': {
            'type': 'Linear',
            'in_names': ['right'],
            'out_features': 6,
        },
    },
    'split_fork_forward': {
        'dag_in_shape': [8],
        'fork': {
            'type': 'SplitFork',
            'shapes': {'left': [4], 'right': [4]},
        },
        'forward_left': {
            'type': 'Linear',
            'in_names': ['left'],
            'out_features': 8,
        },
        'forward_right': {
            'type': 'Linear',
            'in_names': ['right'],
            'out_features': 6,
        },
    },
    'merge_forward_split': {
        'dag_in_shape': {'image': [3, 20, 20], 'vector': [8]},
        'merge': {
            'type': 'FiLMMerge',
            'in_names': ['image', 'vector'],
            'names': {'feature': 'image', 'conditioner': 'vector'},
        },
        'Flatten': {
            'type': 'Flatten',
        },
        'Linear': {
            'type': 'Linear',
            'out_features': 8,
        },
        'fork': {
            'type': 'SplitFork',
            'shapes': {'mean': [4], 'std': [4]},
        }
    },
    'hydra': {
        'dag_in_shape': {'image': [3, 20, 20], 'vector': [8]},
        'image': {
            'type': 'Conv2d',
            'in_names': ['image'],
            'layers': [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            'batch_norm': True,
            'activation': 'ReLU',
            'dropout': 0.2,
            'init': 'kaiming_uniform_',
        },
        'merge': {
            'type': 'FiLMMerge',
            'in_names': ['image', 'vector'],
            'names': {'feature': 'image', 'conditioner': 'vector'},
        },
        'Flatten': {
            'type': 'Flatten'
        },
        'Linear': {
            'type': 'Linear',
            'layers': [64, 32],
            'batch_norm': True,
            'activation': 'ReLU',
            'dropout': 0.2,
            'init': 'kaiming_uniform_',
        },
        'out': {
            'type': 'Linear',
            'out_features': 8,
        },
        'fork': {
            'type': 'SplitFork',
            'shapes': {'mean': [4], 'std': [4]},
        }
    },
}


def validate(arc: dict) -> None:
    '''Validate the arc format according to reference'''
    # TODO implement arc validation
    raise NotImplementedError
