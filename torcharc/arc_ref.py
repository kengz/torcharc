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
    'perceiver_im_classifer': {
        'type': 'Perceiver',
        'in_shape': [224, 224, 3],
        'arc': {
            'preprocessor': {
                'type': 'FourierPreprocessor',
                'num_freq_bands': 64,
                'max_reso': [224, 224],
                'cat_pos': True,
            },
            'encoder': {
                'type': 'PerceiverEncoder',
                'latent_shape': [512, 1024],
                'head_dim': 1024,  # usually preserves latent_shape[-1]
                'v_head_dim': None,  # defaults to head_dim
                'cross_attn_num_heads': 1,
                'cross_attn_widening_factor': 1,
                'num_self_attn_blocks': 8,
                'num_self_attn_per_block': 6,
                'self_attn_num_heads': 8,
                'self_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'decoder': {
                'type': 'PerceiverDecoder',
                'out_shape': [1, 1024],
                'head_dim': 1024,  # usually preserves out_shape[-1]
                'v_head_dim': None,  # defaults to head_dim
                'cross_attn_num_heads': 1,
                'cross_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'postprocessor': {
                'type': 'ClassificationPostprocessor',
                'out_dim': 10,
            }
        }
    },
    'perceiver_im2text': {
        'type': 'Perceiver',
        'in_shape': [224, 224, 3],
        'arc': {
            'preprocessor': {
                'type': 'FourierPreprocessor',
                'num_freq_bands': 64,
                'max_reso': [224, 224],
                'cat_pos': True,
            },
            'encoder': {
                'type': 'PerceiverEncoder',
                'latent_shape': [512, 1024],
                'head_dim': 1024,  # usually preserves latent_shape[-1]
                'v_head_dim': None,  # defaults to head_dim
                'cross_attn_num_heads': 1,
                'cross_attn_widening_factor': 1,
                'num_self_attn_blocks': 8,
                'num_self_attn_per_block': 6,
                'self_attn_num_heads': 8,
                'self_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'decoder': {
                'type': 'PerceiverDecoder',
                'out_shape': [512, 256],  # [max_seq_len, E]
                'head_dim': 256,  # usually preserves out_shape[-1]
                'v_head_dim': None,  # defaults to head_dim
                'cross_attn_num_heads': 1,
                'cross_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'postprocessor': {
                'type': 'ClassificationPostprocessor',
                'out_dim': 1024,  # vocab_size, out_shape will be [decoder.out_shape[0]=max_seq_len, vocab_size]
            }
        }
    },
    'perceiver_text2text': {
        'type': 'Perceiver',
        'in_shape': [512],  # max_seq_len
        'arc': {
            'preprocessor': {
                'type': 'TextPreprocessor',
                'vocab_size': 1024,
                'embed_dim': 256,
                'max_seq_len': 512,
            },
            'encoder': {
                'type': 'PerceiverEncoder',
                'latent_shape': [256, 160],
                'head_dim': 32,
                'v_head_dim': 160,
                'cross_attn_num_heads': 8,
                'cross_attn_widening_factor': 1,
                'num_self_attn_blocks': 1,
                'num_self_attn_per_block': 26,
                'self_attn_num_heads': 8,
                'self_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'decoder': {
                'type': 'PerceiverDecoder',
                'out_shape': [512, 160],  # [max_seq_len, E]
                'head_dim': 32,
                'v_head_dim': 96,
                'cross_attn_num_heads': 8,
                'cross_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'postprocessor': {
                'type': 'ClassificationPostprocessor',
                'out_dim': 1024,  # vocab_size, out_shape will be [decoder.out_shape[0]=max_seq_len, vocab_size]
            }
        }
    },
    'perceiver_ts2ts': {  # time series to time series
        'type': 'Perceiver',
        'in_shape': [512],  # max_seq_len
        'arc': {
            'preprocessor': {
                'type': 'Identity',  # nothing to preprocess
            },
            'encoder': {
                'type': 'PerceiverEncoder',
                'latent_shape': [256, 160],
                'head_dim': 32,
                'v_head_dim': 160,
                'cross_attn_num_heads': 8,
                'cross_attn_widening_factor': 1,
                'num_self_attn_blocks': 1,
                'num_self_attn_per_block': 26,
                'self_attn_num_heads': 8,
                'self_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'decoder': {
                'type': 'PerceiverDecoder',
                'out_shape': [512, 160],  # [max_seq_len, E]
                'head_dim': 32,
                'v_head_dim': 96,
                'cross_attn_num_heads': 8,
                'cross_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'postprocessor': {
                'type': 'ProjectionPostprocessor',
                'out_dim': 1,
            }
        }
    },
    'perceiver_ts2classifier_ts': {  # time series to time series of classes
        'type': 'Perceiver',
        'in_shape': [512],  # max_seq_len
        'arc': {
            'preprocessor': {
                'type': 'Identity',  # nothing to preprocess
            },
            'encoder': {
                'type': 'PerceiverEncoder',
                'latent_shape': [256, 160],
                'head_dim': 32,
                'v_head_dim': 160,
                'cross_attn_num_heads': 8,
                'cross_attn_widening_factor': 1,
                'num_self_attn_blocks': 1,
                'num_self_attn_per_block': 26,
                'self_attn_num_heads': 8,
                'self_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'decoder': {
                'type': 'PerceiverDecoder',
                'out_shape': [512, 160],  # [max_seq_len, E]
                'head_dim': 32,
                'v_head_dim': 96,
                'cross_attn_num_heads': 8,
                'cross_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'postprocessor': {
                'type': 'ClassificationPostprocessor',
                'out_dim': 2,  # vocab_size, out_shape will be [decoder.out_shape[0]=max_seq_len, vocab_size]
            }
        }
    },
    'perceiver_multimodal2classifier': {
        'type': 'Perceiver',
        'in_shapes': {'image': [224, 224, 3], 'vector': [31, 2]},
        'arc': {
            'preprocessor': {
                'type': 'MultimodalPreprocessor',
                'arc': {
                    'image': {
                        'type': 'FourierPreprocessor',
                        'num_freq_bands': 64,
                        'max_reso': [224, 224],
                        'cat_pos': True,
                    },
                    'vector': {
                        'type': 'FourierPreprocessor',
                        'num_freq_bands': 16,
                        'max_reso': [31],
                        'cat_pos': True,
                    },
                },
                'pad_channels': 2,
            },
            'encoder': {
                'type': 'PerceiverEncoder',
                'latent_shape': [512, 1024],
                'head_dim': 1024,  # usually preserves latent_shape[-1]
                'v_head_dim': None,  # defaults to head_dim
                'cross_attn_num_heads': 1,
                'cross_attn_widening_factor': 1,
                'num_self_attn_blocks': 8,
                'num_self_attn_per_block': 6,
                'self_attn_num_heads': 8,
                'self_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'decoder': {
                'type': 'PerceiverDecoder',
                'out_shape': [1, 1024],
                'head_dim': 1024,  # usually preserves out_shape[-1]
                'v_head_dim': None,  # defaults to head_dim
                'cross_attn_num_heads': 1,
                'cross_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'postprocessor': {
                'type': 'ClassificationPostprocessor',
                'out_dim': 10,
            }
        }
    },
    'perceiver_ts2multimodal': {
        'type': 'Perceiver',
        'in_shape': [512],  # max_seq_len
        'arc': {
            'preprocessor': {
                'type': 'Identity',  # nothing to preprocess
            },
            'encoder': {
                'type': 'PerceiverEncoder',
                'latent_shape': [256, 160],
                'head_dim': 32,
                'v_head_dim': 160,
                'cross_attn_num_heads': 8,
                'cross_attn_widening_factor': 1,
                'num_self_attn_blocks': 1,
                'num_self_attn_per_block': 26,
                'self_attn_num_heads': 8,
                'self_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'decoder': {
                'type': 'PerceiverDecoder',
                'out_shape': [641, 160],  # [max_seq_len, E]
                'head_dim': 32,
                'v_head_dim': 96,
                'cross_attn_num_heads': 8,
                'cross_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'postprocessor': {
                'type': 'MultimodalPostprocessor',
                'in_shapes': {'classifier': [1, 160], 'ts_1': [512, 160], 'ts_2': [128, 160]},
                'arc': {
                    'classifier': {
                        'type': 'ClassificationPostprocessor',
                        'out_dim': 10,
                    },
                    'ts_1': {
                        'type': 'ProjectionPostprocessor',
                        'out_dim': 4,
                    },
                    'ts_2': {
                        'type': 'ProjectionPostprocessor',
                        'out_dim': 4,
                    },
                },
            }
        }
    },
    'perceiver_multimodal2multimodal': {
        'type': 'Perceiver',
        'in_shapes': {'image': [224, 224, 3], 'vector': [31, 2]},
        'arc': {
            'preprocessor': {
                'type': 'MultimodalPreprocessor',
                'arc': {
                    'image': {
                        'type': 'FourierPreprocessor',
                        'num_freq_bands': 64,
                        'max_reso': [224, 224],
                        'cat_pos': True,
                    },
                    'vector': {
                        'type': 'FourierPreprocessor',
                        'num_freq_bands': 16,
                        'max_reso': [31],
                        'cat_pos': True,
                    },
                },
                'pad_channels': 2,
            },
            'encoder': {
                'type': 'PerceiverEncoder',
                'latent_shape': [512, 1024],
                'head_dim': 1024,  # usually preserves latent_shape[-1]
                'v_head_dim': None,  # defaults to head_dim
                'cross_attn_num_heads': 1,
                'cross_attn_widening_factor': 1,
                'num_self_attn_blocks': 8,
                'num_self_attn_per_block': 6,
                'self_attn_num_heads': 8,
                'self_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'decoder': {
                'type': 'PerceiverDecoder',
                'out_shape': [641, 160],  # [max_seq_len, E]
                'head_dim': 32,
                'v_head_dim': 96,
                'cross_attn_num_heads': 8,
                'cross_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'postprocessor': {
                'type': 'MultimodalPostprocessor',
                'in_shapes': {'classifier': [1, 160], 'ts_1': [512, 160], 'ts_2': [128, 160]},
                'arc': {
                    'classifier': {
                        'type': 'ClassificationPostprocessor',
                        'out_dim': 10,
                    },
                    'ts_1': {
                        'type': 'ProjectionPostprocessor',
                        'out_dim': 4,
                    },
                    'ts_2': {
                        'type': 'ProjectionPostprocessor',
                        'out_dim': 4,
                    },
                },
            }
        }
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
