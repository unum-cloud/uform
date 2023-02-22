model2config = {
    'english': {
        'text_encoder': {
            'backbone': 'google/bert_uncased_L-4_H-768_A-12',
            'backbone_type': 'bert',
            'unimodal_n_layers': 2,
            'context_dim': 768,
            'dim': 768,
            'output_dim': 256,
            'pooling': 'cls',
            'head_one_neuron': False
        },
        'img_encoder': {
            'backbone': 'deit3_base_patch16_224_in21ft1k',
            'dim': 768,
            'output_dim': 256,
            'backbone_type': 'vit',
            'pooling': 'cls'
        },
        'checkpoint_name': '5frc00zw.pt'
    },
    'multilingual': {
        'text_encoder': {
            'backbone': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'backbone_type': 'bert',
            'unimodal_n_layers': 8,
            'context_dim': 768,
            'dim': 384,
            'output_dim': 256,
            'pooling': 'mean',
            'head_one_neuron': True
        },
        'img_encoder': {
            'backbone': 'deit3_base_patch16_224_in21ft1k',
            'dim': 768,
            'output_dim': 256,
            'backbone_type': 'vit',
            'pooling': 'cls'
        },
        'checkpoint_name': 'wp37uexl.pt'
    }
}