# Helper functions for initializing the Transformer with ImageNet weights as well as the patchification kernels in the feature adaptation module.

import numpy as np
import tensorflow as tf

def load_transformer_weights(model):

    weights = np.load('ViT-B_16_imagenet21k+imagenet2012.npz.1') #weight file for imagenet pretraining
      
    unichannel_embedding_weights = np.expand_dims(np.mean(weights['embedding/kernel'],axis=2),axis=2)
    model.layers[14].set_weights([unichannel_embedding_weights, weights['embedding/bias']])
    model.layers[16].set_weights(np.expand_dims(weights['cls'],axis=0))
    init_positional_embeddings = np.squeeze(weights['Transformer/posembed_input/pos_embedding'])
    final_positional_embeddings = np.zeros((384,768)) #(48x8)xTimesteps
    for k in range(0,768):
        unflattened_positional_embeddings = np.reshape(init_positional_embeddings[1:,k],(24,24))
        temp_positional_embeddings = tf.image.resize(np.expand_dims(unflattened_positional_embeddings,-1),[48,8])
        final_positional_embeddings[:,k] = np.reshape(temp_positional_embeddings,(384,))
    z = np.concatenate((init_positional_embeddings[:1,:],final_positional_embeddings))
    model.layers[17].set_weights([np.expand_dims(z,axis=0)])
    
    for i in range(0,12):
        model.layers[9*i+18].set_weights([weights['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/scale'],weights['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/bias']])
        model.layers[9*i+19].set_weights([weights['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/kernel'],weights['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/bias'],weights['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/kernel'],weights['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/bias'],weights['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/kernel'],weights['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/bias'],weights['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/kernel'],weights['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/bias']])
        model.layers[9*i+21].set_weights([weights['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/scale'],weights['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/bias']])
        model.layers[9*i+22].set_weights([weights['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/kernel'],weights['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/bias']]) 
        model.layers[9*i+24].set_weights([weights['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/kernel'],weights['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/bias']])   

    model.layers[126].set_weights([weights['Transformer/encoder_norm/scale'],weights['Transformer/encoder_norm/bias']])
    return model

def load_adapter_weights(model):

    weights = np.load('ViT-B_16_imagenet21k+imagenet2012.npz.1')
    unichannel_embedding_weights = np.expand_dims(np.mean(weights['embedding/kernel'],axis=2),axis=2)
    for i in range(0,4):
        interp_ = tf.image.resize(np.squeeze(unichannel_embedding_weights),(16//2**(i+1), 16//2**(i)))
        if i == 3:
            interp_ = np.tile(np.expand_dims(interp_,-2),(1,1,192,1))
        else:
            interp_ = np.tile(np.expand_dims(interp_,-2),(1,1,2**(i+5),1))
        model.layers[len(model.layers)-123+24*i].set_weights([interp_,weights['embedding/bias']]) #-1
    
    return model

