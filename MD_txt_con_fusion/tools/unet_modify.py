import torch
from einops import rearrange
from diffusers.models.attention import Attention as CrossAttention
from diffusers.models.attention_processor import XFormersAttnProcessor
# from magicdrive.networks.occ_adapter import Adapter_AttnProcessor as AAP

class MyCrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape#;print('===============\n','batch_size, sequence_length',batch_size, sequence_length)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)#;print('query',query.shape)
        # from <diffusion self-guidance>

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        # #<modify>
        # print('hidden_state.shape,encoder_hidden_states',hidden_states.shape,encoder_hidden_states.shape)#hidden_state.shape,encoder_hidden_states torch.Size([48, 1400, 320]) torch.Size([12, 106, 768])
        # if hidden_states.shape[0] // encoder_hidden_states.shape[0] == 4:
        #     encoder_hidden_states = torch.stack([encoder_hidden_states for i in range(4)])
        #     encoder_hidden_states = rearrange(encoder_hidden_states,'p b ... -> (b p) ...')
        # #</modify>

        key = attn.to_k(encoder_hidden_states)#;print('key',key.shape)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)#;print('query_',query.shape)
        key = attn.head_to_batch_dim(key)#;print('key_',key.shape)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)#;print('attn map',attention_probs.shape)
        # # from <diffusion self-guidance>
        # # there are 6 views stacked together, avg over attn channels inside each view
        # # retain_dim = 6
        # # attention_probs_ = attention_probs.reshape(retain_dim, -1, *attention_probs.shape[1:]).mean(1);print('attn map',attention_probs_.shape)
        # # attn.attn_probs = attention_probs_
        
        # # h=w=math.isqrt(scores_.shape[1])  # we don't have square feat map here
        # # scores_ = scores_.reshape(len(scores_), h, w, -1)
        # # new bookkeeping to save the attn probs
        
        '''
        /opt/data/private/aigc/MagicDrive/third_party/diffusers/src/diffusers/pipelines/controlnet/pipeline_controlnet.py L435:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        so we should retrieve the latter part of the cross attention map
        '''
        attn.attn_probs_original = attention_probs.chunk(2)[1]  # under cfg there are 2 identical chunks in the tensor, grab one

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# class Adapter_AttnProcessor(AAP): # modify to save attn.attn_probs_original and attn.attn_probs
#     def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
#         batch_size, sequence_length, _ = hidden_states.shape
#         attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

#         query = attn.to_q(hidden_states)#;print('query',query.shape)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         else:
#             # modify: split encoder_hidden_states into cam_prompt_emb & occ_emb
#             end_pos = encoder_hidden_states.shape[1] - self.num_tokens
#             encoder_hidden_states, occ_hidden_states = (
#                 encoder_hidden_states[:, :end_pos, :],
#                 encoder_hidden_states[:, end_pos:, :],
#             )

#         # ---------------------------original---------------------------
#         key = attn.to_k(encoder_hidden_states)
#         value = attn.to_v(encoder_hidden_states)

#         query = attn.head_to_batch_dim(query)
#         key = attn.head_to_batch_dim(key)
#         value = attn.head_to_batch_dim(value)

#         attention_probs = attn.get_attention_scores(query, key, attention_mask)
#         attn.attn_probs_original = attention_probs.chunk(2)[1]
#         hidden_states = torch.bmm(attention_probs, value)
#         hidden_states = attn.batch_to_head_dim(hidden_states)

#         # ---------------------------occ_adapter---------------------------
#         if not len(occ_hidden_states)==0:
#             occ_key = self.to_k_occ(occ_hidden_states)
#             occ_value = self.to_v_occ(occ_hidden_states)

#             # query = attn.head_to_batch_dim(query)
#             occ_key = attn.head_to_batch_dim(occ_key)
#             occ_value = attn.head_to_batch_dim(occ_value)

#             attention_probs = attn.get_attention_scores(query, occ_key, attention_mask)
#             attn.attn_probs = attention_probs.chunk(2)[1] if attention_probs.shape[-2]==28*50 else None   # under cfg there are 2 identical chunks in the tensor, grab one
#             occ_hidden_states = torch.bmm(attention_probs, occ_value)
#             occ_hidden_states = attn.batch_to_head_dim(occ_hidden_states)
#             hidden_states = hidden_states + self.scale * occ_hidden_states
#         else:
#             attn.attn_probs = None # no box, no guidance this round

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         return hidden_states

"""
A function that prepares a U-Net model for training by enabling gradient computation 
for a specified set of parameters and setting the forward pass to be performed by a 
custom cross attention processor.

Parameters:
unet: A U-Net model.

Returns:
unet: The prepared U-Net model.
"""
def prep_unet(unet):
    # # set the gradients for XA maps to be true
    # for name, params in unet.named_parameters():
    #     if 'attn2' in name:
    #         params.requires_grad = True
    #     else:
    #         params.requires_grad = False
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention":
            module.set_processor(MyCrossAttnProcessor())  # up.blocks.....attn2 
            # mid/down.blocks........attn1/2
            # print(f'modify {name}')
    return unet

def prep_controlnet(net):
    attn_procs = {}
    unet_sd = net.state_dict()
    for name in net.attn_processors.keys():
        # attn4 which are added by magicdrive to unet should also be excluded
        cross_attention_dim = None if (name.endswith("attn1.processor") or name.endswith("attn4.processor")) else net.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = net.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(net.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = net.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = XFormersAttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_occ.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_occ.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = Adapter_AttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    net.set_attn_processor(attn_procs)
    return net