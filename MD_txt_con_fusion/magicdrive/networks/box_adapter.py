import torch
import os
import math
from typing import List, Optional, Callable
from diffusers.models.attention import Attention
from diffusers.utils.import_utils import is_xformers_available
if is_xformers_available():
    import xformers
    import xformers.ops
    CHECK_XFORMERS = int(os.getenv("CHECK_XFORMERS", "0")) == 1
    SPLIT_SIZE = int(os.getenv("SPLIT_SIZE", -1))
    ERROR_TOLERANCE = 0.002
    from diffusers.models.attention_processor import XFormersAttnProcessor
else:
    xformers = None

class XFormersAttnProcessor(torch.nn.Module):
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        super().__init__()
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ):
        actual_size = hidden_states.shape[0]
        if SPLIT_SIZE != -1 and actual_size > SPLIT_SIZE:
            split_steps = math.ceil(actual_size / SPLIT_SIZE)
            split_steps = min(split_steps, actual_size)
            hidden_states_out = []
            _hidden_states = hidden_states.chunk(split_steps)
            if encoder_hidden_states is None:
                _encoder_hidden_states = [None] * split_steps
            else:
                _encoder_hidden_states = encoder_hidden_states.chunk(
                    split_steps)
            assert attention_mask is None
            assert temb is None
            for i in range(split_steps):
                hidden_states_out.append(
                    self._real_call(
                        attn, _hidden_states[i], _encoder_hidden_states[i],
                        attention_mask, temb)
                )
            return torch.cat(hidden_states_out, dim=0)
        else:
            return self._real_call(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                temb)

    def _real_call(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE: xformers induce large error when bs is large. so we do not add
        # head to batch and split batch size if necessary.

        def _split_head(tensor):
            head_size = attn.heads
            batch_size, seq_len, dim = tensor.shape
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            return tensor

        def _back_head(tensor):
            batch_size, seq_len, head_size, dim = tensor.shape
            tensor = tensor.reshape(batch_size, seq_len, head_size * dim)
            return tensor

        # query = attn.head_to_batch_dim(query).contiguous()
        # key = attn.head_to_batch_dim(key).contiguous()
        # value = attn.head_to_batch_dim(value).contiguous()
        query = _split_head(query)
        key = _split_head(key)
        value = _split_head(value)

        if attention_mask is not None:
            # from cutlassF
            # HINT: To use an `attn_bias` with a sequence length that is not a
            # multiple of 8, you need to ensure memory is aligned by slicing a
            # bigger tensor.
            # Example: use `attn_bias = torch.zeros([1, 1, 5, 8])[:,:,:,:5]`
            # instead of `torch.zeros([1, 1, 5, 5])`
            b, l1, l2 = attention_mask.shape
            if attention_mask.stride(-2) % 8 != 0:
                l1_align = (l1 // 8 + 1) * 8
                l2_align = (l2 // 8 + 1) * 8
                attention_mask_align = torch.zeros(
                    (b, l1_align, l2_align), dtype=attention_mask.dtype,
                    device=attention_mask.device)
                attention_mask_align[:, :l1, :l2] = attention_mask
                attention_mask = attention_mask_align

            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask[:, :l1, :l2],
                op=self.attention_op, scale=attn.scale)
        else:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask,
                op=self.attention_op, scale=attn.scale)

        hidden_states = hidden_states.to(query.dtype)
        # hidden_states = attn.batch_to_head_dim(_hidden_states)
        hidden_states = _back_head(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class Adapter_XFormersAttnProcessor(torch.nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=200):
        # super().__init__(attention_op=None)  # It is recommended to set to `None` and leave it to xformers
        super().__init__()
        self.attention_op = None

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale  # according to ip_adapter scale of 0.5 better balance between txt & ip
        self.num_tokens = num_tokens  # we use 200 box_query in default

        self.to_k_box = torch.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_box = torch.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_k_cls = torch.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_cls = torch.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ):
        actual_size = hidden_states.shape[0]
        if SPLIT_SIZE != -1 and actual_size > SPLIT_SIZE:
            split_steps = math.ceil(actual_size / SPLIT_SIZE)
            split_steps = min(split_steps, actual_size)
            hidden_states_out = []
            _hidden_states = hidden_states.chunk(split_steps)
            if encoder_hidden_states is None:
                _encoder_hidden_states = [None] * split_steps
            else:
                _encoder_hidden_states = encoder_hidden_states.chunk(
                    split_steps)
            assert attention_mask is None
            assert temb is None
            for i in range(split_steps):
                hidden_states_out.append(
                    self._real_call(
                        attn, _hidden_states[i], _encoder_hidden_states[i],
                        attention_mask, temb)
                )
            return torch.cat(hidden_states_out, dim=0)
        else:
            return self._real_call(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                temb)
    def _real_call(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            batch_size, key_tokens, _ = (hidden_states.shape)
        else:
            if isinstance(encoder_hidden_states, list):
                batch_size, key_tokens, _ = (encoder_hidden_states[0].shape)
            else:
                batch_size, key_tokens, _ = (encoder_hidden_states.shape)

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if isinstance(encoder_hidden_states, list):  # use_box_token
                raise NotImplementedError
                box_hidden_states = encoder_hidden_states[-1]
                encoder_hidden_states = encoder_hidden_states[0]
                box_hidden_states = box_hidden_states.reshape(*box_hidden_states.shape[:2],-1).permute(0,2,1)
            else:
                # modify: split txt+box+cls -> encoder_hidden_states & box_hidden_states & cls_hidden_states
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens
                encoder_hidden_states, cls_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    encoder_hidden_states[:, end_pos:, :],
                )
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens
                encoder_hidden_states, box_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    encoder_hidden_states[:, end_pos:, :],
                )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # ---------------------------original---------------------------
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # ---------------------------box_adapter---------------------------
        if not len(box_hidden_states)==0:
            box_key = self.to_k_box(box_hidden_states)
            box_val = self.to_v_box(box_hidden_states)
            cls_key = self.to_k_cls(cls_hidden_states)
            cls_val = self.to_v_cls(cls_hidden_states)

        # NOTE: xformers induce large error when bs is large. so we do not add
        # head to batch and split batch size if necessary.

        def _split_head(tensor):
            head_size = attn.heads
            batch_size, seq_len, dim = tensor.shape
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            return tensor

        def _back_head(tensor):
            batch_size, seq_len, head_size, dim = tensor.shape
            tensor = tensor.reshape(batch_size, seq_len, head_size * dim)
            return tensor

        # query = attn.head_to_batch_dim(query).contiguous()
        # key = attn.head_to_batch_dim(key).contiguous()
        # value = attn.head_to_batch_dim(value).contiguous()
        query = _split_head(query)
        # ---------------------------original---------------------------
        key = _split_head(key)
        value = _split_head(value)

        if attention_mask is not None:
            # from cutlassF
            # HINT: To use an `attn_bias` with a sequence length that is not a
            # multiple of 8, you need to ensure memory is aligned by slicing a
            # bigger tensor.
            # Example: use `attn_bias = torch.zeros([1, 1, 5, 8])[:,:,:,:5]`
            # instead of `torch.zeros([1, 1, 5, 5])`
            b, l1, l2 = attention_mask.shape
            if attention_mask.stride(-2) % 8 != 0:
                l1_align = (l1 // 8 + 1) * 8
                l2_align = (l2 // 8 + 1) * 8
                attention_mask_align = torch.zeros(
                    (b, l1_align, l2_align), dtype=attention_mask.dtype,
                    device=attention_mask.device)
                attention_mask_align[:, :l1, :l2] = attention_mask
                attention_mask = attention_mask_align

            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask[:, :l1, :l2],
                op=self.attention_op, scale=attn.scale)
        else:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask,
                op=self.attention_op, scale=attn.scale)

        hidden_states = hidden_states.to(query.dtype)
        # hidden_states = attn.batch_to_head_dim(_hidden_states)
        hidden_states = _back_head(hidden_states)

        # ---------------------------box_adapter---------------------------
        if not len(box_hidden_states)==0:
            box_key = _split_head(box_key)
            box_val = _split_head(box_val)
            cls_key = _split_head(cls_key)
            cls_val = _split_head(cls_val)

            # multi-modal cross attn
            box_key_attn = xformers.ops.memory_efficient_attention(
                box_key, cls_key, cls_val, attn_bias=attention_mask,
                op=self.attention_op, scale=attn.scale)
            box_key = box_key + box_key_attn
            box_val_attn = xformers.ops.memory_efficient_attention(
                box_val, cls_key, cls_val, attn_bias=attention_mask,
                op=self.attention_op, scale=attn.scale)
            box_val = box_val + box_val_attn

            if attention_mask is not None:
                # from cutlassF
                # HINT: To use an `attn_bias` with a sequence length that is not a
                # multiple of 8, you need to ensure memory is aligned by slicing a
                # bigger tensor.
                # Example: use `attn_bias = torch.zeros([1, 1, 5, 8])[:,:,:,:5]`
                # instead of `torch.zeros([1, 1, 5, 5])`
                b, l1, l2 = attention_mask.shape
                if attention_mask.stride(-2) % 8 != 0:
                    l1_align = (l1 // 8 + 1) * 8
                    l2_align = (l2 // 8 + 1) * 8
                    attention_mask_align = torch.zeros(
                        (b, l1_align, l2_align), dtype=attention_mask.dtype,
                        device=attention_mask.device)
                    attention_mask_align[:, :l1, :l2] = attention_mask
                    attention_mask = attention_mask_align

                hidden_states = xformers.ops.memory_efficient_attention(
                    query, key, value, attn_bias=attention_mask[:, :l1, :l2],
                    op=self.attention_op, scale=attn.scale)
            else:
                box_hidden_states = xformers.ops.memory_efficient_attention(
                    query, box_key, box_val, attn_bias=attention_mask,
                    op=self.attention_op, scale=attn.scale)

            box_hidden_states = box_hidden_states.to(query.dtype)
            # hidden_states = attn.batch_to_head_dim(_hidden_states)
            box_hidden_states = _back_head(box_hidden_states)

            hidden_states = hidden_states + self.scale * box_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def box_adapter(net, use_box_token=False):
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
            if not use_box_token:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_box.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_box.weight": unet_sd[layer_name + ".to_v.weight"],
                    "to_k_cls.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_cls.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = Adapter_XFormersAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
            else:
                attn_procs[name] = Adapter_XFormersAttnProcessor(hidden_size=hidden_size, cross_attention_dim=128) # boxworld feat token dim 128
    net.set_attn_processor(attn_procs)
    return net