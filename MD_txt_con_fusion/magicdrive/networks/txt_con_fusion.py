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


class txt_con_XFormersAttn(torch.nn.Module):
    def __init__(self, con_dim=320, txt_dim=768, hidden_size=320):
        # super().__init__(attention_op=None)  # It is recommended to set to `None` and leave it to xformers
        super().__init__()
        self.attention_op = None

        # below self params borrow from diffusers.models.attention.Attention
        # while keep all the params to self, and make the modules trainable
        self.inner_dim=self.out_dim=hidden_size
        self.to_q = torch.nn.Linear(con_dim, self.inner_dim, bias=False)
        self.to_k = torch.nn.Linear(txt_dim, self.inner_dim, bias=False)
        self.to_v = torch.nn.Linear(txt_dim, self.inner_dim, bias=False)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, self.out_dim, bias=True),
            torch.nn.Dropout(p=0.0, inplace=False)  # TODO: will this be automatically shutted down during eval?
        ])
        self.heads=8
        dim_head=self.out_dim//self.heads
        self.scale=dim_head ** -0.5
        self.rescale_output_factor=1.0

        # enable residual connection
        self.residual_connection=True

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

        # if attn.spatial_norm is not None:
        #     hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        # if attention_mask is not None:
        #     # expand our mask's singleton query_tokens dimension:
        #     #   [batch*heads,            1, key_tokens] ->
        #     #   [batch*heads, query_tokens, key_tokens]
        #     # so that it can be added as a bias onto the attention scores that xformers computes:
        #     #   [batch*heads, query_tokens, key_tokens]
        #     # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
        #     _, query_tokens, _ = hidden_states.shape
        #     attention_mask = attention_mask.expand(-1, query_tokens, -1)

        # if attn.group_norm is not None:
        #     hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)

        # only for con&txt cross-attn
        assert encoder_hidden_states is not None
        # ---------------------------original---------------------------
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # NOTE: xformers induce large error when bs is large. so we do not add
        # head to batch and split batch size if necessary.

        def _split_head(tensor):
            head_size = self.heads
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
                op=self.attention_op, scale=self.scale)
        else:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask,
                op=self.attention_op, scale=self.scale)

        hidden_states = hidden_states.to(query.dtype)
        # hidden_states = attn.batch_to_head_dim(_hidden_states)
        hidden_states = _back_head(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states


class txt_con_XFormersAttn_plus(torch.nn.Module):
    def __init__(self, con_dim=320, txt_dim=768, hidden_size=320):
        # super().__init__(attention_op=None)  # It is recommended to set to `None` and leave it to xformers
        super().__init__()
        self.attention_op = None

        # below self params borrow from diffusers.models.attention.Attention
        # while keep all the params to self, and make the modules trainable
        self.inner_dim=self.out_dim=hidden_size
        self.to_q_occ = torch.nn.Linear(con_dim, self.inner_dim, bias=False)
        self.to_k_occ = torch.nn.Linear(con_dim, self.inner_dim, bias=False)
        self.to_v_occ = torch.nn.Linear(con_dim, self.inner_dim, bias=False)
        self.to_k_txt = torch.nn.Linear(txt_dim, self.inner_dim, bias=False)
        self.to_v_txt = torch.nn.Linear(txt_dim, self.inner_dim, bias=False)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, self.out_dim, bias=True),
            torch.nn.Dropout(p=0.0, inplace=False)  # TODO: will this be automatically shutted down during eval?
        ])
        self.heads=8
        dim_head=self.out_dim//self.heads
        self.scale=dim_head ** -0.5
        self.rescale_output_factor=1.0

        # enable residual connection
        self.residual_connection=True

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

        # if attn.spatial_norm is not None:
        #     hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        # if attention_mask is not None:
        #     # expand our mask's singleton query_tokens dimension:
        #     #   [batch*heads,            1, key_tokens] ->
        #     #   [batch*heads, query_tokens, key_tokens]
        #     # so that it can be added as a bias onto the attention scores that xformers computes:
        #     #   [batch*heads, query_tokens, key_tokens]
        #     # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
        #     _, query_tokens, _ = hidden_states.shape
        #     attention_mask = attention_mask.expand(-1, query_tokens, -1)

        # if attn.group_norm is not None:
        #     hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        occ_q = self.to_q_occ(hidden_states)
        occ_k = self.to_k_occ(hidden_states)
        occ_v = self.to_v_occ(hidden_states)

        # only for con&txt cross-attn
        assert encoder_hidden_states is not None
        # ---------------------------original---------------------------
        txt_k = self.to_k_txt(encoder_hidden_states)
        txt_v = self.to_v_txt(encoder_hidden_states)

        # NOTE: xformers induce large error when bs is large. so we do not add
        # head to batch and split batch size if necessary.

        def _split_head(tensor):
            head_size = self.heads
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
        occ_q = _split_head(occ_q)
        occ_k = _split_head(occ_k)
        occ_v = _split_head(occ_v)
        txt_k = _split_head(txt_k)
        txt_v = _split_head(txt_v)

        assert attention_mask is None, "we don't use attention mask here"
        
        occ_q = xformers.ops.memory_efficient_attention(
            occ_q, txt_k, txt_v, attn_bias=attention_mask,
            op=self.attention_op, scale=self.scale)
        hidden_states = xformers.ops.memory_efficient_attention(
            occ_q, occ_k, occ_v, attn_bias=attention_mask,
            op=self.attention_op, scale=self.scale)

        hidden_states = hidden_states.to(occ_q.dtype)
        # hidden_states = attn.batch_to_head_dim(_hidden_states)
        hidden_states = _back_head(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states
