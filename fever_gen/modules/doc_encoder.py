import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch._C import device
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
)
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from fever_gen.modules import (
    DocumentEncoderLayer,
)

class DocumentEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`DocumentEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        layer = DocumentEncoderLayer(args)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def forward_embedding(
        self, docs_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(docs_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(docs_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        docs_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            docs_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(docs_tokens,
                                       src_lengths,
                                       return_all_hiddens,
                                       token_embeddings)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        docs_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            docs_tokens (LongTensor): tokens in the source language of shape
                `(batch, tgt_len, sen_num, sen_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        # print('**** DocumentEncoder ****')
        # print('docs_tokens', docs_tokens.shape)
        # batch_size, tgt_len, sen_num, sen_len = docs_tokens.shape
        # docs_tokens = docs_tokens.view(batch_size, -1).contiguous()
        # encoder_padding_mask = docs_tokens.eq(self.padding_idx)
        # has_pads = (docs_tokens.device.type == "xla" or encoder_padding_mask.any())

        # x, encoder_embedding = self.forward_embedding(docs_tokens, token_embeddings)

        # # account for padding while computing the representation
        # if encoder_padding_mask is not None:
        #     x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # # B x T x C -> T x B x C
        # x = x.transpose(0, 1)

        # encoder_states = []

        # if return_all_hiddens:
        #     encoder_states.append(x)

        # # encoder layers
        # for layer in self.layers:
        #     x = layer(
        #         x, encoder_padding_mask=encoder_padding_mask if has_pads else None
        #     )
        #     if return_all_hiddens:
        #         assert encoder_states is not None
        #         encoder_states.append(x)

        # if self.layer_norm is not None:
        #     x = self.layer_norm(x)
        # print('***x', x.shape)
        # x = x.view(tgt_len, sen_num, sen_len, batch_size, -1)
        # print('***x', x.shape)

        tgt_len = docs_tokens['tgt_len']
        batch_size = docs_tokens['batch_size']
        # dtype = docs_tokens['doc'][0]['1']['doc'].dtype
        try:
            device = docs_tokens['doc'][0]['1']['docs_tokens'].device
        except:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        doc_padding_mask = torch.ones(batch_size, tgt_len, dtype=torch.bool, device=device)
        batch_doc = torch.zeros(batch_size, tgt_len, 1024, dtype=torch.float16, device=device)
        encoder_embedding = torch.zeros(batch_size, tgt_len, 1024, dtype=torch.float16, device=device)
        encoder_states = []
        for sid, sample in enumerate(docs_tokens['doc']):
            for d in sample.values():
                sentence_tokens = d['docs_tokens']
                # print('***sentence_tokens', sentence_tokens.shape)
                encoder_padding_mask = sentence_tokens.eq(self.padding_idx)
                has_pads = (sentence_tokens.device.type == "xla" or encoder_padding_mask.any())

                x, encoder_embedding = self.forward_embedding(sentence_tokens, token_embeddings)

                # account for padding while computing the representation
                if encoder_padding_mask is not None:
                    x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

                # B x C x H -> C x B x H
                x = x.transpose(0, 1)

                encoder_states = []

                if return_all_hiddens:
                    encoder_states.append(x)

                # encoder layers
                for layer in self.layers:
                    x = layer(
                        x, encoder_padding_mask=encoder_padding_mask if has_pads else None
                    )
                    if return_all_hiddens:
                        assert encoder_states is not None
                        encoder_states.append(x)

                if self.layer_norm is not None:
                    x = self.layer_norm(x)
                # C x B x H
                x = x.mean(dim=0)
                x = x.mean(dim=0)
                # print('***x', x.shape)
                for position in d['position']:
                    batch_doc[sid, position, :] = x
                    doc_padding_mask[sid, position] = False

        if 'sort_order' in docs_tokens:
            sort_order = docs_tokens['sort_order']
            batch_doc = batch_doc.index_select(0, sort_order)
            doc_padding_mask = doc_padding_mask.index_select(0, sort_order)
        # doc_padding_mask = docs_tokens.eq(self.padding_idx)
        # doc_padding_mask = torch.any(doc_padding_mask, dim=-1)
        # doc_padding_mask = torch.any(doc_padding_mask, dim=-1)
        # docs_tokens = docs_tokens.permute(1, 2, 0, 3)
        # batch_doc = []
        # ### [B, T, S, C] -> [T, S, B, C]
        # # print('***tgt len', len(docs_tokens))
        # # [S, B, C]
        # for tgt_tokens in docs_tokens:
        #     docs = []
        #     # print('*****sen num', len(tgt_tokens))
        #     # [B, C]
        #     for sentence_tokens in tgt_tokens:
        #         encoder_padding_mask = sentence_tokens.eq(self.padding_idx)
        #         has_pads = (sentence_tokens.device.type == "xla" or encoder_padding_mask.any())

        #         x, encoder_embedding = self.forward_embedding(sentence_tokens, token_embeddings)

        #         # account for padding while computing the representation
        #         if encoder_padding_mask is not None:
        #             x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        #         # B x C x H -> C x B x H
        #         x = x.transpose(0, 1)

        #         encoder_states = []

        #         if return_all_hiddens:
        #             encoder_states.append(x)

        #         # encoder layers
        #         for layer in self.layers:
        #             x = layer(
        #                 x, encoder_padding_mask=encoder_padding_mask if has_pads else None
        #             )
        #             if return_all_hiddens:
        #                 assert encoder_states is not None
        #                 encoder_states.append(x)

        #         if self.layer_norm is not None:
        #             x = self.layer_norm(x)
        #         x = x.mean(dim=0)
        #         # [B, H]
        #         # print('sentence', x.shape)
        #         # x = x.mean(dim=0)
        #         # print('doc', x.shape)
        #         docs.append(x)
        #     # [S, B, H]
        #     docs = torch.stack(docs, dim=0)
        #     batch_doc.append(docs)
        # # [T, S, B, H]
        # batch_doc = torch.stack(batch_doc, dim=0)
        # print('*****batch_doc', batch_doc.shape)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        batch_doc = batch_doc.transpose(0, 1)
        # print('doc_padding_mask is', doc_padding_mask)
        return {
            "encoder_out": [batch_doc],  # T x B x C
            "encoder_padding_mask": [doc_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "docs_tokens": [],
            "src_lengths": [],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        # if len(encoder_out["encoder_embedding"]) == 0:
        #     new_encoder_embedding = []
        # else:
        #     new_encoder_embedding = [
        #         encoder_out["encoder_embedding"][0].index_select(0, new_order)
        #     ]

        if len(encoder_out["docs_tokens"]) == 0:
            docs_tokens = []
        else:
            docs_tokens = [(encoder_out["docs_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            # "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "docs_tokens": docs_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        DocEncoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward(
                docs_tokens=net_input["docs_tokens"],
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
            )
        else:
            return self.forward_non_torchscript(net_input)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        if 'src_tokens' in net_input:
            del net_input['src_tokens']
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        return self.forward(**encoder_input)
