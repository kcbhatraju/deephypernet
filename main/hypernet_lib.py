import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from itertools import chain

class BaseNet(nn.Module):
    def __init__(self, num_backward_connections=0, connection_type="avg", device="cpu"):
        super().__init__()
        self.num_backward_connections = num_backward_connections
        self.connection_type = connection_type
        self.device = device

    def create_params(self):
        raise NotImplementedError("Subclasses implement this method to initialize the weight generator network.")

class HyperNet(BaseNet):
    """General HyperNetwork class"""
    
    def __init__(self, target_network, num_backward_connections=0, connection_type="avg", device="cpu"):
        super().__init__(num_backward_connections, connection_type, device)
        self._target_network = target_network

class SharedEmbeddingUpdateParams(nn.Module):
    """This class updates the parameters of the target network using the output of the previous weight generator."""
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, all_params, residual_params, prev_out: torch.Tensor, raw: torch.Tensor, embed: torch.Tensor, *args, **kwargs):
        # connect `min(num_backward_connections, len(prev_params))` previous params to the current ones
        pooled_addition = self.module.pool_to_param_shape(torch.stack(all_params[self.module.net_depth+1:self.module.net_depth+1 + self.module.num_backward_connections]))
        
        weighted_addition = pooled_addition * residual_params[self.module.net_depth][self.module.net_depth:self.module.net_depth + self.module.num_backward_connections][:, None]
        # (num_backward_connections, current num_weight_gen_params)
        # average to get rid of first dimension
        final_addition = weighted_addition.mean(dim=0)
        
        params = {}
        start = 0
        # tensor of shape (num_weight_gen_params,)
        curr_param_vector = torch.zeros(self.module.num_weight_gen_params, device=self.module.device)
        for name, p in self.module.weight_generator.named_parameters():
            end = start + np.prod(p.size())
            curr_param = prev_out[start:end] + final_addition[start:end]
            curr_param_vector[start:end] = curr_param
            params[name] = curr_param.view(p.size())
            start = end
        
        all_params[self.module.net_depth] = self.module.pool_to_max_params(curr_param_vector.view(1, -1)).view(-1)
        
        if isinstance(self.module, SharedEmbeddingHyperNet):
            out = torch.func.functional_call(self.module.weight_generator, params, (embed,))
            return self.module.propagate(all_params, residual_params, out, raw, embed, *args, **kwargs)
        elif isinstance(self.module, BaseNet):
            out = torch.func.functional_call(self.module.weight_generator, params, (raw, *args), kwargs)
            return out

class SharedEmbeddingHyperNet(HyperNet):
    """
    If the input data is of shape (B, *S), each HyperNet is provided an `embed` from `SharedEmbedding`
    of shape (1, *S). For all n >= 1, H_n outputs a tensor of shape (P_(n-1),) where P_(n-1) is the
    number of parameters in the (n-1)th HyperNet. The output network is fed in the full data,
    from which it makes a prediction.
    """

    def __init__(self, target_network, num_backward_connections=0, connection_type="avg", device="cpu"):
        super().__init__(target_network, num_backward_connections, connection_type, device)
        self.target_param_updater = SharedEmbeddingUpdateParams(target_network)

    def propagate_forward(self, all_params, residual_params, raw, embed, *args, **kwargs):
        param_vector = nn.utils.parameters_to_vector(self.weight_generator.parameters())
        all_params[self.net_depth] = self.pool_to_max_params(param_vector.view(1, -1)).view(-1)
        
        out = self.weight_generator.forward(embed)
        return self.propagate(all_params, residual_params, out, raw, embed, *args, **kwargs)
    
    def propagate(self, all_params, residual_params, out, raw, embed, *args, **kwargs):
        return self.target_param_updater(all_params, residual_params, out.view(-1), raw, embed, *args, **kwargs)

class SharedEmbedding(nn.Module):
    def __init__(self, top_hypernet, num_embeddings, device="cpu"):
        super().__init__()
        self.top_hypernet = top_hypernet
        self.num_embeddings = num_embeddings
        self.device = device
    
    def create_params(self):
        raise NotImplementedError("Subclasses implement this method to initialize the weight generator network.")

    @staticmethod
    def build_weight_generator(module):
        module.create_params()
        module.num_weight_gen_params = sum(p.numel() for p in module.weight_generator.parameters())
    
    def build(self, input_shape, embedding_dim):
        if isinstance(embedding_dim, int):
            embedding_dim = (embedding_dim,)
        
        self.embedding_dim = embedding_dim
        SharedEmbedding.build_weight_generator(self)

        _hypernet_stack = []
        curr_hypernet = self.top_hypernet
        while isinstance(curr_hypernet, SharedEmbeddingHyperNet):
            _hypernet_stack.append(curr_hypernet)

            curr_hypernet.num_embeddings = self.num_embeddings
            curr_hypernet.embedding_dim = self.embedding_dim
            curr_hypernet = curr_hypernet._target_network
        
        curr_hypernet.input_dim = input_shape[1:]
        _hypernet_stack.append(curr_hypernet)

        max_num_weight_gen_params = 0
        for curr_hypernet in reversed(_hypernet_stack):
            if isinstance(curr_hypernet, SharedEmbeddingHyperNet):
                curr_hypernet.num_params_to_estimate = curr_hypernet._target_network.num_weight_gen_params
            
            SharedEmbedding.build_weight_generator(curr_hypernet)

            if curr_hypernet.connection_type == "avg":
                curr_hypernet.pool_to_param_shape = nn.AdaptiveAvgPool1d(curr_hypernet.num_weight_gen_params)
            elif curr_hypernet.connection_type == "max":
                curr_hypernet.pool_to_param_shape = nn.AdaptiveMaxPool1d(curr_hypernet.num_weight_gen_params)
            else:
                raise ValueError(f"Invalid connection type: {curr_hypernet.connection_type}")

            max_num_weight_gen_params = max(max_num_weight_gen_params, curr_hypernet.num_weight_gen_params)
        
        net_depth = 0
        while _hypernet_stack:
            curr_hypernet = _hypernet_stack.pop()
            curr_hypernet.max_num_weight_gen_params = max_num_weight_gen_params

            curr_hypernet.pool_to_max_params = nn.AdaptiveAvgPool1d(max_num_weight_gen_params)

            curr_hypernet.net_depth = net_depth
            net_depth += 1
        
        self.net_depth = net_depth
        self.max_num_weight_gen_params = max_num_weight_gen_params

        self.all_params = [None for _ in range(self.net_depth)]
        self.residual_params = nn.Parameter(torch.triu(torch.randn(net_depth, net_depth-1, device=self.device)))
    
    def parameters(self):
        return chain(self.weight_generator.parameters(), [self.residual_params], self.top_hypernet.weight_generator.parameters())

# --------------------------------------------------------------------------------
#             SHARED EMBEDDING DYNAMIC HYPERNETWORK ARCHITECTURE
# --------------------------------------------------------------------------------
#             x ------> [embed layer] ------> embed (batch size compressed)
#             |                                          /   |   \
#             |                                       [hypernetworks] ----\
#             |                                              |   ^--------/
#             |                                              |
#             |                                              |
#             \-------------> [output network] <-------------/
# --------------------------------------------------------------------------------
class DynamicSharedEmbedding(SharedEmbedding):
    """Creates a single embedding for a batch of data to be used by SharedEmbeddingHyperNet"""
    def __init__(self, top_hypernet: SharedEmbeddingHyperNet, input_shape):
        super().__init__(top_hypernet, 1)
        self.batch_size = input_shape[0]
        self.top_hypernet = top_hypernet
        self.build(input_shape, input_shape[1:])

    def create_params(self):
        raise NotImplementedError("Subclasses implement this method to initialize the weight generator.")
    
    def embed_and_propagate(self, raw):
        assert raw.shape[0] <= self.batch_size
        
        padded = raw
        diff = self.batch_size - padded.shape[0]
        
        if padded.shape[0] < self.batch_size:
            num_no_pads = np.array([[0, 0] for _ in range(padded.ndim - 1)])
            num_no_pads = num_no_pads.flatten()
            padded = F.pad(padded, (*num_no_pads, 0, diff))
        
        embed = self.weight_generator.forward(padded, diff)
        return self.top_hypernet.propagate_forward(self.all_params, self.residual_params, raw, embed)

# --------------------------------------------------------------------------------
#             SHARED EMBEDDING STATIC HYPERNETWORK ARCHITECTURE
# --------------------------------------------------------------------------------
#             x                                        independent embed
#             |                                            /   |   \
#             |                                         [hypernetworks] ----\
#             |                                                |   ^--------/
#             |                                                |
#             |                                                |
#             \--------------> [output network] <--------------/
# --------------------------------------------------------------------------------

# no need to subclass this as `self.weight_generator` is predefined as an `nn.Embedding` layer
class StaticSharedEmbedding(SharedEmbedding):
    def __init__(self, top_hypernet: SharedEmbeddingHyperNet, num_embeddings, embedding_dim, input_shape):
        super().__init__(top_hypernet, num_embeddings)
        self.context_vector = torch.arange(num_embeddings)
        self.top_hypernet = top_hypernet
        self.build(input_shape, embedding_dim)
    
    def create_params(self):
        self.weight_generator = nn.Sequential(nn.Embedding(self.num_embeddings, np.prod(self.embedding_dim)))
    
    def embed_and_propagate(self, raw):
        embed = self.weight_generator(self.context_vector)
        return self.top_hypernet.propagate_forward(self.all_params, self.residual_params, raw, embed)
