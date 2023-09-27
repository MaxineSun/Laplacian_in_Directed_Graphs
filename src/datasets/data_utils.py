import torch
from torch_sparse import mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric
from torch_sparse import SparseTensor
import torch


def vanilla_Lp(adj):
    """
    Get the out degree matrix of adjacency matrix:
        \mathbf{D} - \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1).to_dense()
    vanilla_Lp = row_sum - adj.to_dense()
    vanilla_Lp = vanilla_Lp.to_sparse()
    return vanilla_Lp


def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj


def directed_degree_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{I} - \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    adj = adj.to_dense()
    idm = torch.eye(adj.to_dense().shape[0]).to(adj.device) * (0.1)
    return adj - idm


def vanilla_Lp_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{I} - 2 * (\mathbf{D} - \mathbf{A}) / \max(\mathbf{D}).
    """
    in_deg = sparsesum(adj, dim=0).to_dense()
    D_max = torch.max(in_deg)
    idm = torch.eye(adj.to_dense().shape[0]).to(adj.device)
    Lp = idm - 2 * (in_deg - adj.to_dense()) / D_max
    return Lp


def MagNetLp(adj, q):
    """
    Applies the MagNet normalization for directed graphs:
        \mathbf{D}_{s} - \mathbf{A}_{s} \odot \exp{i \Theta^{q}}.
    """
    adj = adj.to_dense()
    adj_t = torch.transpose(adj, 0, 1)
    adj_s = adj + adj_t
    adj_s = torch.where(adj_s != 0, torch.tensor(1), adj_s)
    d_s = torch.diag(adj_s.sum(0))
    pi = 3.1415926
    pi = pi.to(adj.device)
    theta_i = 2 * pi * 1j * q * (adj - adj_t)
    Lp_M = d_s - adj_s * torch.exp(theta_i)
    Lp_M = Lp_M.to(torch.complex64)
    return Lp_M


def MagNet_norm(adj, q):
    """
    Applies the MagNet normalization for directed graphs:
        \mathbf{I} - \mathbf{D}_{s}^{-1/2} \mathbf{A}_{s} \mathbf{D}_{s}^{-1/2} \odot \exp{i \Theta^{q}}.
    """
    adj = adj.to_dense()
    adj_t = torch.transpose(adj, 0, 1)
    adj_s = adj + adj_t
    adj_s = torch.where(adj_s != 0, torch.tensor(1), adj_s)
    d_s_sqrt = torch.diag(adj_s.sum(0).pow_(-0.5))
    adj_s = torch.matmul(adj_s, d_s_sqrt)
    adj_s = torch.matmul(d_s_sqrt, adj_s)
    
    pi = torch.tensor([3.1415926])
    pi = pi.to(adj.device)
    theta_i = 2 * pi * 1j * q * (adj - adj_t)
    
    idm = torch.eye(adj.to_dense().shape[0])*(0.0)
    idm = idm.to(adj.device)
    conv = adj_s * torch.exp(theta_i)
    Lp_M = conv - idm
    Lp_M = Lp_M.to(torch.complex64)
    Lp_M = Lp_M.to_sparse()
    return Lp_M
    
    
def Dirichlet_Energy_norm(adj):
    """
    Applies the normalization for dirichlet energy graphs:
        \mathbf{I} - 1/2 （\mathbf{D}^{-1} \mathbf{A} + \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}）
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)
    in_deg_inv = in_deg.pow_(-1.0)
    in_deg_inv.masked_fill_(in_deg_inv == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)
    out_deg_inv = out_deg.pow_(-1.0)
    out_deg_inv.masked_fill_(out_deg_inv == float("inf"), 0.0)

    adjD2 = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    D2adjD2 = mul(adjD2, in_deg_inv_sqrt.view(1, -1))
    D2adjD2 = D2adjD2.to_dense()
    
    adjDo = mul(adj, out_deg_inv.view(-1, 1))
    adjDo = adjDo.to_dense()
    
    adjDi = mul(adj, in_deg_inv.view(-1, 1))
    adjDi = adjDi.to_dense()
    
    idm = torch.eye(adj.to_dense().shape[0]) * 0.4
    idm = idm.to("cuda")
    
    Lp_E = idm - 0.25 * (adjDo + adjDi + 2 * D2adjD2)
    return Lp_E


def get_norm_adj(adj, norm, q=0.25):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "vanilla":
        return vanilla_Lp(adj)
    elif norm == "vanillanorm":
        return vanilla_Lp_norm(adj)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    elif norm == "I-sym":
        return directed_degree_norm(adj)
    elif norm == "MagNet":
        return MagNet_norm(adj, q)
    elif norm == "Dirichlet_Energy":
        return Dirichlet_Energy_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")


def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_adj(edge_index, num_nodes, graph_type="directed"):
    """
    Return the type of adjacency matrix specified by `graph_type` as sparse tensor.
    """
    if graph_type == "transpose":
        edge_index = torch.stack([edge_index[1], edge_index[0]])
    elif graph_type == "undirected":
        edge_index = torch_geometric.utils.to_undirected(edge_index)
    elif graph_type == "directed":
        pass
    else:
        raise ValueError(f"{graph_type} is not a valid graph type")

    value = torch.ones((edge_index.size(1),), device=edge_index.device)
    return SparseTensor(row=edge_index[0], col=edge_index[1], value=value, sparse_sizes=(num_nodes, num_nodes))


def compute_unidirectional_edges_ratio(edge_index):
    num_directed_edges = edge_index.shape[1]
    num_undirected_edges = torch_geometric.utils.to_undirected(edge_index).shape[1]

    num_unidirectional = num_undirected_edges - num_directed_edges

    return (num_unidirectional / (num_undirected_edges / 2)) * 100
