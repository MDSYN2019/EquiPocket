# coding=utf-8
"""
The EquiPocket for our work (EquiPocket: an E(3)-Equivariant Geometric Graph Neural
Network for Ligand Binding Site Prediction)
"""

import torch
from torch import nn
from torch_geometric.nn import MLP, global_mean_pool, global_max_pool, radius_graph

from .baseline_models import Baseline_Models
from .surface_egnn import SurfaceEGNN


def get_cutoff_ratio(pos, cutoff, surface_egnn_depth):
    all_ratio = []
    dist = torch.cdist(pos, pos)
    all_atom = pos.shape[0]
    all_ratio.append(torch.ones(all_atom, 1).fill_(all_atom).long().to(dist.device))
    for i in range(0, surface_egnn_depth + 1):
        tmp_result = (dist < cutoff * i).sum(dim=1)
        tmp_result = tmp_result.unsqueeze(dim=-1)
        tmp_result = tmp_result / all_atom
        all_ratio.append(tmp_result)
    all_ratio = torch.concat(all_ratio, dim=1)
    return all_ratio


class EquiPocket(nn.Module):
    def __init__(
        self,
        local_geometric_modeling=True,
        global_structure_modeling="gat_egnn",
        surface_egnn_depth=4,
        cutoff=6,
        dense_attention=True,
        out_depth=2,
        out_features=128,
    ):
        """
        The dimension of the input data is as follows:

        
        Data(x=[5690, 6],
        edge_index=[2, 11594],
        edge_attr=[11594, 3],
        pos=[5690, 3],
        atom_in_surface=[5690],
        vert_surface=[0, 9],
        vert_pos=[0, 3],
        vert_atom=[0],
        vert_num=0,
        vert_atom_diff=[0, 3],
        vert_batch=[0],
        surface_center_pos=[0, 3],
        surface_descriptor=[0, 7])        
        """
        
        super(EquiPocket, self).__init__()
        self.dense_attention = dense_attention
        self.local_geometric_modeling = local_geometric_modeling # local geometric modeling
        self.global_structure_modeling = global_structure_modeling # global structure modeling 
        self.surface_egnn_depth = surface_egnn_depth # surface egnn depeth 
        self.dense_attention = dense_attention
        self.cutoff = cutoff
        self.out_depth = out_depth
        self.out_features = out_features
        
        atom_channels = 16
        bond_channels = 16
        trans_input_features = 0

        # local_geometric_modeling
        if self.local_geometric_modeling:
        # create the MLP for the local geometric feature and the surface feature 
            self.trans_local_geometric_feature = MLP(
                in_channels=7,
                hidden_channels=out_features // 2,
                out_channels=out_features // 2,
                dropout=0.1,
                num_layers=out_depth,
            ) # this outputs the local geometric embedding for each node, which is then pooled to get the geometric embedding for each node, which has out_features dimensions for each node 
            
            self.trans_surface_feature = MLP(
                in_channels=14,
                hidden_channels=out_features,
                out_channels=out_features,
                dropout=0.1,
                num_layers=out_depth,
            ) # this outputs the surface size embedding for each node, which has out_features dimensions for each node 
            trans_input_features += 2 * out_features # the geomtric emebedding and the surface size embedding are concatenated together, which has 2 * out features dimensions for each node        
            
        # global_structure_modeling - embeds the global structure of the protein through the GNNs
        if self.global_structure_modeling == "gat_egnn":
            self.global_structure_modeling_model = Baseline_Models(
                atom_channels=atom_channels,
                bond_channels=bond_channels,
                out_features=out_features,
                gat_depth=1,
                gcn_depth=0,
                egnn_depth=3,
            )
            trans_input_features += out_features # The global structure emebedding is concatenated with the local geometric embedding and the surface size embedding, which has out_features dimensions for each node 
            
        # concat features 
        self.trans_geo_feature = MLP(
            in_channels=trans_input_features,
            hidden_channels=out_features,
            out_channels=out_features,
            dropout=0.1,
            num_layers=out_depth,
        ) 

        # surface_egnn_depth
        if self.surface_egnn_depth > 0: # if the surface EGNN depth is greater than 0, we will use the surface EGNN to further model the surface features, which can capture the local geometric features of the surface and the relative position between the nodes on the surface

            self.surface_egnn = SurfaceEGNN(
                in_node_nf=out_features,
                hidden_nf=out_features,
                out_node_nf=out_features,
                n_layers=surface_egnn_depth,
            ) # the surface EGNN takes the node embedding as input and outputs the new node embedding after passing through the surface EGNN, which has out_features dimensions foe each node

            if self.dense_attention: # calculate the attention for each node after the surface EGNN, which is based on the cutoff ratio of each node 

                self.cal_attention = nn.Sequential()
                attention_in_features = surface_egnn_depth + 2
                mlp = MLP(
                    in_channels=attention_in_features,
                    hidden_channels=out_features,
                    out_channels=surface_egnn_depth + 1,
                    dropout=0.1,
                    num_layers=out_depth,
                )

                
                self.cal_attention.add_module("cal_attention", mlp)
                self.cal_attention.add_module("sigmoid", nn.Sigmoid())

                
        # predict
        last_out_feature = out_features * (surface_egnn_depth + 1) # if the surface EGNN depth is greater than 0, we will concatenate the node embedding after each layer of the surface EGNN, which has out_features dimensions for each layer, so the total dimension will be out_features * (surface_egnn_depth + 1)x
        self.all_out = MLP(
            in_channels=last_out_feature,
            norm=None,
            hidden_channels=last_out_feature,
            out_channels=1,
            dropout=0.1,
            num_layers=out_depth,
        ) # this outputs the probability of each node being a pocket node, which has 1 dimension for each node 

    def forward(self, batch_data):
        atom_in_surface = batch_data.atom_in_surface
        pos = batch_data.pos
        surface_center_pos = batch_data.surface_center_pos
        batch_index = getattr(batch_data, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(
                pos.size(0), dtype=torch.long, device=pos.device
            )
        node_embedding = []
        
        # local geometric embedding
        if self.local_geometric_modeling: # modelling through the local_geometric modelling
            
            surface_descriptor = batch_data.surface_descriptor # the surface descriptor is the local geometric feature for each node, which is calculated by the MSMS software and has 7 dimensions 
            vert_batch = batch_data.vert_batch
            
            local_geometric_embedding = self.trans_local_geometric_feature(
                 surface_descriptor
             ) # Takes the surface descriptor as input and outputs the local geometric embedding for each node, which has out_features // 2 dimensions for each node 
        
            geometric_embedding = torch.concat(
                [
                    global_mean_pool(local_geometric_embedding, vert_batch), # for each node, we pool the local geometric embedding of its neighbours nodes to get the geometr
                    global_max_pool(local_geometric_embedding, vert_batch), # max pooling to get pooled features for each node, which has out_features dimensions for each node 
                ],
                dim=-1,
            )
            #
            surface_size = torch.concat(
                [
                    global_mean_pool(surface_descriptor, vert_batch), # for each node, we pool the surface descriptor of its neighbours nodes to get the surface size embedding for each node, which has out_features dimensions for each node 
                    global_max_pool(surface_descriptor, vert_batch),  # max pooling to get pooled featurs for each node, which has out_features dimensions for each node
                ],
                dim=1,
            )
            surface_size_embedding = self.trans_surface_feature(surface_size)
            node_embedding += [geometric_embedding, surface_size_embedding] # the local geometric embedding and the surface size embedding are concatenated together, which has 2 * out_features dimensions for each node 

        # global_structure_modeling
        if self.global_structure_modeling:            
            global_structure_node_embedding_all = self.global_structure_modeling_model(
                batch_data
            )
            global_structure_node_embedding = global_structure_node_embedding_all[
                atom_in_surface == 1
            ]
            node_embedding.append(global_structure_node_embedding)
        node_embedding = torch.concat(node_embedding, dim=1) # concatenate the local geometric embedding, the surface size embedding, and the global structure emebedding toegether 

        # trans 3 * out_features -> out_features
        node_embedding = self.trans_geo_feature(node_embedding)
     
        # Surface passing
        if self.surface_egnn_depth > 0:
            new_batch = batch_index[atom_in_surface == 1] # get the batch index for the nodes on the surface 
            surface_pos = batch_data.pos[atom_in_surface == 1]  # get the position for the nodes on the surface 

            all_node_embedding = [] # the list to store the node embedding after passing through the surface EGNN for each graph in the batch
            all_node_pos = [] # the list to store the node position after passing through the surface EGNN for each graph in the batch

            for graph_id in new_batch.unique():
                print(f"graph id {graph_id}")
                
                tmp_node_embedding = node_embedding[new_batch == graph_id] # get the node embedding for the nodes on the surface for the current graph in the batch

                graph_surface_pos = surface_pos[new_batch == graph_id].clone().detach() # get the position for the nodes on the surface for the current graph in the batch

                edge_index = radius_graph( # construct the graph for the nodes on the surface for the current graph in the batch based on the radius graph, which connects the nodes that are within the cutoff idsntace 
                    graph_surface_pos, r=self.cutoff, max_num_neighbors=999 
                ) # get the max number of neighbors for each node, which is set to 999 here, which means that will not limit the number of neighbours for each node 

                full_pos = torch.concat(
                    (
                        surface_pos[new_batch == graph_id].unsqueeze(dim=1), # position of the nodes on the surface 
                        surface_center_pos[new_batch == graph_id].unsqueeze(dim=1), # position of the surface center
                    ),
                    dim=1,
                ) # full position for the nodes on the surface, which is the concatenation of the position of the nodes on the surface and the positoon of the surface center
                
                new_node_embedding, new_pos = self.surface_egnn(
                    tmp_node_embedding, full_pos, edge_index, edge_index
                ) # pass the node emebedding and the full position through the surface EGNN to get the new node embedding and the new position for the nodes on the surface for the current graph in the batch, which has out_features dimensions for each
                
                all_node_embedding.append(new_node_embedding)
                all_node_pos.append(new_pos)

                print(f"all node embedding shape {new_node_embedding.shape}") 
                print(f"all node pos shape {new_pos.shape}")

                
            # all out
            node_embedding = torch.concat(all_node_embedding, dim=0) # concatenate the node embedding after passing through the surface EGNN for each graph in the batch, which has out_features dimensions for each node
            node_pos = torch.concat(all_node_pos, dim=0)[:, 0] # concatenate the node positions after passing through the surface EGNN for each graph in the batch, which has 3 dimensions for each node, and we only take the first 3 dimensions as the new position for each node
            
            if self.surface_egnn_depth > 0 and self.dense_attention:
                tmp_cutoff_ratio = batch_data.cutoff_ratio[
                    batch_data.atom_in_surface == 1
                ]
                cutoff_attention = self.cal_attention(tmp_cutoff_ratio)
                cutoff_attention = torch.repeat_interleave(
                    cutoff_attention, self.out_features, dim=1
                )
                node_embedding = node_embedding * cutoff_attention
                
        # probability and relative direction
        y_hat = self.all_out(node_embedding)        
        angle = None
        if self.surface_egnn_depth > 0:
            angle = node_pos - pos[atom_in_surface == 1]
        return y_hat, angle


if __name__ == "__main__":
    cutoff = 6
    out_depth = 2
    out_features = 128
    dense_attention = True
    local_geometric_modeling = True # enable the local geometric modeling, which models the local geometric features of the protein surface through the MLPs and pooling operationss
    global_structure_modeling = "gat_egnn" # enable the global structure modeling
    surface_egnn_depth = 4
    model = EquiPocket(
        local_geometric_modeling=local_geometric_modeling,
        global_structure_modeling=global_structure_modeling,
        surface_egnn_depth=surface_egnn_depth,
        cutoff=cutoff,
        dense_attention=dense_attention,
        out_depth=out_depth,
        out_features=out_features,
    )


    
