import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== GeometricPoint branch ====================
class ArcLengthEncoding(nn.Module):
    """Fiber-specific position coding"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.position_embedding = nn.Parameter(torch.randn(15, channels))
        
    def forward(self, fibers):
        B = fibers.shape[0]
        positions = torch.linspace(0, 1, 15, device=fibers.device).unsqueeze(0).repeat(B, 1)
        pos_embed = positions.unsqueeze(-1) * self.position_embedding.unsqueeze(0)
        return pos_embed

class GeometricAttention(nn.Module):
    """
    Geometric-aware Attention for Fiber Representation

    Geometry encoded:
    - position
    - arc-length
    - tangent (1st derivative)
    - curvature (2nd derivative)
    """

    def __init__(self, channels, num_heads=4, dropout=0.1, num_points=15):
        super().__init__()

        assert channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_points = num_points

        # QKV
        self.qkv = nn.Linear(channels, channels * 3)

        # output projection
        self.proj = nn.Linear(channels, channels)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # relative position bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_points, num_points))

        # geometric mlps
        self.arc_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)
        )

        self.dir_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)
        )

        self.curv_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)
        )

        # geometry encoding
        self.geo_encoder = nn.Linear(8, channels)


    # --------------------------------------------------
    # geometry encodings
    # --------------------------------------------------

    def compute_arc_length(self, points):

        diff = points[:, 1:] - points[:, :-1]   # B N-1 3
        seg_len = torch.norm(diff, dim=-1)      # B N-1

        arc = torch.cumsum(seg_len, dim=1)
        arc = torch.cat(
            [torch.zeros(points.shape[0],1,device=points.device), arc],
            dim=1
        )

        return arc


    def compute_tangent(self, points):

        tangent = points[:,1:] - points[:,:-1]

        tangent = F.normalize(tangent, dim=-1)

        tangent = torch.cat([tangent, tangent[:,-1:].clone()], dim=1)

        return tangent


    def compute_curvature(self, tangent):

        curv = tangent[:,1:] - tangent[:,:-1]

        curv = torch.norm(curv, dim=-1)

        curv = torch.cat([curv, curv[:,-1:].clone()], dim=1)

        return curv


    # --------------------------------------------------
    # geometry bias
    # --------------------------------------------------

    def compute_geometric_bias(self, points):

        B, N, _ = points.shape

        # arc-length distance
        arc = self.compute_arc_length(points)

        arc_i = arc.unsqueeze(2)
        arc_j = arc.unsqueeze(1)

        arc_dist = torch.abs(arc_i - arc_j)

        # tangent
        tangent = self.compute_tangent(points)

        t_i = tangent.unsqueeze(2)
        t_j = tangent.unsqueeze(1)

        dir_sim = (t_i * t_j).sum(-1)

        # curvature
        curv = self.compute_curvature(tangent)

        c_i = curv.unsqueeze(2)
        c_j = curv.unsqueeze(1)

        curv_diff = torch.abs(c_i - c_j)

        # reshape
        arc_flat = arc_dist.reshape(-1,1)
        dir_flat = dir_sim.reshape(-1,1)
        curv_flat = curv_diff.reshape(-1,1)

        arc_bias = self.arc_mlp(arc_flat)
        dir_bias = self.dir_mlp(dir_flat)
        curv_bias = self.curv_mlp(curv_flat)

        bias = arc_bias + dir_bias + curv_bias

        bias = bias.reshape(B,N,N,self.num_heads).permute(0,3,1,2)

        return bias


    # --------------------------------------------------
    # geometry feature encoding
    # --------------------------------------------------

    def geometry_features(self, points):

        arc = self.compute_arc_length(points).unsqueeze(-1)

        tangent = self.compute_tangent(points)

        curv = self.compute_curvature(tangent).unsqueeze(-1)

        geo = torch.cat([
            points,
            tangent,
            arc,
            curv
        ], dim=-1)

        return geo


    # --------------------------------------------------
    # forward
    # --------------------------------------------------

    def forward(self, x, points):

        B, N, C = x.shape

        # geometry encoding
        geo = self.geometry_features(points)

        geo_feat = self.geo_encoder(geo)

        x = x + geo_feat

        # qkv
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        )

        qkv = qkv.permute(2,0,3,1,4)

        q, k, v = qkv.unbind(0)

        # attention
        attn = (q @ k.transpose(-2,-1)) * self.scale

        # add biases
        attn = attn + self.rel_pos_bias.unsqueeze(0).unsqueeze(0)

        geo_bias = self.compute_geometric_bias(points)

        attn = attn + geo_bias

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v)

        x = x.transpose(1,2).reshape(B,N,C)

        x = self.proj(x)

        x = self.proj_drop(x)

        return x
    
class GeometricPointNetfeat(nn.Module):
    """Point Cloud Feature Extractor for Geometric Perception"""
    def __init__(self, use_attention=True, use_arc_encoding=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_arc_encoding = use_arc_encoding
        
        # Point-wise feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Geometric Perception Module
        if self.use_arc_encoding:
            self.arc_encoding = ArcLengthEncoding(256)
        
        if self.use_attention:
            self.attention = GeometricAttention(256, num_heads=4)
            self.ln1 = nn.LayerNorm(256)
        
        # Global Feature Extraction
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        
    def forward(self, x, return_intermediate=False):
        # x: [B, 3, 15]
        B = x.shape[0]
        
        points_original = x.transpose(1, 2)  # [B, 15, 3]
        
        # Point-wise feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 15]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, 15]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 256, 15]
        
        # Convert to sequence format
        x_seq = x.transpose(1, 2)  # [B, 15, 256]
        
        # Geometric Perception Enhancement
        if self.use_arc_encoding:
            pos_embed = self.arc_encoding(points_original)
            x_seq = x_seq + pos_embed
        
        if self.use_attention:
            residual = x_seq
            x_seq = self.attention(x_seq, points_original)
            x_seq = self.ln1(x_seq + residual)
        
        # Convert back to convolutional format
        x = x_seq.transpose(1, 2)  # [B, 256, 15]
        
        # Global Feature Aggregation
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 512, 15]
        x = F.relu(self.bn5(self.conv5(x)))  # [B, 1024, 15]
        
        # Global Max Pooling
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        x = x.view(B, 1024)  # [B, 1024]
        
        if return_intermediate:
            return x, x_seq  
        return x

# ==================== ASGM branch ====================
class SpatialAttention(nn.Module):
    """Spatial Attention Mechanism"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, H, W]
        attn_map = self.sigmoid(self.conv(x))  # [B, 1, H, W]
        return x * attn_map

class AdaptiveSpatialCNN(nn.Module):
    """Adaptive Spatial Convolutional Neural Network"""
    def __init__(self, num_classes=8, base_channels=32, num_conv_blocks=4, dropout_rate=0.25):
        super().__init__()
        self.num_conv_blocks = num_conv_blocks
        
        # Dynamic Construction of Convolution Blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 3
        
        for i in range(num_conv_blocks):
            out_channels = base_channels * (2 ** i)
            
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                SpatialAttention(out_channels), 
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout_rate)
            )
            
            self.conv_blocks.append(conv_block)
            in_channels = out_channels
        
        # Global Adaptive Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        final_channels = base_channels * (2 ** (num_conv_blocks - 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_intermediate=False):
        if len(x.shape) == 4 and x.shape[1] != 3:
            if x.shape[3] == 3:  # [B, H, W, C]
                x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Storing intermediate features
        intermediate_features = []
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            intermediate_features.append(x)
        
        x = self.global_pool(x)  # [B, C, 1, 1]
        
        x = self.fc(x)  # [B, 512]
        
        if return_intermediate:
            return x, intermediate_features
        return x

# ==================== Feature Fusion Module ====================
class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion"""
    def __init__(self, point_channels, spatial_channels, fusion_channels=512):
        super().__init__()
        
        # Point cloud feature projection
        self.point_proj = nn.Sequential(
            nn.Linear(point_channels, fusion_channels),
            nn.LayerNorm(fusion_channels),
            nn.ReLU()
        )
        
        # Spatial Feature Projection
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_channels, fusion_channels),
            nn.LayerNorm(fusion_channels),
            nn.ReLU()
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_channels,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Integrated gate control mechanism
        self.gate = nn.Sequential(
            nn.Linear(fusion_channels * 2, fusion_channels),
            nn.Sigmoid()
        )
        
    def forward(self, point_feat, spatial_feat):
        # Projected onto the shared space
        point_proj = self.point_proj(point_feat).unsqueeze(1)  # [B, 1, D]
        spatial_proj = self.spatial_proj(spatial_feat).unsqueeze(1)  # [B, 1, D]
        
        combined = torch.cat([point_proj, spatial_proj], dim=1)  # [B, 2, D]
        
        # Cross-modal attention
        attended, _ = self.cross_attention(combined, combined, combined)
        
        # Gate-controlled fusion
        gate_values = self.gate(torch.cat([
            attended[:, 0, :],  
            attended[:, 1, :]   
        ], dim=1))  # [B, D]
        
        # Weighted fusion
        fused = gate_values * attended[:, 0, :] + (1 - gate_values) * attended[:, 1, :]
        
        return fused

# ==================== Main network ====================
class FusionNet(nn.Module):
    """Dual-branch fusion network"""
    def __init__(self, 
                 num_classes=8,
                 use_arc_encoding=True,
                 use_attention=True,
                 spatial_base_channels=32,
                 spatial_num_blocks=4,
                 fusion_method='cross_attention',
                 fusion_dim=512,
                 dropout_rate=0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        classifier_input_dim = fusion_dim

        # ========== GeometricPointNet Branch==========
        self.point_branch = GeometricPointNetfeat(
            use_attention=use_attention,
            use_arc_encoding=use_arc_encoding
        )
        
        # ========== ASGM Branch ==========
        self.spatial_branch = AdaptiveSpatialCNN(
            num_classes=num_classes, 
            base_channels=spatial_base_channels,
            num_conv_blocks=spatial_num_blocks,
            dropout_rate=dropout_rate
        )
        

        self.fusion_module = CrossModalAttention(
            point_channels=1024,
            spatial_channels=512,
            fusion_channels=fusion_dim
        )
            
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(128, num_classes)
        )
        
        self.branch_weights = nn.Parameter(torch.ones(2))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, point_cloud, fibermap, return_features=False):
        """
        parameters:
            point_cloud: Point cloud data [B, 15, 3]
            fibermap: Spatial grid data [B, 32, 32, 3]
        
        Return:
            logits: Classification logits [B, num_classes]
        """
        
        # ========== GeometricPointNet Branch==========
        if point_cloud.shape[1] == 15:  # [B, 15, 3]
            point_cloud = point_cloud.transpose(1, 2)  # [B, 3, 15]
        
        point_feat = self.point_branch(point_cloud)  # [B, 1024]
        
        # ========== ASGM Branch ==========
        if len(fibermap.shape) == 4:
            if fibermap.shape[1] != 3 and fibermap.shape[3] == 3:
                fibermap = fibermap.permute(0, 3, 1, 2)  # [B, 3, H, W]
        
        spatial_feat = self.spatial_branch(fibermap)  # [B, 512]
        
        # ========== Feature fusion ==========
        fused_feat = self.fusion_module(point_feat, spatial_feat)
        
        # ========== Classification ==========
        logits = self.classifier(fused_feat)
        
        if return_features:
            features = {
                'point_features': point_feat,
                'spatial_features': spatial_feat,
                'fused_features': fused_feat,
                'branch_weights': F.softmax(self.branch_weights, dim=0)
            }
            return logits, features
        
        return logits
    
