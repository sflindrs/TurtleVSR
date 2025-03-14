import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedCrossCorrelation(nn.Module):
    """
    Custom implementation of Normalized Cross-Correlation
    
    Args:
        return_map (bool): If True, returns the correlation map. If False, returns the mean correlation value.
        reduction (str): Reduction method for the correlation map. Options: 'mean', 'sum', 'none'.
    """
    def __init__(self, return_map=False, reduction='mean'):
        super(NormalizedCrossCorrelation, self).__init__()
        self.return_map = return_map
        self.reduction = reduction
        
    def forward(self, x, y):
        """
        Compute normalized cross-correlation between x and y.
        
        Args:
            x (Tensor): First input tensor of shape (B, C, H, W)
            y (Tensor): Second input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: If return_map is True, returns correlation map. Otherwise, returns scalar value.
        """
        # Get tensor dimensions
        B, C, H, W = x.shape
        
        # Reshape inputs
        x_flat = x.view(B, C, -1)
        y_flat = y.view(B, C, -1)
        
        # Normalize along the flattened spatial dimensions
        x_mean = torch.mean(x_flat, dim=2, keepdim=True)
        y_mean = torch.mean(y_flat, dim=2, keepdim=True)
        
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean
        
        # Compute the normalization factors
        x_norm = torch.sqrt(torch.sum(x_centered**2, dim=2, keepdim=True) + 1e-8)
        y_norm = torch.sqrt(torch.sum(y_centered**2, dim=2, keepdim=True) + 1e-8)
        
        # Normalize the inputs
        x_normalized = x_centered / x_norm
        y_normalized = y_centered / y_norm
        
        # Compute correlation
        correlation = torch.sum(x_normalized * y_normalized, dim=1)
        
        # Reshape back to spatial dimensions if needed
        if self.return_map:
            correlation = correlation.view(B, H, W)
        
        # Apply reduction if not returning map
        if not self.return_map:
            if self.reduction == 'mean':
                correlation = torch.mean(correlation, dim=1)
            elif self.reduction == 'sum':
                correlation = torch.sum(correlation, dim=1)
            # 'none' does no reduction
        
        return correlation + 1  # Add 1 to match the convention in your code (where you subtract 1 after calling)