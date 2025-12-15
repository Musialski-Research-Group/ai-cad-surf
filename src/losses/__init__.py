from .digs_loss import DiGSLoss
from .neur_cad_recon_loss import NeurCADReconLoss
from .neural_singular_hessian_loss import NeuralSingularHessianLoss
from .neur_cad_recon_fd_loss import NeurCADReconLossFD
from .neural_singular_hessian_fd_loss import NeuralSingularHessianLossFD
from .odw_loss import OffDiagonalWeingartenLoss

LOSSES = {
    'digs': DiGSLoss,
    'neur_cad_recon': NeurCADReconLoss,
    'neural_singular_hessian': NeuralSingularHessianLoss,
    'neur_cad_recon_fd': NeurCADReconLossFD,
    'neural_singular_hessian_fd': NeuralSingularHessianLossFD,
    'odw': OffDiagonalWeingartenLoss,
    }
