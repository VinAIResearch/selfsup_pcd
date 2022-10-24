from .dgcnn_classification import DGCNN, get_loss
from .dgcnn_critical import DGCNN_critical
from .dgcnn_part_segmentation import DGCNN_part_seg
from .dgcnn_sem_segmentation import DGCNN_seg
from .dgcnn_svm import DGCNN_jigsaw, DGCNN_svm


__all__ = ["DGCNN", "get_loss", "DGCNN_critical", "DGCNN_part_seg", "DGCNN_seg", "DGCNN_jigsaw", "DGCNN_svm"]
