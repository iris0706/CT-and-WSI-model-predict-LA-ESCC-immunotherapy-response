"""Settings reported in the manuscript (Methods: Image preprocessing, Model architecture, Training)."""

# CT preprocessing
CT_MARGIN = 10                      # voxels added around the tumor mask bounding box
CT_WINDOW_WIDTH = 400               # soft-tissue window (HU)
CT_WINDOW_LEVEL = 40
CT_SIZE = 128                       # each slice resized to 128 x 128
CT_SLICES_TRAIN = 80                # volume resampled to 80 slices for training
CT_SLICES_INFER = 5                 # and to 5 slices for inference; the CT branch ends in adaptive pooling
CT_ROTATION = 10                    # training volumes augmented once at loading: rotation within +/-10 degrees
                                    # and a horizontal flip, identical for all slices

# WSI patch embeddings (icrnet/wsi.py, scripts/extract_wsi_features.py)
WSI_ENCODER = "uni2h"               # pathology foundation model UNI, UNI2-h release (ViT-H/14), used frozen
WSI_TARGET_MPP = 0.5                # patches taken at 0.5 um per pixel (20x equivalent), the working resolution of UNI2-h,
                                    # by downsampling each slide from its native resolution (MPP read from the scanner metadata)
WSI_PATCH_SIZE = 224                # pixels at the target resolution (the native input size of UNI2-h)
WSI_FEATURE_DIM = 1536              # UNI2-h embedding dimension
# The earlier pipeline (ImageNet ResNet-50 truncated after layer3, 256-pixel patches at 40x, 1,024-d features) is kept
# as `--encoder resnet50_l3 --mpp 0.25 --patch 256`; the models take the embedding dimension as an argument.
FEATURE_DIM = 512                   # CT vector, reduced patch embeddings and fused vector

# Training: episodic, on class-balanced tasks drawn from the training cohort
NUM_CLASSES = 2                     # 0 = non-responder (iSD/iUPD/iCPD), 1 = responder (iCR/iPR)
N_SHOT = 5                          # support set: 5 responders + 5 non-responders
N_QUERY = 5                         # query set (monitoring only): 5 responders + 5 non-responders
TASKS_PER_EPOCH = 50
INNER_STEPS = 5                     # SGD steps on each support set
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3                 # unweighted cross-entropy
SEED = 42                           # a single model
DROPOUT = {"ct": 0.4, "wsi": 0.4, "fusion": 0.0}   # the fused model applied no dropout
INFERENCE_BATCH_SIZE = 16

# Cohorts (manifest column `cohort`): training, internal_test, external_test, prospective_test. The model is trained on
# the training cohort only; the three test cohorts are each evaluated once with the fixed weights and threshold.
# Development split of the Center 1 retrospective cohort (405 patients)
INTERNAL_TEST_FRACTION = 0.2        # 324 training / 81 internal test, stratified on response
SPLIT_RANDOM_STATE = 42

# Frozen response-probability threshold: the training-cohort probability that maximized Youden's index
THRESHOLD = 0.603019
