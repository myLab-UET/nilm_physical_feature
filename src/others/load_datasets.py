import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append("/home/mrcong/Code/mylab-nilm-files/nilm-physical-features/src/common")

from nilm_dao import *

# Getting data from vndale1
train_df = get_vndale1_data(data_type="train", rms_window_size=1800, is_norm=True)
test_df = get_vndale1_data(data_type="test", rms_window_size=1800, is_norm=True)
print(f"VNDALE1 train data shape: {train_df.shape}")
print(f"VNDALE1 test data shape: {test_df.shape}")

# Getting iawe data
train_df = get_iawe_data("train", is_norm=True)
test_df = get_iawe_data("test", is_norm=True)
print(f"iAWE train data shape: {train_df.shape}")
print(f"iAWE test data shape: {test_df.shape}")

# Getting rae data
train_df = get_rae_data("train", is_norm=True)
test_df = get_rae_data("test", is_norm=True)
print(f"RAE train data shape: {train_df.shape}")
print(f"RAE test data shape: {test_df.shape}")
