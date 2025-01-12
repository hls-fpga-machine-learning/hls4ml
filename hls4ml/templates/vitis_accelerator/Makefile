# Check environment ###########################################################

ifndef XILINX_VITIS
$(error XILINX_VITIS variable is not set, please set correctly and rerun)
endif

ifndef XILINX_XRT
$(error XILINX_XRT variable is not set, please set correctly and rerun)
endif

ifndef XILINX_VIVADO
$(error XILINX_VIVADO variable is not set, please set correctly and rerun)
endif

ifneq ($(shell expr $(shell g++ -dumpversion) \>= 5), 1)
CXX := $(XILINX_VIVADO)/tps/lnx64/gcc-6.2.0/bin/g++
$(warning [WARNING]: g++ version older. Using g++ provided by the tool: $(CXX))
endif

# Configuration variables #####################################################

# Absolute path to top directory of accelerator project
PWD := $(shell pwd)

# Target (hw, hw_emu, sw_emu)
TARGET ?= hw

# Accelerator card configuration file
CARD_CFG ?= accelerator_card.cfg

# Platform (currently extracted from accelerator_card.cfg if not already set)
PLATFORM ?= $(shell awk -F '=' '/platform=/ {print $$2}' $(CARD_CFG))

# Board Type (determines whether design will go through packaging step)
BOARD_TYPE := #BOARDTYPE

# Kernel name
KERNEL_NAME := #PRJNAME

# Wrapper name
WRAPPER_NAME := kernel_wrapper

# Top level build directory
BUILD_DIR := ./build_$(TARGET)
ifdef DEBUG
BUILD_DIR := $(BUILD_DIR)_deb
else
BUILD_DIR := $(BUILD_DIR)_rel
endif

# Directories for kernel synthesis
XO_DIR := $(BUILD_DIR)/xo
XCLBIN_DIR := $(BUILD_DIR)/xclbin

# CC flags for v++
XOCCFLAGS := -t $(TARGET) --config $(CARD_CFG) --messageDb=$(BUILD_DIR)/kernel_wrapper.mdb

# Linker flags for V++
XOLDFLAGS := -t $(TARGET) --config $(CARD_CFG) --messageDb=$(BUILD_DIR)/kernel_wrapper.mdb

# C++ compiler & linker flags
CXXFLAGS := -Wall -std=c++11 -Wno-unknown-pragmas
LDFLAGS = -L$(XILINX_XRT)/lib/ -lstdc++ -lpthread -lrt -lOpenCL

ifdef DEBUG
XOCCFLAGS += -g
XOLDFLAGS += -g
CXXFLAGS += -g -O0
else
# Optimization flags can be added here
endif

.PHONY: all xclbin hls run clean cleanhls cleanxclbin ultraclean

all: xclbin host

# Kernel C/RTL synthesis ######################################################

HLS_INCLUDES := -I./ -I./firmware/ -I./firmware/weights -I./firmware/nnet_utils/

$(BUILD_DIR)/$(KERNEL_NAME)_kernel.xo: $(WRAPPER_NAME).cpp firmware/$(KERNEL_NAME).cpp
	mkdir -p $(XO_DIR)
	v++ -c $(XOCCFLAGS) --temp_dir $(XO_DIR) --log_dir $(XO_DIR) -o $@ $^ $(HLS_INCLUDES)

hls: $(BUILD_DIR)/$(KERNEL_NAME)_kernel.xo

# Kernel linking & packaging ##################################################

ifneq (,$(findstring versal,$(BOARD_TYPE)))

# For Versal architecture, linking and packaging are separate steps
$(BUILD_DIR)/$(WRAPPER_NAME).xsa: $(BUILD_DIR)/$(KERNEL_NAME)_kernel.xo
	mkdir -p $(XCLBIN_DIR)
	v++ -l $(XOLDFLAGS) --temp_dir $(XCLBIN_DIR) --log_dir $(XCLBIN_DIR) -o $@ $^

# VCK5000 specific packaging
XOCCPFLAGS := -t $(TARGET) -f $(PLATFORM) --package.boot_mode=ospi --messageDb=$(BUILD_DIR)/kernel_wrapper.mdb
$(BUILD_DIR)/$(WRAPPER_NAME).xclbin: $(BUILD_DIR)/$(WRAPPER_NAME).xsa
	v++ -p $(XOCCPFLAGS) --temp_dir $(XCLBIN_DIR) --log_dir $(XCLBIN_DIR) -o $@ $^

else

# For Standard Alveo, a single step is required for linking and packaging
# This is standard Alveo linking and packaging
$(BUILD_DIR)/$(WRAPPER_NAME).xclbin: $(BUILD_DIR)/$(KERNEL_NAME)_kernel.xo
	mkdir -p $(XCLBIN_DIR)
	v++ -l $(XOLDFLAGS) --temp_dir $(XCLBIN_DIR) --log_dir $(XCLBIN_DIR) -o $@ $^

endif

xclbin: $(BUILD_DIR)/$(WRAPPER_NAME).xclbin

# Host compilation ############################################################

INCLUDES := -I$(XILINX_XRT)/include/ -I$(XILINX_VIVADO)/include/ -I$(XILINX_HLS)/include/
INCLUDES += -I$(PWD)/libs/ -I$(PWD)/firmware/ -I$(PWD)/firmware/nnet_utils/

host: $(KERNEL_NAME)_host_cl.cpp libs/xcl2.cpp $(wildcard libs/*.hpp)
	$(CXX) $(CXXFLAGS) $(KERNEL_NAME)_host_cl.cpp libs/xcl2.cpp -o $@ $(INCLUDES) $(LDFLAGS)

# Execute program #############################################################

run: ./host $(BUILD_DIR)/$(WRAPPER_NAME).xclbin
ifeq ($(TARGET), hw)
	@echo "TARGET is hw, not setting XCL_EMULATION_MODE"
	$(eval EMULATION_MODE :=)
else
	@echo "Setting XCL_EMULATION_MODE to $(TARGET)"
	$(eval EMULATION_MODE := XCL_EMULATION_MODE=$(TARGET))
endif
	@cd firmware && $(EMULATION_MODE) ../host ../$(BUILD_DIR)/$(WRAPPER_NAME).xclbin $(PROFILING_DATA_REPEAT_COUNT)

# Cleanup #####################################################################

cleanxclbin:
	rm -rf host tb_data/hw_results.dat tb_data/tb_input_features.dat
	rm -rf *$(WRAPPER_NAME)*.log
	rm -rf $(BUILD_DIR)/$(WRAPPER_NAME).xclbin.* $(BUILD_DIR)/$(WRAPPER_NAME).xsa* $(BUILD_DIR)/$(WRAPPER_NAME).ltx $(BUILD_DIR)/$(WRAPPER_NAME).mdb
	rm -rf $(XCLBIN_DIR)

cleanhls:
	rm -rf *$(KERNEL_NAME)_kernel*.log
	rm -rf $(BUILD_DIR)/$(KERNEL_NAME)_kernel.xo.*
	rm -rf $(XO_DIR)

clean: cleanxclbin cleanhls

ultraclean:
	rm -rf host tb_data/hw_results.dat tb_data/tb_input_features.dat *.log
	rm -rf $(BUILD_DIR)
