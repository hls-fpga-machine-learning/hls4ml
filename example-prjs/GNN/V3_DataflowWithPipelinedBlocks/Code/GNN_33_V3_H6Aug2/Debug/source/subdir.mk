################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../myproject.cpp 

OBJS += \
./source/myproject.o 

CPP_DEPS += \
./source/myproject.d 


# Each subdirectory must supply rules for building sources it contributes
source/myproject.o: C:/Users/The-Machine/AppData/Roaming/Xilinx/Vivado/GNN_33_V3_H6Aug2/myproject.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -DAESL_TB -D__llvm__ -D__llvm__ -IC:/Xilinx/Vivado/2019.1/include -IC:/Xilinx/Vivado/2019.1/include/ap_sysc -IC:/Xilinx/Vivado/2019.1/win64/tools/systemc/include -IC:/Users/The-Machine/AppData/Roaming/Xilinx/Vivado -IC:/Xilinx/Vivado/2019.1/include/etc -IC:/Xilinx/Vivado/2019.1/win64/tools/auto_cc/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


