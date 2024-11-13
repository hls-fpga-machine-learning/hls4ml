DESIGN := design_1

help:
	@echo "INFO: make <TAB> to show targets"
.PHONY: help

--setup:
	xsct ./setup.tcl $(DESIGN)
.PHONY: --setup

sdk: --setup
	rm -f $(DESIGN)_standalone/src/helloworld.c
	cd  $(DESIGN)_standalone/src && ln -s ../../common/main.c main.c
	cd  $(DESIGN)_standalone/src && ln -s ../../common/data.h data.h
.PHONY: sdk

gui:
	xsdk --workspace . &
.PHONY: gui

clean:
	rm -rf $(DESIGN)_platform
	rm -rf $(DESIGN)_standalone
	rm -rf $(DESIGN)_standalone_bsp
	rm -rf RemoteSystemsTempFiles
	rm -rf .Xil
	rm -rf .metadata
	rm -f *.log
.PHONY: clean

ultraclean: clean
	rm -rf hdf/*.hdf
.PHONY: ultraclean
