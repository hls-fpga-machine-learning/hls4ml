#################
#    HLS4ML
#################
array set opt {
    reset      0
    csim       1
    synth      1
    cosim      1
    validation 1
    export     0
    vsynth     0
    fifo_opt   0
}

set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

proc remove_recursive_log_wave {} {
    set tcldir [file dirname [info script]]
    source [file join $tcldir project.tcl]

    set filename ${project_name}_prj/solution1/sim/verilog/${project_name}.tcl
    set timestamp [clock format [clock seconds] -format {%Y%m%d%H%M%S}]
    set temp     $filename.new.$timestamp
    # set backup   $filename.bak.$timestamp

    set in  [open $filename r]
    set out [open $temp     w]

    # line-by-line, read the original file
    while {[gets $in line] != -1} {
        if {[string equal "$line" "log_wave -r /"]} {
            set line { }
        }
        puts $out $line
    }

    close $in
    close $out

    # move the new data to the proper filename
    file delete -force $filename
    file rename -force $temp $filename
}

proc add_vcd_instructions_tcl {} {
    set tcldir [file dirname [info script]]
    source [file join $tcldir project.tcl]

    set filename ${project_name}_prj/solution1/sim/verilog/${project_name}.tcl
    set timestamp [clock format [clock seconds] -format {%Y%m%d%H%M%S}]
    set temp     $filename.new.$timestamp
    # set backup   $filename.bak.$timestamp

    set in  [open $filename r]
    set out [open $temp     w]

    # line-by-line, read the original file
    while {[gets $in line] != -1} {
        if {[string equal "$line" "log_wave -r /"]} {
            set line {source "../../../../project.tcl"
                if {[string equal "$backend" "vivadoaccelerator"]} {
                    current_scope [get_scopes -regex "/apatb_${project_name}_axi_top/AESL_inst_${project_name}_axi/${project_name}_U0.*"]
                    set scopes [get_scopes -regexp {layer(\d*)_.*data_0_V_U.*}]
                    append scopes { }
                    current_scope "/apatb_${project_name}_axi_top/AESL_inst_${project_name}_axi"
                    append scopes [get_scopes -regexp {(in_local_V_data.*_0_.*)}]
                    append scopes { }
                    append scopes [get_scopes -regexp {(out_local_V_data.*_0_.*)}]
                } else {
                    current_scope [get_scopes -regex "/apatb_${project_name}_top/AESL_inst_${project_name}"]
                    set scopes [get_scopes -regexp {layer(\d*)_.*data_0_V_U.*}]
                }
                open_vcd fifo_opt.vcd
                foreach scope $scopes {
                    current_scope $scope
                    if {[catch [get_objects usedw]] == 0} {
                        puts "$scope skipped"
                        continue
                    }
                    set usedw [get_objects usedw]
                    set depth [get_objects DEPTH]
                    add_wave $usedw
                    log_vcd $usedw
                    log_wave $usedw
                    add_wave $depth
                    log_vcd $depth
                    log_wave $depth
                }
            }
        }

        if {[string equal "$line" "quit"]} {
            set line {flush_vcd
                close_vcd
                quit
            }
        }
        # then write the transformed line
        puts $out $line
    }

    close $in
    close $out

    # move the new data to the proper filename
    file delete -force $filename
    file rename -force $temp $filename
}

foreach arg $::argv {
    foreach o [lsort [array names opt]] {
        regexp "$o=+(\\w+)" $arg unused opt($o)
    }
}

proc report_time { op_name time_start time_end } {
    set time_taken [expr $time_end - $time_start]
    set time_s [expr ($time_taken / 1000) % 60]
    set time_m [expr ($time_taken / (1000*60)) % 60]
    set time_h [expr ($time_taken / (1000*60*60)) % 24]
    puts "***** ${op_name} COMPLETED IN ${time_h}h${time_m}m${time_s}s *****"
}

# Compare file content: 1 = same, 0 = different
proc compare_files {file_1 file_2} {
    # Check if files exist, error otherwise
    if {! ([file exists $file_1] && [file exists $file_2])} {
        return 0
    }
    # Files with different sizes are obviously different
    if {[file size $file_1] != [file size $file_2]} {
        return 0
    }

    # String compare the content of the files
    set fh_1 [open $file_1 r]
    set fh_2 [open $file_2 r]
    set equal [string equal [read $fh_1] [read $fh_2]]
    close $fh_1
    close $fh_2
    return $equal
}

file mkdir tb_data
set CSIM_RESULTS "./tb_data/csim_results.log"
set RTL_COSIM_RESULTS "./tb_data/rtl_cosim_results.log"

if {$opt(reset)} {
    open_project -reset ${project_name}_prj
} else {
    open_project ${project_name}_prj
}
set_top ${project_name}
add_files firmware/${project_name}.cpp -cflags "-std=c++0x"
add_files -tb ${project_name}_test.cpp -cflags "-std=c++0x"
add_files -tb firmware/weights
add_files -tb tb_data
if {$opt(reset)} {
    open_solution -reset "solution1"
} else {
    open_solution "solution1"
}
catch {config_array_partition -maximum_size $maximum_size}
config_compile -name_max_length 80
set_part $part
config_schedule -enable_dsp_full_reg=false
create_clock -period $clock_period -name default
set_clock_uncertainty $clock_uncertainty default


if {$opt(csim)} {
    puts "***** C SIMULATION *****"
    set time_start [clock clicks -milliseconds]
    csim_design
    set time_end [clock clicks -milliseconds]
    report_time "C SIMULATION" $time_start $time_end
}

if {$opt(synth)} {
    puts "***** C/RTL SYNTHESIS *****"

    set time_start [clock clicks -milliseconds]
    csynth_design
    set time_end [clock clicks -milliseconds]
    report_time "C/RTL SYNTHESIS" $time_start $time_end
}

if {$opt(cosim)} {
    puts "***** C/RTL SIMULATION *****"
    # TODO: This is a workaround (Xilinx defines __RTL_SIMULATION__ only for SystemC testbenches).
    add_files -tb ${project_name}_test.cpp -cflags "-std=c++0x -DRTL_SIM"
    set time_start [clock clicks -milliseconds]

    cosim_design -trace_level all -setup

    if {$opt(fifo_opt)} {
        puts "\[hls4ml\] - FIFO optimization started"

        if {[string equal "$backend" "vivado"] || [string equal $backend "vivadoaccelerator"]} {
            add_vcd_instructions_tcl
        }
    }

    remove_recursive_log_wave
    set old_pwd [pwd]
    cd ${project_name}_prj/solution1/sim/verilog/
    source run_sim.tcl
    cd $old_pwd

    set time_end [clock clicks -milliseconds]
    puts "INFO:"
    if {[string equal "$backend" "vivadoaccelerator"]} {
        puts [read [open ${project_name}_prj/solution1/sim/report/${project_name}_axi_cosim.rpt r]]
    } else {
        puts [read [open ${project_name}_prj/solution1/sim/report/${project_name}_cosim.rpt r]]
    }
    report_time "C/RTL SIMULATION" $time_start $time_end
}

if {$opt(validation)} {
    puts "***** C/RTL VALIDATION *****"
    if {[compare_files $CSIM_RESULTS $RTL_COSIM_RESULTS]} {
        puts "INFO: Test PASSED"
    } else {
        puts "ERROR: Test failed"
        puts "ERROR: - csim log:      $CSIM_RESULTS"
        puts "ERROR: - RTL-cosim log: $RTL_COSIM_RESULTS"
        exit 1
    }
}

if {$opt(export)} {
    puts "***** EXPORT IP *****"
    set time_start [clock clicks -milliseconds]
    export_design -format ip_catalog -version $version
    set time_end [clock clicks -milliseconds]
    report_time "EXPORT IP" $time_start $time_end
}

if {$opt(vsynth)} {
    puts "***** VIVADO SYNTHESIS *****"
    if {[file exist ${project_name}_prj/solution1/syn/verilog]} {
        set time_start [clock clicks -milliseconds]
        exec vivado -mode batch -source vivado_synth.tcl >@ stdout
        set time_end [clock clicks -milliseconds]
        report_time "VIVADO SYNTHESIS" $time_start $time_end
    } else {
        puts "ERROR: Cannot find generated Verilog files. Did you run C synthesis?"
        exit 1
    }
}

exit
