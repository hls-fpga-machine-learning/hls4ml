//Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2019.2 (lin64) Build 2708876 Wed Nov  6 21:39:14 MST 2019
//Date        : Wed Oct  5 16:49:36 2022
//Host        : subnugler running 64-bit Ubuntu 22.04.1 LTS
//Command     : generate_target design_1_wrapper.bd
//Design      : design_1_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module design_1_wrapper
   (ddr4_sdram_act_n,
    ddr4_sdram_adr,
    ddr4_sdram_ba,
    ddr4_sdram_bg,
    ddr4_sdram_ck_c,
    ddr4_sdram_ck_t,
    ddr4_sdram_cke,
    ddr4_sdram_cs_n,
    ddr4_sdram_dm_n,
    ddr4_sdram_dq,
    ddr4_sdram_dqs_c,
    ddr4_sdram_dqs_t,
    ddr4_sdram_odt,
    ddr4_sdram_reset_n,
    default_100mhz_clk_clk_n,
    default_100mhz_clk_clk_p,
    dummy_port_in,
    rs232_uart_0_rxd,
    rs232_uart_0_txd);
  output ddr4_sdram_act_n;
  output [16:0]ddr4_sdram_adr;
  output [1:0]ddr4_sdram_ba;
  output ddr4_sdram_bg;
  output ddr4_sdram_ck_c;
  output ddr4_sdram_ck_t;
  output ddr4_sdram_cke;
  output [1:0]ddr4_sdram_cs_n;
  inout [8:0]ddr4_sdram_dm_n;
  inout [71:0]ddr4_sdram_dq;
  inout [8:0]ddr4_sdram_dqs_c;
  inout [8:0]ddr4_sdram_dqs_t;
  output ddr4_sdram_odt;
  output ddr4_sdram_reset_n;
  input default_100mhz_clk_clk_n;
  input default_100mhz_clk_clk_p;
  input dummy_port_in;
  input rs232_uart_0_rxd;
  output rs232_uart_0_txd;

  wire ddr4_sdram_act_n;
  wire [16:0]ddr4_sdram_adr;
  wire [1:0]ddr4_sdram_ba;
  wire ddr4_sdram_bg;
  wire ddr4_sdram_ck_c;
  wire ddr4_sdram_ck_t;
  wire ddr4_sdram_cke;
  wire [1:0]ddr4_sdram_cs_n;
  wire [8:0]ddr4_sdram_dm_n;
  wire [71:0]ddr4_sdram_dq;
  wire [8:0]ddr4_sdram_dqs_c;
  wire [8:0]ddr4_sdram_dqs_t;
  wire ddr4_sdram_odt;
  wire ddr4_sdram_reset_n;
  wire default_100mhz_clk_clk_n;
  wire default_100mhz_clk_clk_p;
  wire dummy_port_in;
  wire rs232_uart_0_rxd;
  wire rs232_uart_0_txd;

  design_1 design_1_i
       (.ddr4_sdram_act_n(ddr4_sdram_act_n),
        .ddr4_sdram_adr(ddr4_sdram_adr),
        .ddr4_sdram_ba(ddr4_sdram_ba),
        .ddr4_sdram_bg(ddr4_sdram_bg),
        .ddr4_sdram_ck_c(ddr4_sdram_ck_c),
        .ddr4_sdram_ck_t(ddr4_sdram_ck_t),
        .ddr4_sdram_cke(ddr4_sdram_cke),
        .ddr4_sdram_cs_n(ddr4_sdram_cs_n),
        .ddr4_sdram_dm_n(ddr4_sdram_dm_n),
        .ddr4_sdram_dq(ddr4_sdram_dq),
        .ddr4_sdram_dqs_c(ddr4_sdram_dqs_c),
        .ddr4_sdram_dqs_t(ddr4_sdram_dqs_t),
        .ddr4_sdram_odt(ddr4_sdram_odt),
        .ddr4_sdram_reset_n(ddr4_sdram_reset_n),
        .default_100mhz_clk_clk_n(default_100mhz_clk_clk_n),
        .default_100mhz_clk_clk_p(default_100mhz_clk_clk_p),
        .dummy_port_in(dummy_port_in),
        .rs232_uart_0_rxd(rs232_uart_0_rxd),
        .rs232_uart_0_txd(rs232_uart_0_txd));
endmodule
