* Three-Stage TIA (current input -> voltage output)
* AC current source of 1 A is used so vdb(out) directly represents transimpedance in dB-ohm.
* Parameters: W1, L1, RD1, W2, L2, RD2, W3, L3, RD3, Rin

.title three_stage_tia

Vdd vdd 0 1.8

* 1 A AC input current (photodiode-equivalent small-signal stimulus)
Iin in 0 DC 0 AC 1

* Input shunt sets input impedance / transimpedance base term
Rin in 0 {Rin}

* Stage 1 bias and AC coupling from input node
Cc_in in gate1 100n
Vbias1 vbias1 0 DC 0.7
Rbias1 vbias1 gate1 100k

* Stage 1 common-source
M1 out1 gate1 0 0 NMOS W={W1} L={L1}
RD1 vdd out1 {RD1}
CL1 out1 0 0.1p

* Stage 2 bias and AC coupling
Cc12 out1 gate2 100n
Vbias2 vbias2 0 DC 0.7
Rbias2 vbias2 gate2 100k

* Stage 2 common-source
M2 out2 gate2 0 0 NMOS W={W2} L={L2}
RD2 vdd out2 {RD2}
CL2 out2 0 0.1p

* Stage 3 bias and AC coupling
Cc23 out2 gate3 100n
Vbias3 vbias3 0 DC 0.7
Rbias3 vbias3 gate3 100k

* Stage 3 common-source
M3 out3 gate3 0 0 NMOS W={W3} L={L3}
RD3 vdd out3 {RD3}

* Output load
CL out3 0 0.5p

.model NMOS NMOS (LEVEL=1 VTO=0.5 KP=200u LAMBDA=0.04)

.control
op
let pwr = -i(Vdd) * 1.8
echo MEAS_power = $&pwr

ac dec 100 100 10G

* Transimpedance gain in dB-ohm (because Iin AC magnitude is 1 A)
let zt_db = vdb(out3)[0]
echo MEAS_transimpedance_db = $&zt_db

* -3 dB bandwidth from transimpedance passband
let zt3db = zt_db - 3
meas ac bw_val WHEN vdb(out3)=zt3db
if ( bw_val > 0 )
  echo MEAS_bandwidth = $&bw_val
else
  echo MEAS_bandwidth = 0
end
.endc

.end
