* Folded Cascode OTA (NMOS differential input pair, single-ended output)
* Topology: NMOS diff pair (M1,M2) + PMOS current source load (M3,M4) + NMOS current mirror output (M5,M6)
* Parameters: {W1} {L1} (input pair), {W3} {L3} (PMOS CS), {W5} {L5} (NMOS mirror+tail), {Ib} (bias)

.title folded_cascode_ota

** Supply **
Vdd vdd 0 1.8
Vss vss 0 0

** Input stimulus for AC analysis **
Vinp inp 0 DC 0.9 AC 1
Vinn inn 0 DC 0.9

** NMOS bias: Ibias through diode-connected M_bn sets Vbias_n **
Ibias vdd nbias_n {Ib}
M_bn nbias_n nbias_n vss vss NMOS W={W5} L={L5}

** PMOS bias: Ib_p through diode-connected M_bp sets Vbias_p **
Ib_p nbias_p vss {Ib}
M_bp nbias_p nbias_p vdd vdd PMOS W={W3} L={L3}

** NMOS tail current source (mirrors M_bn, sets differential pair tail current) **
M_tail ntail nbias_n vss vss NMOS W={W5} L={L5}

** NMOS differential input pair **
M1 n1 inp ntail vss NMOS W={W1} L={L1}
M2 out inn ntail vss NMOS W={W1} L={L1}

** PMOS current sources: fold current from vdd into cascode nodes **
M3 n1 nbias_p vdd vdd PMOS W={W3} L={L3}
M4 out nbias_p vdd vdd PMOS W={W3} L={L3}

** NMOS current mirror: M5 reference (diode-connected), M6 output **
M5 n1 n1 vss vss NMOS W={W5} L={L5}
M6 out n1 vss vss NMOS W={W5} L={L5}

** Load capacitor **
CL out 0 10p

** DC operating point bias: weak pull to mid-supply for NGSpice convergence **
Vbias_ref vbias_ref 0 0.9
Rbias out vbias_ref 100Meg

** Transistor models (Level 1 SPICE) **
.model NMOS NMOS (LEVEL=1 VTO=0.5 KP=200u LAMBDA=0.04)
.model PMOS PMOS (LEVEL=1 VTO=-0.5 KP=100u LAMBDA=0.05)

** Analysis **
.control
op
let pwr = -i(Vdd) * 1.8
echo MEAS_power = $&pwr

ac dec 100 1 1G

* DC gain: measure at lowest frequency point (index 0 ~ 1 Hz)
let gdb = vdb(out)[0]
echo MEAS_gain_db = $&gdb

* UGBW and phase margin: only valid if gain crosses 0 dB
meas ac ugbw_val WHEN vdb(out)=0
if ( ugbw_val > 0 )
  meas ac ph_at_ugbw FIND vp(out) WHEN vdb(out)=0
  let pm_v = 180 + ph_at_ugbw
  echo MEAS_ugbw = $&ugbw_val
  echo MEAS_phase_margin = $&pm_v
else
  echo MEAS_ugbw = 0
  echo MEAS_phase_margin = 0
end
.endc

.end
