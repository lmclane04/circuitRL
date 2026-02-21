* Two-Stage CMOS Op-Amp — Parameterized Netlist
* Parameters substituted at runtime: {W1}, {L1}, {W3}, {L3}, {W5}, {L5}, {W7}, {L7}, {Cc}, {Ib}

.title two_stage_opamp

** Supply **
Vdd vdd 0 1.8
Vss vss 0 0

** Input stimulus for AC analysis **
Vinp inp 0 DC 0.9 AC 1
Vinn inn 0 DC 0.9

** Bias current source: flows from vdd into diode-connected M6 **
Ibias vdd nbias {Ib}

** First stage: differential pair (M1, M2) with PMOS loads (M3, M4) **
* NMOS input pair
M1 net1 inp  ntail vss NMOS W={W1} L={L1}
M2 net2 inn  ntail vss NMOS W={W1} L={L1}

* PMOS active load (current mirror)
M3 net1 net1 vdd vdd PMOS W={W3} L={L3}
M4 net2 net1 vdd vdd PMOS W={W3} L={L3}

* NMOS tail current source (mirrors M6)
M5 ntail nbias vss vss NMOS W={W5} L={L5}

** Bias: diode-connected NMOS sets mirror voltage **
M6 nbias nbias vss vss NMOS W={W5} L={L5}

** Second stage: common-source with PMOS load **
M7 out net2 vss vss NMOS W={W7} L={L7}
M8 out net3 vdd vdd PMOS W={W3} L={L3}

* PMOS bias: diode-connected PMOS driven by mirror of Ibias
M9  net3 net3 vdd vdd PMOS W={W3} L={L3}
M10 net3 nbias vss vss NMOS W={W5} L={L5}

** Miller compensation capacitor **
Cc net2 out {Cc}

** Load **
CL out 0 1p

** Transistor models (Level 1 for proof-of-concept) **
.model NMOS NMOS (LEVEL=1 VTO=0.5 KP=200u LAMBDA=0.04)
.model PMOS PMOS (LEVEL=1 VTO=-0.5 KP=100u LAMBDA=0.05)

** Analysis — all inside .control for reliable batch-mode output **
.control
op
let pwr = -i(Vdd) * 1.8
echo MEAS_power = $&pwr

ac dec 100 1 1G

* Gain: measure at low frequency (index 0 = 1 Hz)
let gdb = vdb(out)[0]
echo MEAS_gain_db = $&gdb

* UGBW and phase margin: only meaningful if gain > 0 dB
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
