* Two-Stage CMOS Op-Amp — Parameterized Netlist
* Parameters substituted at runtime: {W1}, {L1}, {W3}, {L3}, {W5}, {L5}, {W7}, {L7}, {Cc}, {Ib}

.title two_stage_opamp

** Supply **
Vdd vdd 0 1.8
Vss vss 0 0

** Input stimulus for AC analysis **
Vinp inp 0 DC 0.9 AC 1
Vinn inn 0 DC 0.9

** Bias current source **
Ibias nbias 0 {Ib}

** First stage: differential pair (M1, M2) with PMOS loads (M3, M4) **
* NMOS input pair
M1 net1 inp  ntail vss NMOS W={W1} L={L1}
M2 net2 inn  ntail vss NMOS W={W1} L={L1}

* PMOS active load
M3 net1 net1 vdd vdd PMOS W={W3} L={L3}
M4 net2 net1 vdd vdd PMOS W={W3} L={L3}

* Tail current source
M5 ntail nbias vss vss NMOS W={W5} L={L5}

** Bias mirror **
M6 nbias nbias vss vss NMOS W={W5} L={L5}

** Second stage: common-source amplifier **
M7 out  net2 vss vss NMOS W={W7} L={L7}
M8 out  pbias vdd vdd PMOS W={W3} L={L3}

* PMOS bias for second stage
M9  pbias pbias vdd vdd PMOS W={W3} L={L3}
M10 pbias nbias vss vss NMOS W={W5} L={L5}

** Compensation capacitor (Miller) **
Cc net2 out {Cc}

** Load **
CL out 0 1p
RL out 0 10k

** Transistor models (simple Level 1 for proof-of-concept) **
.model NMOS NMOS (LEVEL=1 VTO=0.5 KP=200u LAMBDA=0.04)
.model PMOS PMOS (LEVEL=1 VTO=-0.5 KP=100u LAMBDA=0.05)

** AC analysis **
.control
ac dec 100 1 1G
.endc

** Measurements **
.measure AC gain_db MAX vdb(out)
.measure AC ugbw WHEN vdb(out)=0
.measure AC phase_at_ugbw FIND vp(out) WHEN vdb(out)=0
.measure AC pm PARAM='180+phase_at_ugbw'

** DC power measurement **
.measure DC power PARAM='-I(Vdd)*1.8'

.end
