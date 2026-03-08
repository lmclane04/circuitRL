* Two-Stage Cascaded Common-Source Amplifier (AC-Coupled)
* Stage 1: NMOS M1 with resistive load RD1
* Stage 2: NMOS M2 with resistive load RD2, independently biased at 0.7V via Rbias2
* AC coupling between stages via Ccoup (100nF), coupling corner ~16 Hz
* Parameters: W1, L1, RD1 (stage 1), W2, L2, RD2 (stage 2)

.title cascaded_cs_amp

Vdd vdd 0 1.8

* Stage 1: NMOS common-source with input bias + AC signal
Vin gate1 0 DC 0.7 AC 1
M1 out1 gate1 0 0 NMOS W={W1} L={L1}
RD1 vdd out1 {RD1}

* Interstage parasitic load cap (models wiring + gate capacitance)
CL1 out1 0 0.1p

* AC coupling: large cap passes AC signal, blocks DC level shift
Ccoup out1 gate2 100n

* Stage 2 independent DC bias: 100k series resistor + voltage source
* High impedance so AC signal from out1 passes freely through Ccoup
Vbias2 vbias2_node 0 DC 0.7
Rbias2 vbias2_node gate2 100k

* Stage 2: NMOS common-source
M2 out2 gate2 0 0 NMOS W={W2} L={L2}
RD2 vdd out2 {RD2}

* Output load capacitor
CL out2 0 0.5p

.model NMOS NMOS (LEVEL=1 VTO=0.5 KP=200u LAMBDA=0.04)

.control
op
* DC power: both drain currents drawn from Vdd
let pwr = -i(Vdd) * 1.8
echo MEAS_power = $&pwr

* AC sweep from 100 Hz (well above 16 Hz coupling corner) to 10 GHz
ac dec 100 100 10G

* Midband gain at 100 Hz (first sweep point = passband)
let gdb = vdb(out2)[0]
echo MEAS_gain_db = $&gdb

* Upper -3dB bandwidth: first crossing going from passband toward high freq
let gain3db = gdb - 3
meas ac bw_val WHEN vdb(out2)=gain3db
if ( bw_val > 0 )
  echo MEAS_bandwidth = $&bw_val
else
  echo MEAS_bandwidth = 0
end
.endc

.end
