* Common-Source Amplifier with Resistive Load
* Simple circuit: NMOS CS stage with resistor load
* Parameters: W1, L1 (NMOS), RD (drain resistor)

.title cs_amplifier

Vdd vdd 0 1.8
Vss vss 0 0

* Input: DC bias + AC small signal
Vin gate 0 DC 0.7 AC 1

* NMOS common-source transistor
M1 out gate vss vss NMOS W={W1} L={L1}

* Resistive load
RD vdd out {RD}

* Output load capacitor
CL out 0 0.5p

.model NMOS NMOS (LEVEL=1 VTO=0.5 KP=200u LAMBDA=0.04)

.control
op
let pwr = -i(Vdd) * 1.8
echo MEAS_power = $&pwr

ac dec 100 1 10G
let gdb = vdb(out)[0]
echo MEAS_gain_db = $&gdb

* Find 3dB bandwidth
let gain3db = gdb - 3
meas ac bw_val WHEN vdb(out)=gain3db
if ( bw_val > 0 )
  echo MEAS_bandwidth = $&bw_val
else
  echo MEAS_bandwidth = 0
end
.endc

.end
