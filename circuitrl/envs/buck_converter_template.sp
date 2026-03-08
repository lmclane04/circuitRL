* Buck DC-DC Converter (Step-Down)
* Fixed: Vin=5V, D=0.66 (ton=0.66us), fsw=1MHz (Tperiod=1us), Rload=5 ohm
* Parameters: L_ind (output filter inductor in H), C_out (output filter cap in F)
* ICs set near steady state to skip the LC settling transient.
* Expected steady-state: Vout ~= D*Vin = 3.3V, IL_avg = Vout/Rload = 0.66A

.title buck_converter

* DC input supply
Vin vin 0 DC 5.0

* PWM switch control: 0V=off, 5V=on; ton=0.66us, period=1us
Vpwm pwm 0 PULSE(0 5 0 1n 1n 0.66u 1u)

* Ideal voltage-controlled switch (Ron=0.01 ohm, Roff=1GOhm)
* ON when V(pwm) > 2.5V, OFF when V(pwm) < 2.5V
Sw vin sw_node pwm 0 SW
.model SW SW(Ron=0.01 Roff=1e9 Vt=2.5 Vh=0.1)

* Freewheeling diode: anode=gnd, cathode=sw_node
* When switch opens, inductor drives sw_node negative, forward-biasing D1
* Current path: 0 -> D1 -> sw_node -> L_ind -> vout -> Rload -> 0
D1 0 sw_node DFREE
.model DFREE D(Is=1e-15 N=1 Rs=0.001)

* Small snubber cap at sw_node to aid convergence during switching transitions
Csnub sw_node 0 100p

* LC output filter
* IC=0.66 sets initial inductor current to expected steady-state average
L_ind sw_node vout {L_ind} IC=0.66

* IC=3.3 sets initial output voltage to expected steady-state (D*Vin=3.3V)
C_out vout 0 {C_out} IC=3.3

* Resistive load
Rload vout 0 5

.control
* UIC: use element IC= values instead of computing DC operating point
* Start near steady state; 1ns step resolves 1us switching transitions cleanly
tran 1n 20u UIC

* Measure over last 10us (10 switching cycles) to capture steady-state ripple
meas tran vout_max MAX v(vout) from=10u to=20u
meas tran vout_min MIN v(vout) from=10u to=20u
let ripple_v = vout_max - vout_min
let ripple_mv = ripple_v * 1000
echo MEAS_ripple_mv = $&ripple_mv

* Inductor current ripple (peak-to-peak, proportional to L choice)
meas tran il_max MAX i(L_ind) from=10u to=20u
meas tran il_min MIN i(L_ind) from=10u to=20u
let il_ripple = il_max - il_min
echo MEAS_il_ripple_a = $&il_ripple

.endc

.end
