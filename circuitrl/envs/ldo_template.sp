* Simple PMOS LDO with behavioral error amplifier
* Parameters: Wp, Lp (pass PMOS), Rtop/Rbot (feedback divider), Cc (error-amp comp), Cout
* Metrics:
*   vout_error: absolute regulation error at Vin=1.8V relative to 1.2V target
*   line_reg_mv_v: line regulation from Vin=1.6V to 2.0V (mV/V)
*   power: input power at nominal Vin=1.8V (includes load + regulator loss)

.title simple_ldo

.param VIN_NOM=1.8
.param VREF=0.8

* Input supply
Vin vin 0 DC {{VIN_NOM}}

* Error amplifier drives pass-device gate, centered near mid-supply
* Gain is intentionally moderate for robust convergence in this simplified setup.
Eerr gate 0 VALUE = {{0.9 + 10*(VREF - v(vfb))}}

* PMOS pass device (source at vin, drain at regulated output)
Mpass vout gate vin vin PMOS W={Wp} L={Lp}

* Feedback divider
Rtop vout vfb {Rtop}
Rbot vfb 0 {Rbot}

* Compensation and output capacitor
Cc gate vfb {Cc}
Cout vout 0 {Cout}

* Resistive load (~12 mA at 1.2V)
Rload vout 0 100

.model PMOS PMOS (LEVEL=1 VTO=-0.5 KP=100u LAMBDA=0.05)

.control
op

* Nominal-point output error wrt 1.2V target
let verr = abs(v(vout) - 1.2)
echo MEAS_vout_error = $&verr

* Nominal-point input power
let pwr = -i(Vin) * 1.8
echo MEAS_power = $&pwr

* Line regulation across input sweep
save v(vout)
dc Vin 1.6 2.0 0.02
meas dc vout_lo FIND v(vout) AT=1.6
meas dc vout_hi FIND v(vout) AT=2.0
let line_reg = abs(vout_hi - vout_lo) / (2.0 - 1.6) * 1e3
echo MEAS_line_reg_mv_v = $&line_reg
.endc

.end
