namespace sample7 {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Oracles;
    open Common as Cmn;
    open Microsoft.Quantum.Diagnostics as Diagnostics;
    open Microsoft.Quantum.Measurement as Measurement;

    operation CheckIfOracleIsBalanced(oracle: (Qubit, Qubit) => Unit): Bool {
        use (control, target) = (Qubit(), Qubit());
        let preparTarget = BoundCA([X, H]);

        H(control);
        within {
            preparTarget(target);
        } apply {
            oracle(control, target);
        }
        return Measurement.MResetX(control) == One;
    }

    @EntryPoint()
    operation HelloQ() : Unit {
        Message("Hello quantum world!");

        let result = CheckIfOracleIsBalanced(ApplyOneOracle);
        Message($"{result}");
    }
}

