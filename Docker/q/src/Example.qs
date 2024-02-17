namespace Example{

    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Diagnostics as Diag;
    open Microsoft.Quantum.Measurement as Meas;

    @EntryPoint()
    operation Main(): Unit {
        Message("TEST");
        use q = Qubit(); 

        H(q);
        Diag.DumpMachine();
        let d = Meas.MResetX(q);

        Message($"Result : {d}");
    }
}