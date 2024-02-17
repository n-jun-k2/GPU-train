namespace sample6 {

    open Microsoft.Quantum.Diagnostics as Diag;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Math as Math;
    open Microsoft.Quantum.Measurement as Meas;
    open Common as Custom;

    operation PrepareBiasedCoin(probebility: Double, qubit: Qubit): Unit is Ctl+Adj{
        let angle = 2.0 * Math.ArcCos(Math.Sqrt(1.0 - probebility));
        R(PauliY, angle, qubit);
    }

    operation GetNextRandomBit(preparationOperation: (Qubit => Unit)): Result {
        use q = Qubit();
        preparationOperation(q);
        Message("_/_/_/_/_/_/");
        Diag.DumpMachine();
        Message("_/_/_/_/_/_/");
        return Meas.MResetZ(q);
    }

    operation PlayMorganasGame(probability: Double) : Unit {
        mutable numRounds = 0;
        mutable done = false;

        let prep = PrepareBiasedCoin(probability, _);
        repeat {
            set numRounds += 1;
            set done = GetNextRandomBit(prep) == Zero;
        }
        until done;

        Message($"{numRounds}");
    }

    @EntryPoint()
    operation HelloQ() : Unit {
        Message("Hello quantum world!");

        Message($"({Custom.DegToRad(90.0)})");

        PlayMorganasGame(0.4);
    }
}

