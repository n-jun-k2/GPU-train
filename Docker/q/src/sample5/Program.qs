namespace sample5 {

    open Microsoft.Quantum.Diagnostics as Diag;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Arrays as Arrays;
    open Microsoft.Quantum.Math as Math;

    function DegToRad(deg: Double) : Double {
        return deg * Math.PI() / 180.0;
    }


    @EntryPoint()
    operation HelloQ() : Unit {
        Message("Hello quantum world!");

        use register = Qubit[2];

        let angle = DegToRad(36.0);
        let (ctrl, target) = Arrays.MostAndTail(register);

        ApplyToEachCA(H, register);

        // Controlled R1(register[0..0], (angle, register[1]));
        // |0⟩      0.500000 +  0.000000 i  ==     ******               [ 0.250000 ]     --- [  0.00000 rad ]
        // |1⟩      0.500000 +  0.000000 i  ==     ******               [ 0.250000 ]     --- [  0.00000 rad ]
        // |2⟩      0.500000 +  0.000000 i  ==     ******               [ 0.250000 ]     --- [  0.00000 rad ]
        // |3⟩      0.404508 +  0.293893 i  ==     ******               [ 0.250000 ]      /  [  0.62832 rad ]

        // Controlled R1(register[1..1], (angle, register[0]));
        // |0⟩      0.500000 +  0.000000 i  ==     ******               [ 0.250000 ]     --- [  0.00000 rad ]
        // |1⟩      0.500000 +  0.000000 i  ==     ******               [ 0.250000 ]     --- [  0.00000 rad ]
        // |2⟩      0.500000 +  0.000000 i  ==     ******               [ 0.250000 ]     --- [  0.00000 rad ]
        // |3⟩      0.404508 +  0.293893 i  ==     ******               [ 0.250000 ]      /  [  0.62832 rad ]

        ApplyControlledOnInt(1, R1(angle, _), ctrl, target);
        // |0⟩      0.500000 +  0.000000 i  ==     ******               [ 0.250000 ]     --- [  0.00000 rad ]
        // |1⟩      0.500000 +  0.000000 i  ==     ******               [ 0.250000 ]     --- [  0.00000 rad ]
        // |2⟩      0.500000 +  0.000000 i  ==     ******               [ 0.250000 ]     --- [  0.00000 rad ]
        // |3⟩      0.404508 +  0.293893 i  ==     ******               [ 0.250000 ]      /  [  0.62832 rad ]

        Diag.DumpMachine();

        ResetAll(register);
    }
}

