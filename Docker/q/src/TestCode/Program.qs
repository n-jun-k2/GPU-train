namespace TestCode {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Logical; // EqualB
    open Microsoft.Quantum.Convert; // ResultArrayAsBoolArray 
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Diagnostics; // DumpMachine
    open Microsoft.Quantum.Measurement; // for MultiM
    open Microsoft.Quantum.Arithmetic; // LittleEndian
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Arrays; // IndexRange

    newtype LambdaOperationAction = (Apply: (Qubit) => Unit);
    newtype OracleType = (action: (Qubit) => Unit is Adj + Ctl);

    function DegToRad(deg: Double) : Double {
        return deg * PI() / 180.0;
    }

    /// Not演算の半分の処理　量子もつれの状態は対応外
    operation RootOfNot(q: Qubit) : Unit is Adj + Ctl {
        H(q);
        R1(-90.0, q);
        H(q);
    }

    // operation Write(q: Qubit, value: Result) : Unit {
    //     let a = LambdaOperationAction(x => Message($"{x}"));
    //     a::Apply(q);
    // }

    operation GenerateRandom(size: Int): Bool[] {
        use qs = Qubit[size];
        ApplyToEach(H, qs);
        let results = MultiM(qs);
        ResetAll(qs);
        return ResultArrayAsBoolArray(results);
    }

    operation Spy(qubit: Qubit, is_active: Bool): Unit {
        if not is_active  {
            Message($"SPY no activate.");
            return ();
        }

        H(qubit);

        let result = M(qubit);
        Message($"SPY Read qubit: {result}");

        Reset(qubit);
        X(qubit);
        H(qubit);

        return ();
    }

    operation SpyHunter(is_spy_active: Bool) : Unit {
        // Alice Flow
        let rand_A = GenerateRandom(2);

        use alice_qubit = Qubit();
        use bob_qubit = Qubit();
        use connection_qubit = Qubit();

        // set Value
        if rand_A[0] {
            Message("alice set value");
            X(alice_qubit);
        }
        // apply Had
        if rand_A[1] {
            Message("alice apply had");
            H(alice_qubit);
        }

        SWAP(alice_qubit, connection_qubit);
        Message($"alice send!! {M(alice_qubit)}");

        Spy(connection_qubit, is_spy_active);

        // bob flow
        let recv_had = GenerateRandom(1);
        SWAP(connection_qubit, bob_qubit);

        // apply Had
        if recv_had[0] {
            Message("bob apply Had!!");
            H(bob_qubit);
        }

        // read value
        let recv_val = M(bob_qubit);
        Message($"bob recv {recv_val}");

        if rand_A[1] == recv_had[0] {
            if rand_A[0] != ResultAsBool(recv_val) {
                Message($"Caught a SPY !!");
            }
        }
        DumpMachine("dump.text");
    }

    operation MultiQubitPair1() : Unit {
        use qbyte = Qubit[3];
        // 度からラジアン
        let angle = 90.0 * PI() / 180.0;

        // 全ての量子ビットに対してHadamard演算
        ApplyToEach(H, qbyte);

        // 単一キュビット演算）PHASE演算
        R1(angle, qbyte[0]); 

        // PHASEの結果を表示
        DumpMachine();
        // |0⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |1⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |2⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |3⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |4⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |5⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |6⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |7⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]

        Message($"");
        
        // 単一キュビット演算）NOT演算
        X(qbyte[0]);

        // NOTの結果を表示
        DumpMachine();
        // |0⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |1⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |2⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |3⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |4⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |5⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |6⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |7⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]

        // 全ての量子ビット
        ResetAll(qbyte);
    }


    operation MultiQubitPair2() : Unit {
        use qbyte = Qubit[3];

        let angle = 90.0 * PI() / 180.0;

        ApplyToEach(H, qbyte);

        // PHASE
        R1(angle, qbyte[1]); 
        DumpMachine();
        // |0⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |1⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |2⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |3⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |4⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |5⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |6⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |7⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]

        Message($"");

        X(qbyte[1]);
        DumpMachine();
        // |0⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |1⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |2⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |3⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |4⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |5⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |6⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |7⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]

        ResetAll(qbyte);
    }

    operation MultiQubitPair3() : Unit {
        use qbyte = Qubit[3];
        let angle = 90.0 * PI() / 180.0;

        ApplyToEach(H, qbyte);
        R1(angle, qbyte[2]); // qbitの|1>に対してPhaseする
        DumpMachine();
        // |0⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |1⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |2⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |3⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |4⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |5⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |6⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |7⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]

        Message($"");
        X(qbyte[2]);
        DumpMachine();
        // |0⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |1⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |2⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |3⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |4⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |5⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |6⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |7⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]

        ResetAll(qbyte);
    }

    operation ControlledNot(): Unit {
        use qbyte = Qubit[2];

        // qbyte[0] = |1>
        X(qbyte[0]);

        CNOT(qbyte[0], qbyte[1]);

        let result = MultiM(qbyte);
        Message($"{result}");
        //[One,One,Zero]

        ResetAll(qbyte);
    }

    operation ControlledNot2(): Unit {
        use qbyte = Qubit[2];

        X(qbyte[0]);

        Controlled X([qbyte[0]], (qbyte[1]));

        let result = MultiM(qbyte);
        Message($"{result}");
        ResetAll(qbyte);
    }

    operation NOT(q: Qubit): Unit is Ctl {
        X(q);
    }

    operation ControlledNot3(): Unit {
        use qbyte = Qubit[2];

        X(qbyte[0]);

        Controlled NOT([qbyte[0]], (qbyte[1]));

        let result = MultiM(qbyte);
        Message($"{result}");
        ResetAll(qbyte);
    }

    operation BellPaire(): Unit {
        // a, b = Zero, Zero
        use (a, b) = (Qubit(), Qubit());

        H(a);
        CNOT(a, b);
        DumpMachine();

        let result = MultiM([a, b]);
        Message($"{result}");

        ResetAll([a, b]);
    }

    operation test(): Unit {
        use qbyte = Qubit[3];

        mutable target = qbyte[2];
        mutable controlledRegister = qbyte[0..1];
        ApplyToEach(H, controlledRegister);

        let oracle = OracleType((x) => X(x));

        let controlledOracle = ControlledOnInt(2, oracle::action);
        controlledOracle(controlledRegister, target);

        DumpMachine();

        let result = MultiM(qbyte);
        Message($"{result}");

        ResetAll(qbyte);
    }

    operation CPhase(): Unit {
        use qbyte = Qubit[3];

        ApplyToEach(H, qbyte);

        let control = qbyte[2..2];
        let target = qbyte[0];
        let angle = DegToRad(36.0);

        Message($"controlled {control}");
        Message($"target {target}");
        Controlled R1(control, (angle, target));
        // |0⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |1⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |2⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |3⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |4⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |5⟩      0.286031 +  0.207813 i  ==     ***                  [ 0.125000 ]      /  [  0.62832 rad ]
        // |6⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |7⟩      0.286031 +  0.207813 i  ==     ***                  [ 0.125000 ]      /  [  0.62832 rad ]

        DumpMachine();

        ResetAll(qbyte);
    }

    operation CPhase2(): Unit {
        use qbyte = Qubit[3];

        ApplyToEach(H, qbyte);

        let control = qbyte[1..1];
        let target = qbyte[0];
        let angle = DegToRad(36.0);

        Message($"controlled {control}");
        Message($"target {target}");
        Controlled R1(control, (angle, target));
        // |0⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |1⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |2⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |3⟩      0.286031 +  0.207813 i  ==     ***                  [ 0.125000 ]      /  [  0.62832 rad ]
        // |4⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |5⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |6⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |7⟩      0.286031 +  0.207813 i  ==     ***                  [ 0.125000 ]      /  [  0.62832 rad ]

        DumpMachine();

        ResetAll(qbyte);
    }

    operation PhaseKickBack(): Unit {
        let (angle1, angle2) = (DegToRad(45.0), DegToRad(90.0));
        use (register1, register2) = (Qubit[2], Qubit());
        ApplyToEach(H, register1);

        // |1>で初期化
        X(register2);

        Controlled R1(register1[0..0], (angle1, register2));
        Controlled R1(register1[1..1], (angle2, register2));

        DumpRegister((), register1);
        // |0⟩      0.500000 +  0.000000 i  ==     *****                [ 0.250000 ]     --- [  0.00000 rad ]
        // |1⟩      0.353553 +  0.353553 i  ==     *****                [ 0.250000 ]      /  [  0.78540 rad ]
        // |2⟩      0.000000 +  0.500000 i  ==     *****                [ 0.250000 ]    ↑    [  1.57080 rad ]
        // |3⟩     -0.353553 +  0.353553 i  ==     *****                [ 0.250000 ]  \      [  2.35619 rad ]
        Message("======");
        DumpRegister((), [register2]);
        // |0⟩      0.000000 +  0.000000 i  ==                          [ 0.000000 ]
        // |1⟩      1.000000 +  0.000000 i  ==     ******************** [ 1.000000 ]     --- [  0.00000 rad ]

        ResetAll(register1);
        Reset(register2);
    }

    operation Toffoli(): Unit {
        let angle = DegToRad(90.0);
        use register = Qubit[3];
        ApplyToEach(H, register);

        R1(angle, register[0]); 
        X(register[0]);
        DumpRegister((), register);
        // |0⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |1⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |2⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |3⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |4⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |5⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |6⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |7⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]

        Message("==");

        Adjoint X(register[0]);
        CNOT(register[1], register[0]);
        // Controlled X([register[1]], (register[0]));
        DumpRegister((), register);
        // |0⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |1⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |2⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |3⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |4⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |5⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |6⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |7⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]

        Message("==");

        // Toffoli gate (1かつ２が真である場合0を反転)
        Adjoint CNOT(register[1], register[0]);
        // CCNOT(register[1], register[2], register[0]);
        Controlled X([register[1], register[2]], (register[0]));
        DumpRegister((), register);
        // |0⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |1⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |2⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |3⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |4⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |5⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |6⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |7⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]

        ResetAll(register);
    }

    operation Exchange(): Unit {
        let angle = DegToRad(90.0);
        use register = Qubit[3];

        ApplyToEach(H, register);

        // initilize
        R1(angle, register[2]);
        DumpRegister((), register);
        // |0⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |1⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |2⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |3⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |4⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |5⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |6⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |7⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]

        Message("==");

        SWAP(register[2], register[0]);
        DumpRegister((), register);
        // |0⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |1⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |2⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |3⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |4⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |5⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]
        // |6⟩      0.353553 +  0.000000 i  ==     ***                  [ 0.125000 ]     --- [  0.00000 rad ]
        // |7⟩      0.000000 +  0.353553 i  ==     ***                  [ 0.125000 ]    ↑    [  1.57080 rad ]

        ResetAll(register);
    }

    operation SwapTest() : Unit {
        use (input1, input2, output) = (Qubit(), Qubit(), Qubit());

        ResetAll([input1, input2]);
        X(input1);

        // swap test [input1, input2]が類似していると
        H(output);
        Controlled SWAP([output], (input1, input2));
        H(output);

        X(output);
        let output_result = M(output);
        Message($"output_result = {output_result}");
        Message($"(input1, input2) = ({MultiM([input1, input2])})");

        ResetAll([input1, input2, output]);
    }

    operation CPhaseMain() : Unit {
        use (a, b) = (Qubit(), Qubit());

        ApplyToEach(H, [a, b]);
        DumpRegister((), [a, b]);

        Message("---");

        let theta = 90.0 / 2.0;
        let angle = DegToRad(theta);
        Controlled R1([a], (angle, b));
        Controlled R1([a], (angle, b));
        DumpRegister((), [a, b]);

        ResetAll([a, b]);
    }

    operation RandomMain() : Unit {
        use (a, b) = (Qubit(), Qubit());
        let theta = DegToRad(45.0);

        ApplyToEach(H, [a, b]);

        R1(theta, b);
        // DumpRegister((), [a, b]);
        // |0⟩      0.500000 +  0.000000 i  ==     *****                [ 0.250000 ]     --- [  0.00000 rad ]
        // |1⟩      0.500000 +  0.000000 i  ==     *****                [ 0.250000 ]     --- [  0.00000 rad ]
        // |2⟩      0.353553 +  0.353553 i  ==     ******               [ 0.250000 ]      /  [  0.78540 rad ]
        // |3⟩      0.353553 +  0.353553 i  ==     ******               [ 0.250000 ]      /  [  0.78540 rad ]
        H(b);
        // DumpRegister((), [a, b]);
        // |0⟩      0.603553 +  0.250000 i  ==     *********            [ 0.426777 ]      /- [  0.39270 rad ]
        // |1⟩      0.603553 +  0.250000 i  ==     *********            [ 0.426777 ]      /- [  0.39270 rad ]
        // |2⟩      0.103553 + -0.250000 i  ==     **                   [ 0.073223 ]     -\  [ -1.17810 rad ]
        // |3⟩      0.103553 + -0.250000 i  ==     **                   [ 0.073223 ]     -\  [ -1.17810 rad ]
        CNOT(a, b);
        DumpRegister((), [a, b]);
        // |0⟩      0.603553 +  0.250000 i  ==     *********            [ 0.426777 ]      /- [  0.39270 rad ]
        // |1⟩      0.103553 + -0.250000 i  ==     **                   [ 0.073223 ]     -\  [ -1.17810 rad ]
        // |2⟩      0.103553 + -0.250000 i  ==     **                   [ 0.073223 ]     -\  [ -1.17810 rad ]
        // |3⟩      0.603553 +  0.250000 i  ==     *********            [ 0.426777 ]      /- [  0.39270 rad ]

        let read_a = M(a);
        Message($"|0> = {read_a}");
        let read_b = M(b);
        Message($"|1> = {read_b}");
        ResetAll([a, b]);
    }

    operation QuantumTeleportMain() : Unit {
        let TARGET_SIZE = 3;

        use register = Qubit[TARGET_SIZE];

        let psi = 0;    //　送信したい量子ビット（送信元）
        let alpha = 1;  //　エンタングルした量子ビット
        let beta = 2;   //　エンタングルした量子ビット（送信先）

        // |psi> = α|0> + β|1>(任意のゲート)
        H(register[psi]);

        // 任意の回転
        let theta = DegToRad(90.0);
        R1(theta, register[psi]);
        
        // 現在の送信したい量子ビットを表示
        DumpRegister((), register[psi..psi]);
        //|0⟩      0.707107 +  0.000000 i  ==     ***********          [ 0.500000 ]     --- [  0.00000 rad ]
        //|1⟩      0.000000 +  0.707107 i  ==     ***********          [ 0.500000 ]    ↑    [  1.57080 rad ]

        // 状態を送信
        // AとBのベル状態 1/√2 (|00> + |11>)
        H(register[alpha]);
        CNOT(register[alpha], register[beta]);

        CNOT(register[psi], register[alpha]);
        H(register[psi]);

        let ctrl_X = M(register[alpha]);
        let ctrl_Z = M(register[psi]);

        if (ctrl_X == One) {
            X(register[beta]);
        }

        if (ctrl_Z == One) {
            Z(register[beta]);
        }

        // 転送された結果を出力
        DumpRegister((), register[beta..beta]);
        //|0⟩      0.707107 +  0.000000 i  ==     ***********          [ 0.500000 ]     --- [  0.00000 rad ]
        //|1⟩      0.000000 +  0.707107 i  ==     ***********          [ 0.500000 ]    ↑    [  1.57080 rad ]

        ResetAll(register);
    }

    /// # SUmmary
    /// ゲートテレポーテーション
    /// # Input
    /// - q (Qubit): 適当な量子ビット
    /// - U ((Qubit) => Unit): 適当な量子ゲート処理
    operation GateTeleport(q: Qubit, U: Qubit => Unit is Adj + Ctl) : Unit {
        use aux = Qubit();

        H(aux);
        CNOT(aux, q);

        U(aux);

        // Controlled Unitary
        if (M(q) == One) {
            U(aux);
            X(aux);
            Adjoint U(aux);
        }

        Reset(aux);
    }

    operation GateTeleportMain() : Unit {
        // ∣ψ⟩:=α∣0⟩+β∣1⟩
        use A = Qubit();

        // 適当な量子ビットAを定義
        H(A);

        DumpRegister((), [A]);

        GateTeleport(A, T);

        DumpMachine();

        Reset(A);
    }

    operation QuantumMain() : Unit {
        use register = Qubit[3];

        ApplyToEach(X, register);

        DumpMachine();
        Message("==");

        ApplyToEach(H, [register[0], register[1]]);

        DumpMachine();
        Message("==");

        CCNOT(register[2], register[1], register[0]);

        DumpMachine();
        Message("==");

        H(register[0]);

        DumpMachine();
        Message("==");

        let result = M(register[0]);
        Message($"result: {result}");

        ResetAll(register);
    }

    operation ConvertToKetMinus(): Unit{
        use q = Qubit(); // |0>

        DumpMachine();
        Message($"---");

        X(q); // |1>

        DumpMachine();
        Message($"----");

        H(q); // |->

        DumpMachine();
        Message($"----");

        Reset(q);
    }

    operation TargetUnitary(register: Qubit[]): Unit is Adj + Ctl{
        ApplyToEachCA(X, register);
        ApplyToEachCA(Z, register);
        ApplyToEachCA(I, register);
    }

    /// # Summary
    /// アダマールテスト
    operation HadamardTestMain(): Unit {
        use register = Qubit[3];

        H(register[0]);

        Controlled TargetUnitary([register[0]], register[1...]);
        DumpMachine();

        H(register[0]);

        let m = M(register[0]);
        ResetAll(register);

        Message($"m = {m}");
    }

    /// # Summary
    /// 量子論理演算の実装
    operation IfThenInvert(): Unit {
        use (a, b, c) = (Qubit(), Qubit(), Qubit());

        X(a);
        H(b);
        CCNOT(a, b, c);

        DumpMachine();
        Message($"C={M(c)}");

        ResetAll([a, b, c]);
    }

    // # Summary
    // forループで利用する配列のMaxインデックスを取得
    function MaxRangeIndex<'T>(register: 'T[]): Int {
        return Length(register) - 1;
    }

    operation ControlledAreaNot(register: Qubit[], area: Range): Unit is Adj + Ctl {
        for idx in area {
            Controlled X(register[...(idx-1)], register[idx]);
        }
    }

    // # Summary
    // 量子インクリメント処理の前処理
    //
    // # Input
    // - register (Qubit[]): インクリメント処理を行いたいQubit配列を指定
    // - minIndex (Int): 0以上、Length(register)より小さい数を指定
    //
    operation PreIncrement(register: Qubit[], minIndex: Int): Unit is Adj + Ctl {
        let maxIndex = Length(register) - 1;
        ControlledAreaNot(register, maxIndex..-1..minIndex);
    }

    // # Summary
    // 量子インクリメント処理
    operation Increment(register: Qubit[]): Unit is Adj + Ctl {
        PreIncrement(register, 1);
        X(register[0]);
    }

    // # Summary
    // 量子デクリメント処理の前処理
    //
    // # Input
    // - register (Qubit[]): デクリメント処理を行いたいQubit配列を指定
    // - minIndex (Int): 0以上、Length(register)より小さい数を指定
    //
    operation PreDecrement(register: Qubit[], minIndex: Int): Unit is Adj + Ctl {
        let maxIndex = Length(register) - 1;
        ControlledAreaNot(register, minIndex..maxIndex);
    }

    // # Summary
    // 量子デクリメント処理
    operation Decrement(register: Qubit[]): Unit is Adj + Ctl {
        X(register[0]);
        PreDecrement(register, 1);
    }

    operation NibbleMain(): Unit{
        let nibbleSize = 4;
        let prepareAngle = DegToRad(45.0);

        use qunibble = Qubit[nibbleSize];

        // prepare
        X(qunibble[0]);
        H(qunibble[2]);
        R1(prepareAngle, qunibble[2]);

        DumpRegister((), qunibble);
        Message("");

        // Increment(qunibble);
        Decrement(qunibble);

        DumpRegister((), qunibble);

        let results = MultiM(qunibble);
        Message($"results = {results}");

        ResetAll(qunibble);
    }

    /// # Summary
    /// 符号反転処理
    operation NegativeSign(qubit: Qubit[]): Unit is Adj + Ctl {
        ApplyToEachCA(X, qubit);
        Increment(qubit);
    }

    /// # Summary
    /// 加算代入処理(逆順序で減算代入)
    operation Addtion(a: Qubit[], b: Qubit[]): Unit is Adj + Ctl {
        let upperControl = Flattened([[b[0]], a]);
        PreIncrement(upperControl, 1);
        let bottomControl = Flattened([[b[1]], a[1...]]);
        PreIncrement(bottomControl, 1);
    }

    operation AddMain(): Unit {
        let prepareAAngle = DegToRad(45.0);
        let prepareBAngle = DegToRad(90.0);

        use aQunibble = Qubit[4];
        use bQunibble = Qubit[2];

        // prepare
        X(aQunibble[0]);
        H(aQunibble[2]);
        R1(prepareAAngle, aQunibble[2]);

        X(bQunibble[0]);
        H(bQunibble[1]);
        R1(prepareBAngle, bQunibble[1]);

        Addtion(aQunibble, bQunibble);

        DumpMachine();

        ResetAll(aQunibble);
        ResetAll(bQunibble);
    }

    @EntryPoint()
    operation IntegerMain(): Unit{
        let bitSize = 3;
        use a = Qubit[bitSize];
        use b = Qubit[bitSize];

        let aE = LittleEndian(a);
        let bE = LittleEndian(b);

        // |1> + |5>
        // X(a[0]);
        H(a[0]);

        // |1> + |3>
        // X(b[0]);
        // H(b[1]);

        // AddI(aE, bE);
        IncrementByInteger(5, aE);

        DumpRegister((), a);
        let result = MeasureInteger(aE);
        Message($"result = {result}");
        ResetAll(Flattened([a, b]));
    }

}

