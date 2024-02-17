namespace sample8 {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Convert as Convert;
    open Microsoft.Quantum.Measurement as Meas;
    open Microsoft.Quantum.Arrays as Arrays;
    open Microsoft.Quantum.Diagnostics as Diag;

    newtype ScaleRotation = ((Double, Qubit) => Unit is Adj+Ctl);

    operation ApplyScaleRotationCA(angle: Double, scale: Double, target: Qubit): Unit is Adj+Ctl {
        R1(angle * scale, target);
    }

    operation RunsloatFlow(numMeas: Int, scale: Double, rotaion: ScaleRotation): Double {
        mutable oneCounter = 0;

        use q = Qubit();
        for _ in 0..numMeas - 1 {
            within {
                H(q);
            } apply {
                rotaion!(scale, q);
            }
            let result = Meas.MResetZ(q);
            set oneCounter +=  Convert.ResultArrayAsInt([result]);
        }
        return Convert.IntAsDouble(oneCounter) / Convert.IntAsDouble(numMeas);
    }

    @EntryPoint()
    operation RunGame(hiddenAngle: Double, scales: Double[], perScale: Int) : Double[] {
        Message("Hello quantum world!");

        let rotaion = ScaleRotation(ApplyScaleRotationCA(hiddenAngle, _, _));
        let action = RunsloatFlow(perScale, _, rotaion);
        let results = Arrays.ForEach(action, scales);
        return results;
    }
}
