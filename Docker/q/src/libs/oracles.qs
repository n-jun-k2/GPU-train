namespace Oracles {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;

    operation ApplyZeroOracle(control: Qubit, target: Qubit): Unit is Adj+Ctl {
    }

    operation ApplyOneOracle(control: Qubit, target: Qubit): Unit is Adj+Ctl {
        X(target);
    }

    operation ApplyIdOracle(control: Qubit, target: Qubit): Unit is Adj+Ctl {
        Controlled X([control], target);
    }

    operation ApplyNotOracle(control: Qubit, target: Qubit): Unit is Adj+Ctl {
        within {
            X(control);
        }
        apply {
            Controlled X([control], target);
        }
    }
}