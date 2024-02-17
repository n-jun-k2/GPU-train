namespace sample7tests {
    
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Intrinsic;
    open Oracles;
    open sample7;

    @Test("QuantumSimulator")
    operation RunDeutschJozsaAlgorithm() : Unit {
        Fact(not CheckIfOracleIsBalanced(ApplyZeroOracle), "Test failed for zero oracle");
        Fact(not CheckIfOracleIsBalanced(ApplyOneOracle), "Test failed for one oracle");
        Fact(CheckIfOracleIsBalanced(ApplyIdOracle), "Test failed for id oracle");
        Fact(CheckIfOracleIsBalanced(ApplyNotOracle), "Test failed for not oracle");
    }
}
