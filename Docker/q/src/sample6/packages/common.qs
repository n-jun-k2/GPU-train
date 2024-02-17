namespace Common {
    open Microsoft.Quantum.Math as Math;

    function DegToRad(deg: Double) : Double {
        return deg * Math.PI() / 180.0;
    }
}