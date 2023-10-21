namespace sample1 {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Arrays;

    function IntFunc(): Unit{
        let b = 0b11110000;
        let o = 0o52;
        let d = 5;
        let h = 0xFF;
        Message($"IntFunc():{b}:{o}:{d}:{h}:{5+5}");
    }

    function BigIntFunc(): Unit {
        let bb = 0b1101010L;
        let bo = 0o32L;
        let bd = 3L;
        let bh = 0xFFFFL;
        Message($"BigIntFunc():{bb}:{bo}:{bd}:{bh}:{15L+5L}");
    }

    function StringFunc(): Unit {
        let a = "aaa";
        let b = "bbb";
        let c = a + b;
        Message($"StringFunc():--{a}--{b}--{c}--");
    }

    function BoolFunc(): Unit {
        let b = true;
        let f = false;
        Message($"BoolFunc():{b}/{f}");
    }

    function ArrayFunc(): Unit{
        let a = [1, 2, 3];
        let b = [2.1, 3.1];
        let c = [a, [4, 5, 6]];
        let d = [5.5, size=3];
        // let d = [a, c]; error
        Message($"ArrayFunc():{a}/{b}/{["aiueo", "aiueo"]} {c} {d}");
    }

    function RangeFunc(): Unit {
        let a = 1..3;
        let b = 1..2..10;
        let c = 2..2..5;
        Message($"RangeFunc():{a}/{b}/{c}");
    }

    function TupleFunc(): Unit {
        let a = (1, 0.5, "f");
        let b = ("fabo","tama", 33);

        let v1 = 2;
        let v2 = 0.3;
        let v3 = "c";
        
        let c = (v3, (v2, v1));

        Message($"TupleFunc(): {a} / {b} / {c} / {c}");
    }

    function UpdateVariableFunc(): Unit {
        mutable a = 10;
        set a = 5;
        // set a w/=0 <- 4;

        mutable b = 0.0;
        for i in 1..2..10 {
            set b += 1.0;
        }

        mutable arr1 = [0, size = 3];
        set arr1 w/=0 <- 10;
        set arr1 w/=1 <- 2;

        mutable arr2 = [1.0, size=2];
        for i in IndexRange(arr2) {
            // set arr2 w/= i <- 2; Error
            set arr2 w/= i <- 2.0;
        }

        mutable tuple1 = (1, 2, "a");
        set tuple1 = (2, 4, "b");

        Message($"UpdateVariableFunc(): {a} / {b} / {arr1} {arr2} {tuple1}");
    }

    newtype NamedType = (Integer: Int, Data: Double[]);
    newtype NonNamedType = (Double, (Int, Int, String), Int);

    function UserTypeFunc(): Unit {
        mutable original = NamedType(10, [5.0, size=3]);
        Message($"UserTypeFunc(): {original}");
        set original w/= Integer <- 8;
        set original w/= Data <- [4.0, size=2];
        mutable a = original::Data;
        set a w/= 0 <- 5.0;
        Message($"UserTypeFunc(): {original} {original::Data}");
    }

    function UserType2Func(): Unit {
        mutable original = NonNamedType(1.1, (5, 5, "STRING"), 5);
        let (head, _, tail) = original!;
        Message($"UserType2Func(): {head}");
    }

    @EntryPoint()
    operation execute() : Unit {
        IntFunc();
        BigIntFunc();
        StringFunc();
        BoolFunc();
        ArrayFunc();
        RangeFunc();
        TupleFunc();
        UpdateVariableFunc();
        UserTypeFunc();
        UserType2Func();
    }
}

