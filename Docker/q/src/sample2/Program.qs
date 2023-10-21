namespace sample2 {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Arrays;

    /// # Summary
    /// 例外排出する為の練習
    ///
    /// # Input
    /// ## code (Int)
    /// 0の値を渡すと例外が排出される。
    ///
    /// # Output
    /// ## Unit 
    ///
    /// # Example
    /// ```Q#
    /// F(0); <- error
    /// F(1);
    /// code = 11
    /// ```
    ///
    function F(code: Int) : Unit {
        if code == 0 {
            fail $"ERROR : 0 == {code}";
        }

        let p = 10 + code;
        Message($"code = {p}");
    }

    /// # Summary
    /// 条件付きループの対応方法
    /// 
    /// # Input
    /// ## code (String)
    ///
    /// QS4001: 非推奨(repeat-until-success-loops)
    /// 
    function F2(count: Int) : Unit {
        mutable c = count;
        repeat {
            Message($"repeat {c}");
            set c -= 1;
            let d = 10;
        }
        until c < 1 and d == 10
        fixup {
            Message($"fixup {c} d = {d}");
        }
    }

    /// # Summary
    /// 条件付きループの練習
    /// 
    /// # Input
    /// ## arr (int[])
    /// 
    function F3(arr: Int[]) : Unit {
        mutable (item, idx) = (-1, 0);
        while idx < Length(arr) and item < 0 {
            set item = arr[idx];
            set idx += 1;
            Message($"item[{idx}] = {item}");
        }
    }

    /// # Summary
    /// 単なるループ
    /// 
    /// # Input
    /// ## arr (Int[]): 整数型の配列
    /// 
    function F4(arr: Int[]) : Unit {
        for item in arr {
            Message($"F4 : {item}");
        }

        for item in 0..Length(arr) - 1 {
            Message($"F4 : {arr[item]}");
        }

        // IndexRange <- open Microsoft.Quantum.Arrays;
        for item in IndexRange(arr) {
            Message($"F4 : {arr[item]}");
        }
    }

    @EntryPoint()
    operation Main() : Unit {
        F(10);
        F2(10);
        F2(0);
        F3([1, 2, 3]);
        F3([-1, -2, -3]);
        F3([]);

        F4([1, 2, 13]);
    }
}

